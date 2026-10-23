"""PROSE：过程语义历史动力学。

形状约定：B 为批次大小，L 为窗口长度，d 为变量数，k 为语义维度，h 为记忆通道数。

    u_{i,t}   = [x_{i,t}, dx_{i,t}]                                  (B, L, d, 2)
    v_{j,t}   = tanh(Q u_{j,t})                                      (B, L, d, k)
    C_t       = sum_j e^C_j * v_{j,t}                                (B, L, k)
    c^{-i}_t  = (C_t - e^C_i * v_{i,t}) / (d - 1)                    (B, L, d, k)
    eta_{i,t} = A_i u_{i,t} + beta_i + k W_o (e^H_i * c^{-i}_t)      (B, L, d, h)
    a = sigmoid(eta),  delta = softplus(-eta),  Theta = cumsum(delta)
    m_{i,t}   = a m_{i,t-1} + (1 - a) tanh(W_i u_{i,t} + c_i)
    z_{i,t}   = x_{i,t} + w_i^T m_{i,t}

常数 k 用于维度校正。对单位向量 e^H_i、e^C_j，逐维乘积 e^H_i * e^C_j 的分量为 O(1/k)；
乘以 k 后各分量为 O(1)，且在 k 个维度上的均值等于两视角的余弦相似度。
不作校正时，W_o 的跨变量输入约为 A_i 局部输入的 1e-2；AdamW 的步长不随输入尺度缩放，
训练后条件语义场的量级仅约为局部项的 1e-3–1e-2。

跨变量项按耦合族形式计算：k W_o (e^H_i * c^{-i}_t) = sum_{j != i} M_ij v_{j,t} / (d - 1)，
M_ij = W_o Diag(p_ij)，p_ij = k (e^H_i * e^C_j)。全部 M_ij 拼成 (d h, d k) 的块矩阵，
跨变量项即一次矩阵乘法，不生成 (B, L, d, k) 的逐元素中间量。

结构开关（默认 coupling="semantic"、local_term=True，即完整 PROSE）：
    coupling = "semantic" | "free" | "none"
        semantic：由冻结语义生成跨变量项（完整 PROSE）；
        free：保持相同结构，将缩放后的语义乘积 k (e^H_i * e^C_j)
              替换为每对变量独立的可学习向量（数据门控），初始化时保持相同的 O(1) 分量尺度；
        none：去掉跨变量项，即去掉条件语义场。
    local_term = False 将 A_i 固定为 0。
    coupling="none" 且 local_term=False 时，退化为按时间索引的指数核。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

COUPLINGS = ("semantic", "free", "none")
SCANS = ("parallel", "recurrent")


@dataclass(frozen=True)
class ProseOutput:
    z: Tensor  # (B, L, d) 语义化历史表示
    eta: Tensor  # (B, L, d, h) 存续门预激活值
    cross: Tensor  # (B, L, d, h) eta 的跨变量项
    survival: Tensor  # (B, L, d, h) a
    process_time: Tensor  # (B, L, d, h) Theta
    memory: Tensor  # (B, L, d, h) m


def timescale_bias(window_size: int, memory_dim: int) -> Tensor:
    """beta_{i,c} = logit(exp(-1/tau_c))，tau_c 在 [1, L] 上按对数均匀分布取值。"""

    tau = torch.logspace(0.0, math.log10(max(window_size, 1)), memory_dim, dtype=torch.float64)
    return torch.logit(torch.exp(-1.0 / tau)).float()


def history_weights(eta: Tensor) -> Tensor:
    """对 s <= t，omega(s, t) = (1 - a_s) exp(-(Theta_t - Theta_s))。

    eta 的形状为 (..., L, h)，返回结果的形状为 (..., L_t, L_s, h)。
    """

    log_write = -F.softplus(eta)
    theta = torch.cumsum(F.softplus(-eta), dim=-2)
    gap = theta.unsqueeze(-2) - theta.unsqueeze(-3)  # (..., t, s, h)
    length = eta.shape[-2]
    future = torch.ones(length, length, dtype=torch.bool, device=eta.device).triu(1).unsqueeze(-1)
    return torch.exp(log_write.unsqueeze(-3) - gap.masked_fill(future, float("inf")))  # s > t 的权重为 0


def parallel_memory(eta: Tensor, written: Tensor) -> Tensor:
    """沿第 1 维计算 m_t = sum_{s<=t} (1 - a_s) exp(-(Theta_t - Theta_s)) written_s。

    使用输入的数据类型，在对数空间中计算前缀和。写入值拆为两个非负部分，
    分别加上公共偏置 1，再在相减时精确抵消偏置。
    float32 的绝对误差随过程时间 Theta 增大：Theta ~ 1e2 时约为 1e-5，
    Theta ~ 1e3 时低于 1e-3。
    """

    theta = torch.cumsum(F.softplus(-eta), dim=1)
    log_gain = theta - F.softplus(eta)
    positive = torch.logcumsumexp(log_gain + torch.log1p(written.clamp_min(0.0)), dim=1)
    negative = torch.logcumsumexp(log_gain + torch.log1p((-written).clamp_min(0.0)), dim=1)
    return torch.exp(positive - theta) - torch.exp(negative - theta)


def recurrent_memory(eta: Tensor, written: Tensor) -> Tensor:
    survival = torch.sigmoid(eta)
    state = torch.zeros_like(written[:, 0])
    outputs = []
    for step in range(written.shape[1]):
        state = survival[:, step] * state + (1.0 - survival[:, step]) * written[:, step]
        outputs.append(state)
    return torch.stack(outputs, dim=1)


class ProseAdapter(nn.Module):
    def __init__(
        self,
        *,
        history_semantics: np.ndarray,
        condition_semantics: np.ndarray,
        window_size: int,
        memory_dim: int = 16,
        coupling: str = "semantic",
        local_term: bool = True,
        scan: str = "parallel",
    ) -> None:
        super().__init__()
        if coupling not in COUPLINGS:
            raise ValueError(f"coupling must be one of {COUPLINGS}")
        if scan not in SCANS:
            raise ValueError(f"scan must be one of {SCANS}")
        e_h = np.asarray(history_semantics, dtype=np.float64)
        e_c = np.asarray(condition_semantics, dtype=np.float64)
        if e_h.ndim != 2 or e_h.shape != e_c.shape or e_h.shape[0] < 2:
            raise ValueError("semantics must be two (d, k) arrays with d >= 2")
        d, k = e_h.shape
        h = int(memory_dim)
        self.num_variables, self.semantic_dim, self.memory_dim = d, k, h
        self.coupling, self.local_term, self.scan = coupling, bool(local_term), scan

        if coupling != "none":  # 没有跨变量项时，不创建 Q 与 W_o
            self.q = nn.Linear(2, k, bias=False)  # 共享映射 Q
            self.w_o = nn.Parameter(torch.zeros(h, k))  # 共享映射 W_o
        if coupling == "semantic":  # 冻结的 p_ij = k (e^H_i * e^C_j)，对角线为零
            pairs = k * e_h[:, None, :] * e_c[None, :, :] * (1.0 - np.eye(d))[:, :, None]
            self.register_buffer("semantic_pairs", torch.as_tensor(pairs, dtype=torch.float32))
        if coupling == "free":  # 每个有序变量对 i != j 对应一个自由向量
            rows, cols = torch.nonzero(1 - torch.eye(d, dtype=torch.long), as_tuple=True)
            self.register_buffer("pair_rows", rows)
            self.register_buffer("pair_cols", cols)
            self.pair_coupling = nn.Parameter(torch.randn(d * (d - 1), k))  # 替代 k (e^H_i * e^C_j)
        self.local = nn.Parameter(torch.zeros(d, h, 2)) if self.local_term else None  # A_i
        self.beta = nn.Parameter(timescale_bias(window_size, h).expand(d, h).clone())
        bound = 1.0 / math.sqrt(2.0)
        self.write_weight = nn.Parameter(torch.empty(d, h, 2).uniform_(-bound, bound))  # W_i
        self.write_bias = nn.Parameter(torch.empty(d, h).uniform_(-bound, bound))  # c_i
        self.readout = nn.Parameter(torch.zeros(d, h))  # w_i 零初始化，使 Z = X

    def condition_features(self, x: Tensor) -> Tensor:
        delta = torch.diff(x, dim=1, prepend=x[:, :1])
        return torch.stack((x, delta), dim=-1)

    def pair_vectors(self) -> Tensor:
        """形状为 (d, d, k) 的 p_ij，对角线为零：语义耦合为冻结的 k (e^H_i * e^C_j)，自由耦合为可学习向量。"""

        if self.coupling == "semantic":
            return self.semantic_pairs
        d, k = self.num_variables, self.semantic_dim
        pairs = self.pair_coupling.new_zeros(d, d, k)
        return pairs.index_put((self.pair_rows, self.pair_cols), self.pair_coupling)

    def coupling_matrices(self) -> Tensor:
        """形状为 (d, d, h, k) 的 M_ij = W_o Diag(p_ij)。"""

        return self.pair_vectors().unsqueeze(2) * self.w_o

    def cross_term(self, u: Tensor) -> Tensor:
        """sum_{j != i} M_ij tanh(Q u_j) / (d - 1)，形状为 (B, L, d, h)。"""

        d, k, h = self.num_variables, self.semantic_dim, self.memory_dim
        if self.coupling == "none":
            return u.new_zeros((*u.shape[:3], h))
        v = torch.tanh(self.q(u)).flatten(-2)  # (B, L, d k)，按 (j, k) 展平
        blocks = self.coupling_matrices().permute(0, 2, 1, 3).reshape(d * h, d * k) / (d - 1)
        return (v @ blocks.T).unflatten(-1, (d, h))

    def forward(self, x: Tensor, *, return_components: bool = False) -> Tensor | ProseOutput:
        if x.shape[-1] != self.num_variables:
            raise ValueError(f"expected {self.num_variables} variables, got {x.shape[-1]}")
        u = self.condition_features(x)
        cross = self.cross_term(u)
        eta = self.beta + cross
        if self.local is not None:
            eta = eta + torch.einsum("bldc,dhc->bldh", u, self.local)
        written = torch.tanh(torch.einsum("bldc,dhc->bldh", u, self.write_weight) + self.write_bias)
        memory = parallel_memory(eta, written) if self.scan == "parallel" else recurrent_memory(eta, written)
        z = x + torch.einsum("bldh,dh->bld", memory, self.readout)
        if not return_components:
            return z
        return ProseOutput(
            z=z,
            eta=eta,
            cross=cross,
            survival=torch.sigmoid(eta),
            process_time=torch.cumsum(F.softplus(-eta), dim=1),
            memory=memory,
        )

    @torch.no_grad()
    def coupling_norms(self) -> Tensor:
        """||M_ij||_F，形状为 (d, d)，对角线为零。"""

        if self.coupling == "none":
            return self.beta.new_zeros(self.num_variables, self.num_variables)
        return torch.linalg.matrix_norm(self.coupling_matrices())
