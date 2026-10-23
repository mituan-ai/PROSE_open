"""8 个预测主干：输入 (B, L, d)，输出 (B, output_dim) 的窗口表示，由线性输出头映射为预测。

超参数全部来自 ``configs/backbones.yaml``；``build_backbone`` 只补充由数据决定的窗口长度。
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GRUBackbone(nn.Module):
    """用于窗口级预测的 GRU 编码器。"""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_positive("input_dim", input_dim)
        _validate_positive("hidden_size", hidden_size)
        _validate_positive("num_layers", num_layers)
        effective_dropout = float(dropout) if num_layers > 1 else 0.0
        self.input_dim = int(input_dim)
        self.output_dim = int(hidden_size)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        _require_sequence_inputs(inputs, "GRUBackbone")
        _, hidden = self.gru(inputs)
        return hidden[-1]


class LSTMBackbone(nn.Module):
    """与 GRUBackbone 采用相同对外接口的 LSTM 编码器。"""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_positive("input_dim", input_dim)
        _validate_positive("hidden_size", hidden_size)
        _validate_positive("num_layers", num_layers)
        effective_dropout = float(dropout) if num_layers > 1 else 0.0
        self.input_dim = int(input_dim)
        self.output_dim = int(hidden_size)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        _require_sequence_inputs(inputs, "LSTMBackbone")
        _, (hidden, _) = self.lstm(inputs)
        return hidden[-1]


class TransformerBackbone(nn.Module):
    """对序列窗口沿时间轴编码的 Transformer，仅使用编码器。"""

    def __init__(
        self,
        *,
        input_dim: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        window_size: int,
    ) -> None:
        super().__init__()
        _validate_transformer_args(input_dim, d_model, num_layers, num_heads, d_ff, window_size)
        self.input_dim = int(input_dim)
        self.output_dim = int(d_model)
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, window_size, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        _init_parameter(self.position_embedding, std=0.02)

    def forward(self, inputs: Tensor) -> Tensor:
        _require_sequence_inputs(inputs, "TransformerBackbone")
        if inputs.shape[1] > self.position_embedding.shape[1]:
            raise ValueError("TransformerBackbone input window exceeds configured window_size")
        projected = self.input_projection(inputs)
        encoded = self.encoder(projected + self.position_embedding[:, : inputs.shape[1], :])
        return self.norm(encoded[:, -1, :])


class InformerBackbone(nn.Module):
    """HuggingFace Informer 编码器的轻量封装。"""

    def __init__(
        self,
        *,
        input_dim: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        distil: bool = True,
        sampling_factor: int = 5,
        window_size: int,
    ) -> None:
        super().__init__()
        _validate_transformer_args(input_dim, d_model, num_layers, num_heads, d_ff, window_size)
        _validate_positive("sampling_factor", sampling_factor)
        try:
            from transformers import InformerConfig, InformerModel
        except ImportError as exc:  # pragma: no cover - 依赖可选依赖组。
            raise ImportError(
                "Informer backbone requires transformers. "
                "Install the train extra or add transformers before running Informer experiments."
            ) from exc
        self.input_dim = int(input_dim)
        self.output_dim = int(d_model)
        self.window_size = int(window_size)
        self.context_length = int(window_size) - 1  # lags_sequence=[1] 另需一步历史
        self.time_feature_size = 1
        config = InformerConfig(
            input_size=input_dim,
            prediction_length=1,
            context_length=self.context_length,
            lags_sequence=[1],
            d_model=d_model,
            encoder_layers=num_layers,
            decoder_layers=1,
            encoder_attention_heads=num_heads,
            decoder_attention_heads=num_heads,
            encoder_ffn_dim=d_ff,
            decoder_ffn_dim=d_ff,
            dropout=float(dropout),
            attention_dropout=float(dropout),
            activation_dropout=float(dropout),
            num_time_features=self.time_feature_size,
            num_dynamic_real_features=0,
            num_static_real_features=0,
            num_static_categorical_features=0,
            distil=bool(distil),
            sampling_factor=int(sampling_factor),
            scaling=None,  # 输入已按训练集统计量标准化；官方 Informer 不做实例缩放
        )
        self.model = InformerModel(config)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, inputs: Tensor) -> Tensor:
        _require_sequence_inputs(inputs, "InformerBackbone")
        if inputs.shape[1] != self.window_size:
            raise ValueError(f"InformerBackbone expects window {self.window_size}, got {inputs.shape[1]}")
        batch_size, sequence_length, _ = inputs.shape
        past_time_features = torch.zeros(
            batch_size,
            sequence_length,
            self.time_feature_size,
            dtype=inputs.dtype,
            device=inputs.device,
        )
        future_time_features = torch.zeros(
            batch_size,
            1,
            self.time_feature_size,
            dtype=inputs.dtype,
            device=inputs.device,
        )
        outputs = self.model(
            past_values=inputs,
            past_time_features=past_time_features,
            past_observed_mask=torch.ones_like(inputs),
            future_time_features=future_time_features,
        )
        return self.norm(outputs.last_hidden_state[:, -1, :])


class MambaBackbone(nn.Module):
    """HuggingFace Mamba 序列模型的轻量封装。"""

    def __init__(
        self,
        *,
        input_dim: int,
        d_model: int,
        num_layers: int,
        state_size: int,
        conv_kernel: int,
        expand: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_positive("input_dim", input_dim)
        _validate_positive("d_model", d_model)
        _validate_positive("num_layers", num_layers)
        _validate_positive("state_size", state_size)
        _validate_positive("conv_kernel", conv_kernel)
        _validate_positive("expand", expand)
        try:
            from transformers import MambaConfig, MambaModel
        except ImportError as exc:  # pragma: no cover - 依赖可选依赖组。
            raise ImportError(
                "Mamba backbone requires transformers with MambaModel support. "
                "Install the train extra or add transformers before running Mamba experiments."
            ) from exc
        from prose.models.mamba_kernels import enable_hf_mamba_kernels

        self.fast_path = enable_hf_mamba_kernels()  # 为 False 时回退到 HF 串行扫描
        self.input_dim = int(input_dim)
        self.output_dim = int(d_model)
        self.input_projection = nn.Linear(input_dim, d_model)
        self.model = MambaModel(
            MambaConfig(
                vocab_size=1,
                hidden_size=d_model,
                state_size=state_size,
                num_hidden_layers=num_layers,
                conv_kernel=conv_kernel,
                expand=expand,
                intermediate_size=d_model * expand,
            )
        )
        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, inputs: Tensor) -> Tensor:
        _require_sequence_inputs(inputs, "MambaBackbone")
        projected = self.dropout(self.input_projection(inputs))
        outputs = self.model(inputs_embeds=projected)
        return self.norm(outputs.last_hidden_state[:, -1, :])


class ITransformerBackbone(nn.Module):
    """iTransformer（Liu 等，ICLR 2024），按官方实现（thuml/iTransformer）。

    每个变量的整段历史经倒置嵌入 ``Linear(L, d_model)`` 成为一个 token，没有位置或变量编码；
    编码器为后归一化的注意力层与 GELU 前馈层，末端 LayerNorm。输出按官方分类头取
    GELU、dropout 后展平的各变量表示。输入已按训练集统计量标准化，因此不做实例归一化。
    """

    def __init__(
        self,
        *,
        input_dim: int,
        window_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        _validate_transformer_args(input_dim, d_model, num_layers, num_heads, d_ff, window_size)
        self.input_dim = int(input_dim)
        self.window_size = int(window_size)
        self.output_dim = self.input_dim * int(d_model)
        self.embedding = nn.Linear(self.window_size, d_model)
        self.embedding_dropout = nn.Dropout(float(dropout))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False
        )
        self.output_dropout = nn.Dropout(float(dropout))

    def forward(self, inputs: Tensor) -> Tensor:
        _require_sequence_inputs(inputs, "ITransformerBackbone")
        if inputs.shape[1] != self.window_size:
            raise ValueError(f"ITransformerBackbone expects window {self.window_size}, got {inputs.shape[1]}")
        tokens = self.embedding_dropout(self.embedding(inputs.transpose(1, 2)))  # (B, d, d_model)
        return self.output_dropout(F.gelu(self.encoder(tokens))).flatten(1)


class PatchTSTBackbone(nn.Module):
    """PatchTST（Nie 等，ICLR 2023），调用 HuggingFace ``PatchTSTModel``。

    取官方监督设置：各变量独立分块并共享权重，BatchNorm、后归一化，注意力权重不加 dropout；
    位置编码用库提供的固定正弦编码（官方的可学习编码在库中以标准差 1 初始化，量级过大）。
    输入已按训练集统计量标准化，因此不做实例归一化（``scaling=None``）。库在窗口开头舍去不足
    一个步长的时间步，使最后一个分块以当前时刻结束。输出按 Time-Series-Library 分类头取
    展平的各变量分块表示并加 dropout。
    """

    def __init__(
        self,
        *,
        input_dim: int,
        window_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        patch_len: int,
        stride: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        _validate_transformer_args(input_dim, d_model, num_layers, num_heads, d_ff, window_size)
        from transformers import PatchTSTConfig, PatchTSTModel

        self.input_dim = int(input_dim)
        self.window_size = int(window_size)
        self.model = PatchTSTModel(
            PatchTSTConfig(
                num_input_channels=self.input_dim,
                context_length=self.window_size,
                patch_length=int(patch_len),
                patch_stride=int(stride),
                num_hidden_layers=int(num_layers),
                d_model=int(d_model),
                num_attention_heads=int(num_heads),
                ffn_dim=int(d_ff),
                norm_type="batchnorm",
                pre_norm=False,
                share_embedding=True,
                channel_attention=False,
                positional_encoding_type="sincos",
                use_cls_token=False,
                attention_dropout=0.0,
                positional_dropout=float(dropout),
                path_dropout=float(dropout),
                ff_dropout=float(dropout),
                scaling=None,
                do_mask_input=False,
            )
        )
        self.output_dim = self.input_dim * self.model.patchifier.num_patches * int(d_model)
        self.output_dropout = nn.Dropout(float(dropout))

    def forward(self, inputs: Tensor) -> Tensor:
        _require_sequence_inputs(inputs, "PatchTSTBackbone")
        hidden = self.model(past_values=inputs).last_hidden_state  # (B, d, P, d_model)
        return self.output_dropout(hidden.flatten(1))


class ModernTCNBackbone(nn.Module):
    """ModernTCN（Luo 和 Wang，ICLR 2024），按官方分类实现（luodhhh/ModernTCN）。

    每个变量的序列经分块 stem（``Conv1d(1, D, patch_size, patch_stride)`` 与 BatchNorm，
    末端复制最后一步补齐）嵌入；每个 stage 由若干块组成，块内依次为大核与小核两路深度卷积
    （各带 BatchNorm，训练时不合并）、BatchNorm、按变量分组的 ConvFFN1 与按特征分组的 ConvFFN2，
    并有残差连接；stage 之间以 BatchNorm 与步长卷积下采样。输出按官方分类头取 GELU、dropout
    后展平的全部特征。分类任务不使用 RevIN 与多尺度头，与官方分类设置一致。
    """

    def __init__(
        self,
        *,
        input_dim: int,
        window_size: int,
        patch_size: int,
        patch_stride: int,
        dims: list[int],
        num_blocks: list[int],
        large_size: list[int],
        small_size: list[int],
        ffn_ratio: int,
        downsample_ratio: int = 2,
        dropout: float = 0.1,
        class_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        stages = len(dims)
        if not stages or not len(num_blocks) == len(large_size) == len(small_size) == stages:
            raise ValueError("dims, num_blocks, large_size and small_size must list one value per stage")
        if any(small > large for small, large in zip(small_size, large_size, strict=True)):
            raise ValueError("small_size must not exceed large_size")
        self.input_dim = int(input_dim)
        self.patch_size, self.patch_stride = int(patch_size), int(patch_stride)
        self.downsample_ratio = int(downsample_ratio)
        self.downsample_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(1, dims[0], kernel_size=patch_size, stride=patch_stride),
                    nn.BatchNorm1d(dims[0]),
                )
            ]
            + [
                nn.Sequential(
                    nn.BatchNorm1d(dims[index]),
                    nn.Conv1d(
                        dims[index], dims[index + 1], kernel_size=downsample_ratio, stride=downsample_ratio
                    ),
                )
                for index in range(stages - 1)
            ]
        )
        self.stages = nn.ModuleList(
            nn.Sequential(
                *(
                    _ModernTCNBlock(
                        large_size=large_size[index],
                        small_size=small_size[index],
                        dmodel=dims[index],
                        dff=dims[index] * int(ffn_ratio),
                        nvars=self.input_dim,
                        drop=float(dropout),
                    )
                    for _ in range(num_blocks[index])
                )
            )
            for index in range(stages)
        )
        padded = int(window_size) + self.patch_size - self.patch_stride
        length = (padded - self.patch_size) // self.patch_stride + 1
        for _ in range(stages - 1):
            length = math.ceil(length / self.downsample_ratio)
        self.output_dim = self.input_dim * int(dims[-1]) * length
        self.class_dropout = nn.Dropout(float(class_dropout))

    def forward(self, inputs: Tensor) -> Tensor:
        _require_sequence_inputs(inputs, "ModernTCNBackbone")
        x = inputs.transpose(1, 2).unsqueeze(2)  # (B, M, 1, L)
        for index, (downsample, stage) in enumerate(zip(self.downsample_layers, self.stages, strict=True)):
            batch, variables, channels, length = x.shape
            x = x.reshape(batch * variables, channels, length)
            if index == 0 and self.patch_size != self.patch_stride:  # stem：末端复制最后一步
                x = torch.cat((x, x[:, :, -1:].repeat(1, 1, self.patch_size - self.patch_stride)), dim=-1)
            elif index > 0 and length % self.downsample_ratio:
                x = torch.cat(
                    (x, x[:, :, -(self.downsample_ratio - length % self.downsample_ratio) :]), dim=-1
                )
            x = downsample(x)
            x = stage(x.reshape(batch, variables, *x.shape[1:]))
        return self.class_dropout(F.gelu(x)).flatten(1)


def _conv_bn(channels: int, kernel_size: int) -> nn.Sequential:
    """官方 ``conv_bn``：无偏置的深度卷积（保持长度）后接 BatchNorm。"""

    return nn.Sequential(
        OrderedDict(
            conv=nn.Conv1d(
                channels, channels, kernel_size, padding=kernel_size // 2, groups=channels, bias=False
            ),
            bn=nn.BatchNorm1d(channels),
        )
    )


class _ModernTCNBlock(nn.Module):
    """官方 ``Block``：输入与输出形状均为 (B, M, D, N)。"""

    def __init__(
        self, *, large_size: int, small_size: int, dmodel: int, dff: int, nvars: int, drop: float
    ) -> None:
        super().__init__()
        channels = nvars * dmodel
        self.large_conv = _conv_bn(channels, large_size)
        self.small_conv = _conv_bn(channels, small_size)
        self.norm = nn.BatchNorm1d(dmodel)
        self.ffn1 = nn.Sequential(  # 按变量分组：同一变量的各特征之间混合
            nn.Conv1d(channels, nvars * dff, kernel_size=1, groups=nvars),
            nn.Dropout(drop),
            nn.GELU(),
            nn.Conv1d(nvars * dff, channels, kernel_size=1, groups=nvars),
            nn.Dropout(drop),
        )
        self.ffn2 = nn.Sequential(  # 按特征分组：同一特征的各变量之间混合
            nn.Conv1d(channels, nvars * dff, kernel_size=1, groups=dmodel),
            nn.Dropout(drop),
            nn.GELU(),
            nn.Conv1d(nvars * dff, channels, kernel_size=1, groups=dmodel),
            nn.Dropout(drop),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        batch, variables, channels, length = inputs.shape
        x = inputs.reshape(batch, variables * channels, length)
        x = self.large_conv(x) + self.small_conv(x)
        x = self.norm(x.reshape(batch * variables, channels, length))
        x = self.ffn1(x.reshape(batch, variables * channels, length))
        x = (
            x.reshape(batch, variables, channels, length)
            .transpose(1, 2)
            .reshape(batch, channels * variables, length)
        )
        x = self.ffn2(x)
        x = x.reshape(batch, channels, variables, length).transpose(1, 2)
        return inputs + x


BACKBONES: dict[str, type[nn.Module]] = {
    "gru": GRUBackbone,
    "lstm": LSTMBackbone,
    "transformer": TransformerBackbone,
    "informer": InformerBackbone,
    "mamba": MambaBackbone,
    "itransformer": ITransformerBackbone,
    "patchtst": PatchTSTBackbone,
    "moderntcn": ModernTCNBackbone,
}
_UNWINDOWED = ("gru", "lstm", "mamba")  # 其余主干的结构依赖窗口长度


def build_backbone(kind: str, *, input_dim: int, window_size: int, config: dict[str, Any]) -> nn.Module:
    """按配置构建主干；配置中出现主干不接受的超参数时直接报错。"""

    if kind not in BACKBONES:
        raise ValueError(f"Unsupported backbone: {kind}. Expected one of {tuple(BACKBONES)}")
    window = {} if kind in _UNWINDOWED else {"window_size": window_size}
    return BACKBONES[kind](input_dim=input_dim, **window, **config)


def _require_sequence_inputs(inputs: Tensor, module_name: str) -> None:
    if inputs.ndim != 3:
        raise ValueError(f"{module_name} inputs must have shape (batch, time, features)")


def _validate_positive(name: str, value: int) -> None:
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_transformer_args(
    input_dim: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    window_size: int,
) -> None:
    _validate_positive("input_dim", input_dim)
    _validate_positive("d_model", d_model)
    _validate_positive("num_layers", num_layers)
    _validate_positive("num_heads", num_heads)
    _validate_positive("d_ff", d_ff)
    _validate_positive("window_size", window_size)
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")


def _init_parameter(parameter: nn.Parameter, *, std: float) -> None:
    """标准差为 ``std`` 的均匀分布初始化。"""

    bound = std * math.sqrt(3.0)
    nn.init.uniform_(parameter, -bound, bound)
