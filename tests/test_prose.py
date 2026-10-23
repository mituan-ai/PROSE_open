"""检查 PROSE 的定义与性质。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from prose.methods.prose import ProseAdapter, history_weights, parallel_memory, recurrent_memory
from prose.semantic_artifact import center_normalize

D, K, H, L = 4, 6, 5, 12


def semantics() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    return center_normalize(rng.normal(size=(D, K))), center_normalize(rng.normal(size=(D, K)))


def make_adapter(**kwargs) -> ProseAdapter:
    history, condition = semantics()
    adapter = ProseAdapter(
        history_semantics=history,
        condition_semantics=condition,
        window_size=L,
        memory_dim=H,
        **kwargs,
    ).double()
    torch.manual_seed(0)
    with torch.no_grad():
        for name in ("w_o", "readout", "local"):
            if getattr(adapter, name, None) is not None:
                getattr(adapter, name).normal_(0.0, 0.5)
    return adapter


def inputs(batch: int = 3) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(batch, L, D, dtype=torch.float64)


def test_zero_readout_returns_the_original_window() -> None:
    adapter = ProseAdapter(
        history_semantics=center_normalize(np.random.default_rng(0).normal(size=(D, K))),
        condition_semantics=center_normalize(np.random.default_rng(1).normal(size=(D, K))),
        window_size=L,
        memory_dim=H,
    )
    x = torch.randn(2, L, D)
    assert torch.equal(adapter(x), x)


def test_parameter_count_matches_the_definition() -> None:
    adapter = ProseAdapter(
        history_semantics=np.eye(D, K), condition_semantics=np.eye(D, K), window_size=L, memory_dim=H
    )
    assert sum(parameter.numel() for parameter in adapter.parameters()) == K * (2 + H) + 7 * H * D


@pytest.mark.parametrize(
    "coupling, local_term, expected",
    [
        ("semantic", True, K * (2 + H) + 7 * H * D),
        ("free", True, K * (2 + H) + 7 * H * D + D * (D - 1) * K),
        ("none", True, 7 * H * D),
        ("none", False, 5 * H * D),
    ],
)
def test_variant_parameter_counts(coupling: str, local_term: bool, expected: int) -> None:
    adapter = ProseAdapter(
        history_semantics=np.eye(D, K),
        condition_semantics=np.eye(D, K),
        window_size=L,
        memory_dim=H,
        coupling=coupling,
        local_term=local_term,
    )
    assert sum(parameter.numel() for parameter in adapter.parameters()) == expected


def test_initial_process_clocks_run_at_multi_timescale_rates() -> None:
    adapter = ProseAdapter(
        history_semantics=np.eye(D, K), condition_semantics=np.eye(D, K), window_size=L, memory_dim=H
    )
    rate = torch.nn.functional.softplus(-adapter.beta[0]).double()
    tau = torch.logspace(0.0, np.log10(L), H, dtype=torch.float64)
    assert torch.allclose(rate, 1.0 / tau, atol=1e-6)


@pytest.mark.parametrize("coupling", ["semantic", "free", "none"])
def test_parallel_scan_equals_the_recurrence(coupling: str) -> None:
    adapter = make_adapter(coupling=coupling)
    out = adapter(inputs(), return_components=True)
    assert torch.allclose(
        out.memory,
        recurrent_memory(
            out.eta,
            torch.tanh(
                torch.einsum("bldc,dhc->bldh", adapter.condition_features(inputs()), adapter.write_weight)
                + adapter.write_bias
            ),
        ),
        atol=1e-10,
    )


@pytest.mark.parametrize("scale, shift, tolerance", [(1.0, 0.0, 5e-5), (4.0, -3.0, 1e-3)])
def test_parallel_scan_accuracy_in_float32(scale: float, shift: float, tolerance: float) -> None:
    torch.manual_seed(2)
    eta = torch.randn(4, 120, 3, 8) * scale + shift  # shift < 0 时过程时钟较快，Theta ~ 6e2
    written = torch.tanh(torch.randn(4, 120, 3, 8))
    reference = recurrent_memory(eta.double(), written.double())
    assert (parallel_memory(eta, written).double() - reference).abs().max() < tolerance


def test_history_weights_are_bounded_and_reproduce_memory() -> None:
    adapter = make_adapter()
    out = adapter(inputs(), return_components=True)
    weights = history_weights(out.eta.permute(0, 2, 1, 3))  # (B, d, t, s, h)
    assert torch.all(weights.sum(dim=-2) <= 1.0 + 1e-12)
    written = out.memory.new_tensor(
        np.tanh(
            np.einsum(
                "bldc,dhc->bldh",
                adapter.condition_features(inputs()).numpy(),
                adapter.write_weight.detach().numpy(),
            )
            + adapter.write_bias.detach().numpy()
        )
    )
    rebuilt = torch.einsum("bdtsh,bdsh->btdh", weights, written.permute(0, 2, 1, 3))
    assert torch.allclose(rebuilt, out.memory, atol=1e-10)
    assert out.memory.abs().max() <= 1.0


def test_time_index_variant_is_an_exponential_kernel() -> None:
    adapter = make_adapter(coupling="none", local_term=False)
    out = adapter(inputs(), return_components=True)
    rate = torch.nn.functional.softplus(-adapter.beta)  # (d, h)
    weights = history_weights(out.eta.permute(0, 2, 1, 3))[0]  # (d, t, s, h)
    gap = (torch.arange(L)[:, None] - torch.arange(L)[None, :]).double()
    kernel = (1 - torch.exp(-rate))[:, None, None, :] * torch.exp(
        -rate[:, None, None, :] * gap[None, :, :, None]
    )
    kernel = kernel * torch.tril(torch.ones(L, L, dtype=torch.float64))[None, :, :, None]
    assert torch.allclose(weights, kernel, atol=1e-12)


def test_cross_term_is_the_semantic_coupling_family() -> None:
    """跨变量项等于原式 k W_o (e^H_i * c^{-i})，也等于 sum_{j != i} M_ij v_j / (d - 1)。"""

    adapter = make_adapter()
    e_h, e_c = (torch.as_tensor(array) for array in semantics())
    u = adapter.condition_features(inputs())
    v = torch.tanh(adapter.q(u))
    cross = adapter.cross_term(u)
    field = torch.einsum("bljk,jk->blk", v, e_c)
    others = (field.unsqueeze(2) - v * e_c) / (D - 1)
    # 冻结语义以 float32 存储，因此与 float64 原式的差约为 1e-7
    assert torch.allclose(cross, K * (others * e_h) @ adapter.w_o.T, atol=1e-6)
    expected = torch.zeros_like(cross)
    for i in range(D):
        for j in range(D):
            if i != j:
                m_ij = adapter.w_o @ torch.diag(K * e_h[i] * e_c[j])
                expected[:, :, i] += v[:, :, j] @ m_ij.T / (D - 1)
                norm = torch.linalg.matrix_norm(m_ij)
                assert torch.isclose(adapter.coupling_norms()[i, j], norm, atol=1e-6)
    assert torch.allclose(cross, expected, atol=1e-6)
    assert torch.all(adapter.coupling_norms().diagonal() == 0)


def test_free_cross_term_uses_the_free_pair_vectors() -> None:
    adapter = make_adapter(coupling="free")
    u = adapter.condition_features(inputs())
    v = torch.tanh(adapter.q(u))
    pairs = adapter.pair_vectors()
    assert torch.all(pairs.diagonal(dim1=0, dim2=1) == 0)
    expected = (torch.einsum("ijk,bljk->blik", pairs, v) / (D - 1)) @ adapter.w_o.T
    assert torch.allclose(adapter.cross_term(u), expected, atol=1e-12)


@pytest.mark.parametrize("coupling, crosses", [("semantic", True), ("free", True), ("none", False)])
def test_conditions_only_change_other_histories_through_the_coupling(coupling: str, crosses: bool) -> None:
    adapter = make_adapter(coupling=coupling)
    x = inputs(1).requires_grad_(True)
    out = adapter(x, return_components=True)
    grad = torch.autograd.grad(out.survival[0, -1, 0].sum(), x)[0]
    assert (grad[0, -1, 1:].abs().sum() > 0) == crosses
    memory_grad = torch.autograd.grad(adapter(x, return_components=True).memory[0, -1, 0].sum(), x)[0]
    assert (memory_grad[0, :, 1:].abs().sum() > 0) == crosses


def test_free_coupling_starts_at_the_scale_of_the_scaled_semantic_product() -> None:
    rng = np.random.default_rng(0)
    k = 128
    e_h, e_c = center_normalize(rng.normal(size=(8, k))), center_normalize(rng.normal(size=(8, k)))
    torch.manual_seed(0)
    adapter = ProseAdapter(
        history_semantics=e_h, condition_semantics=e_c, window_size=L, memory_dim=H, coupling="free"
    )
    semantic = k * e_h[:, None, :] * e_c[None, :, :]
    off_diagonal = semantic[~np.eye(8, dtype=bool)]
    assert abs(adapter.pair_coupling.detach().std().item() / off_diagonal.std() - 1.0) < 0.1
