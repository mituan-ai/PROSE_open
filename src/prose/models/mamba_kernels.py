"""从 HuggingFace kernel hub 加载 Mamba CUDA 内核，并固定到指定提交。

transformers 每次启动时都会通过 hub API 按版本查找内核；离线运行时会悄然退回
Python 串行扫描。此处将各内核固定到提交，优先从本地 hub 缓存加载，无需联网；
缓存不存在时，再按指定提交下载一次。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

from torch import Tensor

PINNED_KERNELS = {  # 内核名称 -> (hub 仓库, 提交)；采用 transformers 5.8 使用的版本 1
    "causal-conv1d": ("kernels-community/causal-conv1d", "19d5632c27565e020a0aa0169bc7ad44a2c7080b"),
    "mamba-ssm": ("kernels-community/mamba-ssm", "a39ff24c08103278583f168091409653ada4c292"),
}


def load_pinned_kernel(name: str) -> ModuleType | None:
    """加载固定版本的内核，并注册到 transformers 查找内核的位置。"""

    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        from kernels import get_kernel, get_local_kernel
        from transformers.integrations import hub_kernels
    except ImportError:
        return None
    cached = hub_kernels._KERNEL_MODULE_MAPPING.get(name)
    if isinstance(cached, ModuleType):
        return cached
    repo, commit = PINNED_KERNELS[name]
    snapshot = Path(HF_HUB_CACHE) / f"kernels--{repo.replace('/', '--')}" / "snapshots" / commit
    module = None
    if snapshot.exists():
        try:
            module = get_local_kernel(snapshot)
        except Exception:  # 快照不完整，随后重新下载
            module = None
    if module is None:
        try:
            module = get_kernel(repo, revision=commit)
        except Exception:
            return None
    hub_kernels._KERNEL_MODULE_MAPPING[name] = module
    return module


def enable_hf_mamba_kernels() -> bool:
    """以 HuggingFace Mamba 所需的名称暴露固定版本内核。"""

    causal_conv1d_kernel = load_pinned_kernel("causal-conv1d")
    if causal_conv1d_kernel is None or getattr(causal_conv1d_kernel, "__file__", None) is None:
        return False

    kernel_root = str(Path(causal_conv1d_kernel.__file__).parent)
    if kernel_root not in sys.path:
        sys.path.insert(0, kernel_root)

    try:
        import causal_conv1d.causal_conv1d_interface as causal_interface
        from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    except ImportError:
        return False

    if not hasattr(causal_interface, "causal_conv1d_cuda"):
        causal_interface.causal_conv1d_cuda = _CausalConv1dCudaAdapter

    mamba_kernel = load_pinned_kernel("mamba-ssm")
    if mamba_kernel is None:
        return False

    try:
        import importlib

        import transformers.models.mamba.modeling_mamba as hf_mamba

        scan_interface = importlib.import_module(f"{mamba_kernel.__name__}.ops.selective_scan_interface")
    except ImportError, AttributeError:
        return False

    if getattr(scan_interface, "causal_conv1d_fn", None) is None:
        scan_interface.causal_conv1d_fn = causal_conv1d_fn
    if getattr(scan_interface, "causal_conv1d_cuda", None) is None:
        scan_interface.causal_conv1d_cuda = causal_interface.causal_conv1d_cuda

    hf_mamba.causal_conv1d_fn = causal_conv1d_fn
    hf_mamba.causal_conv1d_update = causal_conv1d_update
    mamba_inner_fn = getattr(mamba_kernel, "mamba_inner_fn", None)
    if mamba_inner_fn is not None:
        hf_mamba.mamba_inner_fn = mamba_inner_fn
    selective_scan_fn = getattr(mamba_kernel, "selective_scan_fn", None)
    if selective_scan_fn is not None:
        hf_mamba.selective_scan_fn = selective_scan_fn
    return True


class _CausalConv1dCudaAdapter:
    @staticmethod
    def causal_conv1d_fwd(
        x: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        seq_idx: Tensor | None,
        initial_states: Tensor | None,
        final_states_out: Tensor | None,
        silu_activation: bool,
    ) -> Tensor:
        from causal_conv1d.cpp_functions import causal_conv1d_fwd_function

        return causal_conv1d_fwd_function(
            x,
            weight,
            bias,
            seq_idx,
            initial_states,
            final_states_out,
            silu_activation,
        )

    @staticmethod
    def causal_conv1d_bwd(
        x: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        dout: Tensor,
        seq_idx: Tensor | None,
        initial_states: Tensor | None,
        dfinal_states: Tensor | None,
        dx: Tensor | None,
        return_dinitial_states: bool,
        silu_activation: bool,
    ) -> tuple[Tensor | None, ...]:
        from causal_conv1d.cpp_functions import causal_conv1d_bwd_function

        return causal_conv1d_bwd_function(
            x,
            weight,
            bias,
            dout,
            seq_idx,
            initial_states,
            dfinal_states,
            dx,
            return_dinitial_states,
            silu_activation,
        )

    @staticmethod
    def causal_conv1d_update(
        x: Tensor,
        conv_state: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        silu_activation: bool,
        cache_seqlens: Tensor | None,
        conv_state_indices: Tensor | None,
    ) -> Tensor:
        from causal_conv1d.cpp_functions import causal_conv1d_update_function

        return causal_conv1d_update_function(
            x,
            conv_state,
            weight,
            bias,
            silu_activation,
            cache_seqlens,
            conv_state_indices,
        )
