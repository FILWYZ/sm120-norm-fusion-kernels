import torch

try:
    from . import _C
except ImportError as error:  # pragma: no cover - exercised before extension build
    _C = None
    _IMPORT_ERROR = error
else:
    _IMPORT_ERROR = None


def _require_extension() -> None:
    if _C is None:
        raise RuntimeError(
            "CUDA extension is not built. Run `python -m pip install -e . --no-build-isolation`."
        ) from _IMPORT_ERROR


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
    version: int = 2,
) -> torch.Tensor:
    _require_extension()
    return _C.rms_norm(x, weight, epsilon, version)


def rms_norm_out(
    x: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    epsilon: float = 1e-6,
    version: int = 2,
) -> None:
    _require_extension()
    _C.rms_norm_out(x, weight, output, epsilon, version)


def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
    version: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_extension()
    output, residual_output = _C.fused_add_rms_norm(
        x, residual, weight, epsilon, version
    )
    return output, residual_output


def fused_add_rms_norm_out(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    residual_output: torch.Tensor,
    epsilon: float = 1e-6,
    version: int = 0,
) -> None:
    _require_extension()
    _C.fused_add_rms_norm_out(
        x, residual, weight, output, residual_output, epsilon, version
    )


def fused_add_rms_norm_inplace(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
    version: int = 0,
) -> None:
    """Update ``residual += x`` and replace ``x`` with RMSNorm(residual)."""
    _require_extension()
    _C.fused_add_rms_norm_inplace(x, residual, weight, epsilon, version)


def fused_qk_rms_norm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    epsilon: float = 1e-6,
    version: int = 2,
) -> None:
    """Apply per-head RMSNorm and half-split RoPE to Q/K in place."""
    _require_extension()
    torch.ops.sm120_norm.fused_qk_rms_norm_rope_(
        query,
        key,
        query_weight,
        key_weight,
        positions,
        cos_sin_cache,
        epsilon,
        version,
    )


def fused_qk_rms_norm_rope_kv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    epsilon: float = 1e-6,
    version: int = 1,
) -> None:
    """Fuse Q/K Norm+RoPE and scatter rotated K plus V into paged cache."""
    _require_extension()
    torch.ops.sm120_norm.fused_qk_rms_norm_rope_kv_(
        query,
        key,
        value,
        query_weight,
        key_weight,
        positions,
        cos_sin_cache,
        key_cache,
        value_cache,
        slot_mapping,
        epsilon,
        version,
    )


if _C is not None:

    @torch.library.register_fake("sm120_norm::fused_qk_rms_norm_rope_")
    def _fused_qk_rms_norm_rope_fake(
        query: torch.Tensor,
        key: torch.Tensor,
        query_weight: torch.Tensor,
        key_weight: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        epsilon: float = 1e-6,
        version: int = 2,
    ) -> None:
        return None

    @torch.library.register_fake("sm120_norm::fused_qk_rms_norm_rope_kv_")
    def _fused_qk_rms_norm_rope_kv_fake(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_weight: torch.Tensor,
        key_weight: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        epsilon: float = 1e-6,
        version: int = 1,
    ) -> None:
        return None
