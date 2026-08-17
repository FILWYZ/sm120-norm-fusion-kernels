import pytest
import torch

from sm120_rmsnorm import (
    fused_add_rms_norm,
    fused_add_rms_norm_inplace,
    fused_add_rms_norm_reference,
    fused_qk_rms_norm_rope,
    fused_qk_rms_norm_rope_kv,
    qk_rms_norm_rope_reference,
    rms_norm,
    rms_norm_reference,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(1, 1024), (17, 1024), (8, 768), (3, 1000), (2, 4, 2048)])
@pytest.mark.parametrize("version", [1, 2])
def test_rms_norm(dtype: torch.dtype, shape: tuple[int, ...], version: int) -> None:
    torch.manual_seed(7)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype)
    expected = rms_norm_reference(x, weight)
    actual = rms_norm(x, weight, version=version)

    atol = 3e-2 if dtype == torch.bfloat16 else 4e-3
    rtol = 3e-2 if dtype == torch.bfloat16 else 4e-3
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens", [1, 7, 64])
@pytest.mark.parametrize("version", [0, 2, 3, 4])
def test_fused_add_rms_norm(dtype: torch.dtype, tokens: int, version: int) -> None:
    torch.manual_seed(11)
    shape = (tokens, 1024)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    residual = torch.randn_like(x)
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype)
    expected, expected_residual = fused_add_rms_norm_reference(x, residual, weight)
    actual, actual_residual = fused_add_rms_norm(
        x, residual, weight, version=version
    )

    atol = 3e-2 if dtype == torch.bfloat16 else 5e-3
    rtol = 3e-2 if dtype == torch.bfloat16 else 5e-3
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(
        actual_residual, expected_residual, atol=atol, rtol=rtol
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens", [1, 7, 64])
@pytest.mark.parametrize("version", [0, 2, 3, 4])
def test_fused_add_rms_norm_inplace(
    dtype: torch.dtype, tokens: int, version: int
) -> None:
    torch.manual_seed(13)
    shape = (tokens, 1024)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    residual = torch.randn_like(x)
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype)
    expected, expected_residual = fused_add_rms_norm_reference(x, residual, weight)

    fused_add_rms_norm_inplace(x, residual, weight, version=version)

    atol = 3e-2 if dtype == torch.bfloat16 else 5e-3
    rtol = 3e-2 if dtype == torch.bfloat16 else 5e-3
    torch.testing.assert_close(x, expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(
        residual, expected_residual, atol=atol, rtol=rtol
    )


@pytest.mark.parametrize("version", [3, 4])
def test_specialized_versions_reject_non_1024_hidden_size(version: int) -> None:
    x = torch.randn((4, 768), device="cuda", dtype=torch.float16)
    residual = torch.randn_like(x)
    weight = torch.randn(768, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="hidden_size=1024"):
        fused_add_rms_norm(x, residual, weight, version=version)


def test_auto_dispatch_falls_back_for_generic_hidden_size() -> None:
    torch.manual_seed(17)
    x = torch.randn((5, 768), device="cuda", dtype=torch.float16)
    residual = torch.randn_like(x)
    weight = torch.randn(768, device="cuda", dtype=torch.float16)
    expected, expected_residual = fused_add_rms_norm_reference(x, residual, weight)

    actual, actual_residual = fused_add_rms_norm(x, residual, weight, version=0)

    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(
        actual_residual, expected_residual, atol=5e-3, rtol=5e-3
    )


def make_rope_cache(max_position: int = 512, base: float = 1_000_000.0) -> torch.Tensor:
    inv_freq = 1.0 / (
        base ** (torch.arange(0, 128, 2, dtype=torch.float32, device="cuda") / 128)
    )
    positions = torch.arange(max_position, dtype=torch.float32, device="cuda")
    frequencies = torch.outer(positions, inv_freq)
    return torch.cat((frequencies.cos(), frequencies.sin()), dim=-1).unsqueeze(1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens", [1, 7, 64])
@pytest.mark.parametrize("heads", [(16, 8), (8, 8), (4, 2)])
@pytest.mark.parametrize("version", [1, 2])
def test_fused_qk_rms_norm_rope(
    dtype: torch.dtype, tokens: int, heads: tuple[int, int], version: int
) -> None:
    torch.manual_seed(29)
    query_heads, key_heads = heads
    query = torch.randn((tokens, query_heads, 128), device="cuda", dtype=dtype)
    key = torch.randn((tokens, key_heads, 128), device="cuda", dtype=dtype)
    query_weight = torch.randn(128, device="cuda", dtype=dtype)
    key_weight = torch.randn(128, device="cuda", dtype=dtype)
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64) * 3
    cache = make_rope_cache(max_position=max(tokens * 3, 1) + 1)
    expected_query, expected_key = qk_rms_norm_rope_reference(
        query, key, query_weight, key_weight, positions, cache
    )

    fused_qk_rms_norm_rope(
        query, key, query_weight, key_weight, positions, cache, version=version
    )

    atol = 3e-2 if dtype == torch.bfloat16 else 5e-3
    rtol = 3e-2 if dtype == torch.bfloat16 else 5e-3
    torch.testing.assert_close(query, expected_query, atol=atol, rtol=rtol)
    torch.testing.assert_close(key, expected_key, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_qk_rms_norm_rope_zero_input(dtype: torch.dtype) -> None:
    query = torch.zeros((2, 16, 128), device="cuda", dtype=dtype)
    key = torch.zeros((2, 8, 128), device="cuda", dtype=dtype)
    weight = torch.ones(128, device="cuda", dtype=dtype)
    positions = torch.tensor([0, 511], device="cuda", dtype=torch.int64)
    cache = make_rope_cache()
    fused_qk_rms_norm_rope(query, key, weight, weight, positions, cache)
    torch.testing.assert_close(query, torch.zeros_like(query))
    torch.testing.assert_close(key, torch.zeros_like(key))


def test_fused_qk_rms_norm_rope_rejects_other_head_dim() -> None:
    query = torch.randn((1, 16, 64), device="cuda", dtype=torch.float16)
    key = torch.randn((1, 8, 64), device="cuda", dtype=torch.float16)
    weight = torch.ones(64, device="cuda", dtype=torch.float16)
    positions = torch.zeros(1, device="cuda", dtype=torch.int64)
    cache = torch.zeros((1, 1, 64), device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="head_dim=128"):
        fused_qk_rms_norm_rope(query, key, weight, weight, positions, cache)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens", [1, 7, 64])
@pytest.mark.parametrize("version", [1, 4])
def test_fused_qk_rms_norm_rope_kv(
    dtype: torch.dtype, tokens: int, version: int
) -> None:
    torch.manual_seed(37)
    query = torch.randn((tokens, 16, 128), device="cuda", dtype=dtype)
    key = torch.randn((tokens, 8, 128), device="cuda", dtype=dtype)
    value = torch.randn_like(key)
    query_weight = torch.randn(128, device="cuda", dtype=dtype)
    key_weight = torch.randn(128, device="cuda", dtype=dtype)
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64) * 2
    rope_cache = make_rope_cache(max_position=max(tokens * 2, 1) + 1)
    slots = torch.arange(tokens, device="cuda", dtype=torch.int32) * 2
    if tokens > 1:
        slots[1] = -1
    key_cache = torch.randn((tokens * 2, 8, 128), device="cuda", dtype=dtype)
    value_cache = torch.randn_like(key_cache)
    expected_cache_k = key_cache.clone()
    expected_cache_v = value_cache.clone()
    expected_query, expected_key = qk_rms_norm_rope_reference(
        query, key, query_weight, key_weight, positions, rope_cache
    )
    valid = slots >= 0
    expected_cache_k[slots[valid].long()] = expected_key[valid]
    expected_cache_v[slots[valid].long()] = value[valid]

    fused_qk_rms_norm_rope_kv(
        query,
        key,
        value,
        query_weight,
        key_weight,
        positions,
        rope_cache,
        key_cache,
        value_cache,
        slots,
        version=version,
    )

    atol = 3e-2 if dtype == torch.bfloat16 else 5e-3
    rtol = 3e-2 if dtype == torch.bfloat16 else 5e-3
    torch.testing.assert_close(query, expected_query, atol=atol, rtol=rtol)
    torch.testing.assert_close(key, expected_key, atol=atol, rtol=rtol)
    torch.testing.assert_close(key_cache, expected_cache_k, atol=atol, rtol=rtol)
    torch.testing.assert_close(value_cache, expected_cache_v, atol=atol, rtol=rtol)
