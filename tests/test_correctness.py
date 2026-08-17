import pytest
import torch

from sm120_rmsnorm import (
    fused_add_rms_norm,
    fused_add_rms_norm_inplace,
    fused_add_rms_norm_reference,
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
