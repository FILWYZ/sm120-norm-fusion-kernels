import torch


def rms_norm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    inv_rms = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + epsilon)
    return (x.float() * inv_rms * weight.float()).to(x.dtype)


def fused_add_rms_norm_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual_out = x + residual
    residual_float = residual_out.float()
    inv_rms = torch.rsqrt(residual_float.square().mean(dim=-1, keepdim=True) + epsilon)
    output = residual_float * inv_rms * weight.float()
    return output.to(x.dtype), residual_out
