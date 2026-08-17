import torch
import torch.nn.functional as F


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


def qk_rms_norm_rope_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    def norm_then_rope(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        normalized = F.rms_norm(
            x.float(), (x.size(-1),), None, epsilon
        ).to(original_dtype)
        normalized = normalized * weight
        cos_sin = cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        first, second = normalized.float().chunk(2, dim=-1)
        rotated_first = first * cos - second * sin
        rotated_second = second * cos + first * sin
        return torch.cat((rotated_first, rotated_second), dim=-1).to(original_dtype)

    return norm_then_rope(query, query_weight), norm_then_rope(key, key_weight)
