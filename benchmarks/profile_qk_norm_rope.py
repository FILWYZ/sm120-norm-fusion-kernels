import argparse

import torch

from sm120_rmsnorm import fused_qk_rms_norm_rope, fused_qk_rms_norm_rope_kv


def make_cache(max_position: int) -> torch.Tensor:
    inv_freq = 1.0 / (
        1_000_000.0
        ** (torch.arange(0, 128, 2, device="cuda", dtype=torch.float32) / 128)
    )
    frequencies = torch.outer(
        torch.arange(max_position, device="cuda", dtype=torch.float32), inv_freq
    )
    return torch.cat((frequencies.cos(), frequencies.sin()), dim=-1).unsqueeze(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--version", type=int, choices=[1, 2, 4], default=2)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--store-kv", action="store_true")
    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    query = torch.randn((args.tokens, 16, 128), device="cuda", dtype=dtype)
    key = torch.randn((args.tokens, 8, 128), device="cuda", dtype=dtype)
    query_weight = torch.ones(128, device="cuda", dtype=dtype)
    key_weight = torch.ones(128, device="cuda", dtype=dtype)
    positions = torch.arange(args.tokens, device="cuda", dtype=torch.int64)
    cache = make_cache(max(args.tokens, 1))

    value = torch.randn_like(key)
    key_cache = torch.empty((args.tokens, 8, 128), device="cuda", dtype=dtype)
    value_cache = torch.empty_like(key_cache)
    slots = torch.arange(args.tokens, device="cuda", dtype=torch.int32)

    def run() -> None:
        if args.store_kv:
            fused_qk_rms_norm_rope_kv(
                query,
                key,
                value,
                query_weight,
                key_weight,
                positions,
                cache,
                key_cache,
                value_cache,
                slots,
                version=args.version,
            )
        else:
            fused_qk_rms_norm_rope(
                query,
                key,
                query_weight,
                key_weight,
                positions,
                cache,
                version=args.version,
            )

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    run()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
