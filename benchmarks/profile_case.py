import argparse

import torch

from sm120_rmsnorm import fused_add_rms_norm_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--version", type=int, choices=[2, 3, 4], default=4)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()

    x = torch.randn((args.tokens, args.hidden), device="cuda", dtype=torch.float16)
    residual = torch.randn_like(x)
    weight = torch.randn((args.hidden,), device="cuda", dtype=torch.float16)
    output = torch.empty_like(x)
    residual_output = torch.empty_like(x)

    for _ in range(args.warmup):
        fused_add_rms_norm_out(
            x, residual, weight, output, residual_output, version=args.version
        )
    torch.cuda.synchronize()
    fused_add_rms_norm_out(
        x, residual, weight, output, residual_output, version=args.version
    )
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
