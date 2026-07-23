from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def elapsed_ms(fn, repetitions: int) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repetitions


def main() -> None:
    parser = argparse.ArgumentParser(description="E2: pinned-memory H2D and SwiGLU block microbenchmark")
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=8192)
    parser.add_argument("--block-widths", type=int, nargs="+", default=(64, 128, 256, 512, 1024, 2048))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", default="results/e2_microbenchmark.csv")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("E2 requires a CUDA GPU.")
    if args.intermediate_size % min(args.block_widths) != 0:
        print("Warning: some block widths do not evenly divide intermediate size.")
    device, dtype = torch.device("cuda"), torch.float16
    torch.manual_seed(0)
    x = torch.randn(args.batch_size, args.hidden_size, device=device, dtype=dtype)
    rows = []
    widths = list(args.block_widths)
    if args.intermediate_size not in widths:
        widths.append(args.intermediate_size)
    for width in widths:
        if width > args.intermediate_size:
            continue
        # A single contiguous blob models a packed gate/up/down block transfer.
        elements = 3 * args.hidden_size * width
        host = torch.randn(elements, dtype=dtype, pin_memory=True)
        destination = torch.empty_like(host, device=device)
        gate = torch.randn(width, args.hidden_size, device=device, dtype=dtype)
        up = torch.randn_like(gate)
        down = torch.randn(args.hidden_size, width, device=device, dtype=dtype)
        for _ in range(args.warmup):
            destination.copy_(host, non_blocking=True)
            _ = (F.silu(x @ gate.T) * (x @ up.T)) @ down.T
        torch.cuda.synchronize()
        copy_ms = elapsed_ms(lambda: destination.copy_(host, non_blocking=True), args.repetitions)
        kernel_ms = elapsed_ms(lambda: (F.silu(x @ gate.T) * (x @ up.T)) @ down.T, args.repetitions)
        bytes_per_block = elements * torch.tensor([], dtype=dtype).element_size()
        blocks_to_full = math.ceil(args.intermediate_size / width)
        serial_ms = copy_ms + kernel_ms
        rows.append({
            "block_width": width,
            "is_whole_expert": width == args.intermediate_size,
            "blocks_to_cover_expert": blocks_to_full,
            "bytes": bytes_per_block,
            "copy_ms": copy_ms,
            "kernel_ms": kernel_ms,
            "serial_ms": serial_ms,
            "aggregate_full_expert_serial_ms": serial_ms * blocks_to_full,
            "effective_h2d_gbps": bytes_per_block / (copy_ms * 1e6),
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    whole = next(row for row in rows if row["is_whole_expert"])
    print(f"Wrote {output}")
    for row in rows:
        ratio = row["serial_ms"] / whole["serial_ms"]
        aggregate_ratio = row["aggregate_full_expert_serial_ms"] / whole["serial_ms"]
        print(f"width={row['block_width']:5d} copy={row['copy_ms']:.3f}ms kernel={row['kernel_ms']:.3f}ms "+
              f"one-block/full={ratio:.3f} aggregate/full={aggregate_ratio:.3f} "+
              f"bandwidth={row['effective_h2d_gbps']:.2f} GB/s")


if __name__ == "__main__":
    main()
