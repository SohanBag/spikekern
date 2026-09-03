"""Reproduce the tables in the README.

    python examples/benchmark.py

Needs a CUDA device and Triton. Without them `fused_lif` silently falls back to
the reference loop, so the speedup would come out at 1.0x and the numbers would
be meaningless rather than merely absent -- the script checks and says so.
"""

from __future__ import annotations

import statistics
import sys
import time

import torch

from spikekern import TRITON_AVAILABLE, fused_lif, lif_sequence

try:
    import snntorch as snn

    SNNTORCH = True
except ImportError:  # pragma: no cover
    SNNTORCH = False


def snntorch_loop(current: torch.Tensor, beta: float = 0.9,
                  reset: str = "zero") -> tuple[torch.Tensor, torch.Tensor]:
    """The loop a snnTorch user would actually write."""
    lif = snn.Leaky(beta=beta, threshold=1.0,
                    reset_mechanism=reset).to(current.device)
    membrane = torch.zeros_like(current[0])
    spikes, membranes = [], []
    for t in range(current.shape[0]):
        spike, membrane = lif(current[t], membrane)
        spikes.append(spike)
        membranes.append(membrane)
    return torch.stack(spikes), torch.stack(membranes)


def timed(fn, x, runs: int = 15, warmup: int = 5) -> float:  # type: ignore[no-untyped-def]
    """Median wall time of forward-plus-backward, in milliseconds.

    Synchronised on both sides. Without it the timer measures how fast Python
    queued the work, which on short kernels is several times too fast.
    """
    for _ in range(warmup):
        a = x.clone().requires_grad_(True)
        spikes, membrane = fn(a)
        (spikes.sum() + 0.5 * membrane.sum()).backward()
    torch.cuda.synchronize()

    samples = []
    for _ in range(runs):
        a = x.clone().requires_grad_(True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        spikes, membrane = fn(a)
        (spikes.sum() + 0.5 * membrane.sum()).backward()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(samples)


def peak_mb(fn, x) -> float:  # type: ignore[no-untyped-def]
    """Peak allocation for one forward-plus-backward.

    The collect-and-empty before resetting the counter is load-bearing. Without
    it, tensors still alive from an earlier measurement stay in the peak and
    both implementations report roughly the same large number -- which made the
    saving look like 1.6% when measured in isolation it is over 20%.
    """
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    a = x.clone().requires_grad_(True)
    spikes, membrane = fn(a)
    (spikes.sum() + 0.5 * membrane.sum()).backward()
    torch.cuda.synchronize()
    peak = float(torch.cuda.max_memory_allocated()) / 1e6

    del a, spikes, membrane
    gc.collect()
    torch.cuda.empty_cache()
    return peak


def correctness() -> None:
    print("\n## Correctness, against the reference and against snnTorch\n")
    torch.manual_seed(0)
    current = torch.randn(32, 8, 128, device="cuda") * 0.6

    for reset in ("zero", "subtract"):
        fused_s, fused_u = fused_lif(current, beta=0.9, reset=reset)
        ref_s, ref_u = lif_sequence(current, beta=0.9, reset=reset)
        line = (f"  reset={reset:<9} vs reference: spikes "
                f"{(ref_s - fused_s).abs().max().item():.2e}"
                f"  membrane {(ref_u - fused_u).abs().max().item():.2e}")
        if SNNTORCH:
            snn_s, snn_u = snntorch_loop(current, 0.9, reset)
            line += (f"   vs snnTorch: spikes "
                     f"{(snn_s - fused_s).abs().max().item():.2e}"
                     f"  membrane {(snn_u - fused_u).abs().max().item():.2e}")
        print(line)

    print()
    for reset in ("zero", "subtract"):
        a = current.clone().requires_grad_(True)
        s, u = lif_sequence(a, beta=0.9, reset=reset)
        (s.sum() + 0.5 * u.sum()).backward()

        b = current.clone().requires_grad_(True)
        s2, u2 = fused_lif(b, beta=0.9, reset=reset)
        (s2.sum() + 0.5 * u2.sum()).backward()

        delta = (a.grad - b.grad).abs().max().item()
        scale = max(a.grad.abs().max().item(), 1e-12)
        print(f"  reset={reset:<9} backward max|diff| {delta:.3e}"
              f"   relative {delta / scale:.3e}")


def throughput() -> None:
    print("\n## Speed, forward plus backward, median of 15 runs\n")
    header = f"{'T':>5} {'B':>5} {'N':>6} {'reference':>12} {'fused':>10} {'speedup':>9}"
    if SNNTORCH:
        header += f" {'snnTorch':>11} {'vs snnTorch':>12}"
    print(header)

    torch.manual_seed(0)
    for timesteps, batch, neurons in ((16, 32, 256), (32, 64, 512),
                                      (64, 64, 512), (128, 64, 512),
                                      (256, 32, 512)):
        x = torch.randn(timesteps, batch, neurons, device="cuda") * 0.6
        reference = timed(lambda a: lif_sequence(a, beta=0.9), x)
        fused = timed(lambda a: fused_lif(a, beta=0.9), x)
        row = (f"{timesteps:5} {batch:5} {neurons:6} "
               f"{reference:10.3f}ms {fused:8.3f}ms {reference / fused:8.1f}x")
        if SNNTORCH:
            native = timed(lambda a: snntorch_loop(a, 0.9), x)
            row += f" {native:9.3f}ms {native / fused:11.1f}x"
        print(row)


def memory() -> None:
    print("\n## Peak memory, forward plus backward\n")
    print(f"{'T':>5} {'B':>5} {'N':>6} {'reference':>11} {'fused':>10} {'saved':>8}")
    torch.manual_seed(0)
    for timesteps, batch, neurons in ((32, 64, 512), (128, 64, 512),
                                      (256, 32, 512)):
        x = torch.randn(timesteps, batch, neurons, device="cuda") * 0.6
        reference = peak_mb(lambda a: lif_sequence(a, beta=0.9), x)
        fused = peak_mb(lambda a: fused_lif(a, beta=0.9), x)
        print(f"{timesteps:5} {batch:5} {neurons:6} {reference:9.1f}MB "
              f"{fused:8.1f}MB {1 - fused / reference:7.1%}")


def launches() -> None:
    print("\n## Why: kernel launches\n")
    print(f"{'T':>6} {'reference':>12} {'fused':>8}  ratio")
    for timesteps in (16, 32, 64, 128, 256):
        # decay, reset, integrate, threshold, cast -- five elementwise ops per
        # step in the forward pass alone.
        print(f"{timesteps:6} {timesteps * 5:12,} {2:8}  {timesteps * 5 / 2:.0f}x")
    print("\n  The fused path launches one forward and one backward kernel")
    print("  regardless of T. That ratio is the speedup's whole explanation.")


def main() -> int:
    if not torch.cuda.is_available():
        print("No CUDA device. These numbers require one.")
        return 1
    if not TRITON_AVAILABLE:
        print("Triton is not installed, so fused_lif would fall back to the")
        print("reference loop and every speedup would read 1.0x. Install with:")
        print('  pip install "spikekern[triton]"')
        return 1

    properties = torch.cuda.get_device_properties(0)
    print(f"device: {properties.name}, {properties.total_memory / 1e9:.2f} GB")
    print(f"torch:  {torch.__version__}")
    print(f"snnTorch: {'available' if SNNTORCH else 'not installed'}")

    correctness()
    # Memory first: the timing runs leave large allocations cached, and a peak
    # measured after them reflects those rather than the kernel.
    memory()
    throughput()
    launches()
    return 0


if __name__ == "__main__":
    sys.exit(main())
