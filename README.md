# spikekern

**Fused Triton kernels for LIF spiking neuron dynamics. Bit-exact with snnTorch, up to 130× faster.**

[![CI](https://github.com/SohanBag/spikekern/actions/workflows/ci.yml/badge.svg)](https://github.com/SohanBag/spikekern/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A leaky integrate-and-fire layer is five elementwise operations per timestep:
decay, reset, integrate, threshold, emit. Written the obvious way — a Python
loop over `T` calling into PyTorch — each of those is a separate CUDA kernel
launch that reads its operands from HBM and writes the result back.

The arithmetic is trivial and the traffic is everything, so the loop is
bandwidth-bound end to end and spends most of its time moving data it moved a
microsecond ago. At `T = 256` that is **1,280 kernel launches** to do work that
fits in two.

```bash
pip install -e ".[dev,triton]"
```

```python
from spikekern import fused_lif

# current: (timesteps, batch, neurons)
spikes, membrane = fused_lif(current, beta=0.9, reset="subtract")
```

Falls back to a plain PyTorch loop on CPU or without Triton, so calling code
never has to branch.

## Measured

RTX 5060 Laptop, 8.52 GB, PyTorch 2.11 + CUDA 12.8, Triton 3.8. Forward plus
backward, median of 15 runs. Reproduce with `python examples/benchmark.py`.

| T | B | N | reference loop | **fused** | speedup | snnTorch | **vs snnTorch** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 32 | 256 | 6.289 ms | **0.350 ms** | 18.0× | 7.711 ms | **22.0×** |
| 32 | 64 | 512 | 10.857 ms | **0.300 ms** | 36.2× | 15.471 ms | **51.6×** |
| 64 | 64 | 512 | 23.182 ms | **0.382 ms** | 60.7× | 28.489 ms | **74.6×** |
| 128 | 64 | 512 | 42.765 ms | **0.663 ms** | 64.5× | 58.087 ms | **87.6×** |
| 256 | 32 | 512 | 84.858 ms | **0.888 ms** | 95.6× | 114.243 ms | **128.7×** |

Peak memory over the same pass falls by a consistent **22%**: 150.9 MB to
117.5 MB at `T = 256`.

### The speedup is a launch-count result, not a clever-arithmetic one

| T | reference launches | fused | ratio |
| ---: | ---: | ---: | ---: |
| 16 | 80 | 2 | 40× |
| 64 | 320 | 2 | 160× |
| 256 | 1,280 | 2 | 640× |

Five elementwise ops per timestep in the forward pass, and the fused path
launches one forward kernel and one backward kernel **regardless of `T`**. Each
Triton program keeps the membrane potential live in a register while it walks
the entire time axis, so per-timestep traffic drops from roughly five round
trips through HBM to one read and two writes.

That is also why the speedup **grows with `T`**: the fused cost is nearly flat
in the number of timesteps while the loop's cost is linear in it. The arithmetic
is identical; only the data movement changed.

### The layout is the other half

Input is `(T, B, N)` and each program owns a contiguous block of the flattened
`B*N` axis, stepping through time in a loop. Neighbouring threads then read
neighbouring addresses within a timestep, so the loads coalesce.

The reverse layout, `(B, N, T)` with time contiguous, looks appealing — each
program's time-walk would be sequential in memory. It is worse: threads within a
warp would be `T` elements apart, turning every load into a gather.

## Correctness

A fast kernel that is subtly wrong is worse than no kernel, because the network
still trains, still converges to something, and nothing reports the difference.
So this is checked against two independent implementations.

| | vs reference | vs snnTorch |
| --- | ---: | ---: |
| `reset="zero"` spikes | **0.00e+00** | **0.00e+00** |
| `reset="zero"` membrane | **0.00e+00** | **0.00e+00** |
| `reset="subtract"` spikes | **0.00e+00** | **0.00e+00** |
| `reset="subtract"` membrane | 4.77e-07 | 4.77e-07 |

Spike trains are **bit-identical** in both conventions. The membrane differs
only by float32 rounding.

Gradients, against the reference: relative error **2.9e-07** for reset-to-zero
and **3.8e-06** for subtract, which is float32 accumulation noise over a
256-step recurrence.

### Both reset conventions, because they are different networks

```
reset to zero:  U[t] = beta * U[t-1] * (1 - S[t-1]) + I[t]
subtract:       U[t] = beta * U[t-1] - S[t-1] * threshold + I[t]
```

**snnTorch defaults to `subtract`.** A kernel supporting only reset-to-zero
would be bit-exact against the convention most users are not using, and swapping
it in would silently change what their network learns. Both are compiled as
separate kernels via a `constexpr`, so neither pays for a runtime branch.

### The bug the gradient tests caught

The first backward kernel was wrong by a relative **0.94** while the forward was
bit-exact throughout — exactly the failure mode that a forward-only check misses.

`U[t]` reaches the loss three ways: directly, through `S[t]` via the surrogate
derivative, and through `U[t+1]` via the decay factor. And `S[t]` has two of its
own: the spike output, and `U[t+1]` through the **reset** term. That last path
was missing. Its derivative is `-beta*U[t]` for reset-to-zero and `-threshold`
for subtract, and omitting it leaves a backward pass that looks plausible,
returns finite gradients, and trains the wrong network.

## Limitations

- **One GPU, one architecture.** RTX 5060 Laptop, compute capability 12.0. The
  launch-count argument is architecture-independent; the multipliers are not.
- **LIF only.** No adaptive threshold, no synaptic current, no recurrent
  connections. `snn.Synaptic` and `snn.Alpha` have no equivalent here.
- **float32 only.** No bf16 or fp16 path, which is where a real training run
  would want to be.
- **No autotuning.** `BLOCK` is fixed at 1024. A Triton autotune sweep over
  block size and warps would likely find more, and has not been run.
- **Not profiled per-kernel.** The launch-count explanation is arithmetic and
  is consistent with how the speedup scales in `T`, but no Nsight Compute trace
  is committed to confirm the memory-traffic claim directly.
- **The benchmark is the layer, not a network.** Real training also spends time
  in the linear layers between spiking layers, so end-to-end speedups will be
  smaller than these.

## Development

```bash
pytest        # 31 tests
ruff check .
mypy          # strict
```

Kernel tests are marked `cuda` and skip without a GPU and Triton; the
reference, the fallback and the validation still run, which is what CI checks.

**Coverage reads 68%, and the gap is measurement rather than untested code.**
`coverage.py` cannot see inside `@triton.jit` kernel bodies, which are compiled
rather than executed as Python, and cannot trace `_FusedLIF.backward`, which
runs on a C++ autograd engine thread. That is about a third of `fused.py`, all
of it exercised by the gradient tests. Marking those lines `no cover` would
raise the number while measuring less, so it stays as it is.

### Installing Triton

The official wheel is Linux-only. On Windows the community
[`triton-windows`](https://pypi.org/project/triton-windows/) port works — it is
what these numbers were measured with, on a Blackwell card. The `triton` extra
picks the right one per platform.

## Related reading

- Tillet et al., *Triton: An Intermediate Language and Compiler for Tiled Neural
  Network Computations*, MAPL 2019.
- Neftci et al., *Surrogate Gradient Learning in Spiking Neural Networks*,
  IEEE SPM 2019.
- Williams et al., *Roofline: An Insightful Visual Performance Model*, CACM 2009.
  The reason to expect a launch-bound elementwise loop to be the bottleneck.

## Related

[spikefit](https://github.com/SohanBag/spikefit) attacks the same problem from
the memory side, with gradient checkpointing over the temporal dimension. The
two compose: checkpointing decides how much of the activation history to keep,
these kernels decide how fast each chunk runs.

## License

MIT. See [LICENSE](LICENSE).
