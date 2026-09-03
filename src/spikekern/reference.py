"""The naive LIF loop, in plain PyTorch.

This is the correctness oracle. Every kernel in this package is checked against
it, so it is written for clarity rather than speed and is deliberately the most
obvious implementation of the dynamics:

    U[t] = decay(U[t-1], S[t-1]) + I[t]
    S[t] = Theta(U[t] - threshold)

with two conventions for the reset, both in use and not interchangeable:

    reset to zero:  U = beta * U[t-1] * (1 - S[t-1])
    subtract:       U = beta * U[t-1] - S[t-1] * threshold

snnTorch defaults to *subtract*. Getting this wrong does not crash; it
produces a network that trains to a different place.

with `Theta` the Heaviside step, whose derivative is zero everywhere and
undefined at the threshold. Backpropagating through that learns nothing, which
is the central difficulty in training spiking networks. The standard answer is
a surrogate gradient: keep the step in the forward pass, substitute a smooth
derivative in the backward one. The fast-sigmoid surrogate used here is

    dS/dU ~= 1 / (1 + slope * |U - threshold|)^2

It is also the *slow* implementation, and slow for a specific reason worth
naming: each of the five elementwise steps above is a separate CUDA kernel
launch that reads its inputs from HBM and writes its output back. At these
tensor sizes the arithmetic is trivial and the traffic is everything, so the
loop is memory-bandwidth-bound and spends most of its time moving data it just
moved. That is what `spikekern.fused` exists to fix.
"""

from __future__ import annotations

from typing import Any, cast

import torch

__all__ = ["SurrogateSpike", "lif_sequence", "lif_step", "spike"]


class SurrogateSpike(torch.autograd.Function):
    """Heaviside forward, fast-sigmoid derivative backward."""

    @staticmethod
    def forward(ctx: Any, membrane: torch.Tensor, slope: float) -> torch.Tensor:
        ctx.save_for_backward(membrane)
        ctx.slope = slope
        return (membrane > 0).to(membrane.dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor
                 ) -> tuple[torch.Tensor, None]:
        (membrane,) = ctx.saved_tensors
        surrogate = 1.0 / (1.0 + ctx.slope * membrane.abs()) ** 2
        return grad_output * surrogate, None


def spike(membrane: torch.Tensor, threshold: float = 1.0,
          slope: float = 25.0) -> torch.Tensor:
    """Emit spikes with a surrogate gradient."""
    out = SurrogateSpike.apply(  # type: ignore[no-untyped-call]
        membrane - threshold, slope)
    return cast(torch.Tensor, out)


def lif_step(
    current: torch.Tensor,
    membrane: torch.Tensor,
    previous_spikes: torch.Tensor,
    *,
    beta: float = 0.95,
    threshold: float = 1.0,
    slope: float = 25.0,
    reset: str = "zero",
) -> tuple[torch.Tensor, torch.Tensor]:
    """One LIF update. Returns `(spikes, membrane)`."""
    if reset == "zero":
        membrane = beta * membrane * (1.0 - previous_spikes) + current
    elif reset == "subtract":
        membrane = beta * membrane - previous_spikes * threshold + current
    else:
        raise ValueError(f"reset must be 'zero' or 'subtract', got {reset!r}")
    return spike(membrane, threshold, slope), membrane


def lif_sequence(
    current: torch.Tensor,
    *,
    beta: float = 0.95,
    threshold: float = 1.0,
    slope: float = 25.0,
    reset: str = "zero",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the LIF loop over a `(T, B, N)` input current.

    Args:
        current: Input current, shape `(timesteps, batch, neurons)`.
        beta: Membrane decay in `[0, 1)`.
        threshold: Firing threshold.
        slope: Surrogate steepness.
        reset: "zero" or "subtract". snnTorch defaults to "subtract"; this
            defaults to "zero" because that is the convention in most of the
            SNN literature. They are different dynamics, not a detail.

    Returns:
        `(spikes, membrane)`, both `(T, B, N)`. The membrane trace is returned
        as well as the spikes because the kernels have to reproduce it exactly,
        and comparing only the spikes would hide a divergence that has not yet
        crossed the threshold.
    """
    if current.dim() != 3:
        raise ValueError(
            f"expected (timesteps, batch, neurons), got {tuple(current.shape)}")
    if not 0.0 <= beta < 1.0:
        raise ValueError(f"beta must be in [0, 1), got {beta}")
    if reset not in ("zero", "subtract"):
        raise ValueError(f"reset must be 'zero' or 'subtract', got {reset!r}")

    timesteps = current.shape[0]
    membrane = torch.zeros_like(current[0])
    spikes = torch.zeros_like(current[0])

    spike_trace = []
    membrane_trace = []
    for t in range(timesteps):
        spikes, membrane = lif_step(
            current[t], membrane, spikes,
            beta=beta, threshold=threshold, slope=slope, reset=reset,
        )
        spike_trace.append(spikes)
        membrane_trace.append(membrane)

    return torch.stack(spike_trace), torch.stack(membrane_trace)
