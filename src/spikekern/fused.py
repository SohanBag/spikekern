"""Fused LIF kernels in Triton.

## What is being fused, and why it is worth fusing

The naive loop in `reference.py` issues five elementwise operations per
timestep: decay, reset, integrate, threshold, and the spike cast. Each is its
own kernel launch, and each reads its operands from HBM and writes its result
back. For a `(B, N)` state tensor the arithmetic per element is a handful of
FLOPs while the traffic is tens of bytes, so the whole loop sits far to the
left of the roofline knee and is bandwidth-bound from end to end.

Fusing them means the membrane state stays in registers across the whole
timestep, and across *every* timestep: each program keeps `U` live in a
register while it walks the time axis, reading only the input current and
writing only the outputs. Traffic per timestep drops from roughly five
round trips to one read and two writes.

## The layout choice

Input is `(T, B, N)` and each Triton program owns a contiguous block of the
flattened `B*N` axis, stepping through `T` in a loop. That ordering is what
makes the loads coalesce: neighbouring threads read neighbouring addresses
within a timestep, and the stride between timesteps is a single large jump
rather than a scatter.

The reverse layout, `(B, N, T)` with time contiguous, looks appealing because
each program's time-walk would be sequential in memory. It is worse: threads
within a warp would then be `T` elements apart, so every load becomes a
gather across the whole tensor.

## The backward pass

Backpropagation through time runs the recurrence in reverse, and the gradient
of the membrane at step `t` depends on the gradient at `t+1` through both the
decay term and the reset term. The backward kernel walks time downward with
that carry in a register, which is the same fusion argument applied to the
reverse pass, and it recomputes nothing: the forward membrane trace is saved.
"""

from __future__ import annotations

from typing import Any, cast

import torch

from .reference import lif_sequence

__all__ = ["TRITON_AVAILABLE", "fused_lif", "triton_unavailable_reason"]

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
    _REASON = ""
except ImportError as exc:  # pragma: no cover - depends on the environment
    TRITON_AVAILABLE = False
    _REASON = str(exc)
    triton = None
    tl = None


def triton_unavailable_reason() -> str:
    """Why Triton could not be imported, or an empty string when it could."""
    return _REASON


if TRITON_AVAILABLE:
    # Triton kernels are not typed and cannot usefully be: @triton.jit does not
    # return a Python callable, and the parameters are device pointers and
    # constexprs rather than values mypy can reason about. The ignores are
    # scoped to the two kernels rather than relaxing strict mode for the file.

    @triton.jit  # type: ignore[untyped-decorator]
    def _lif_forward_kernel(  # type: ignore[no-untyped-def]
        current_ptr, spike_ptr, membrane_ptr,
        n_elements, timesteps,
        beta, threshold,
        BLOCK: tl.constexpr,
        RESET_ZERO: tl.constexpr,
    ):
        """One program per block of neurons; walks the whole time axis.

        `RESET_ZERO` is a constexpr, so Triton compiles a separate kernel for
        each reset convention and neither pays for a runtime branch.
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements

        # The two pieces of state that stay in registers for the whole loop.
        # This is the fusion: neither ever touches memory.
        membrane = tl.zeros([BLOCK], dtype=tl.float32)
        spikes = tl.zeros([BLOCK], dtype=tl.float32)

        for t in range(timesteps):
            base = t * n_elements
            current = tl.load(current_ptr + base + offsets, mask=mask, other=0.0)

            # decay, reset, integrate -- one expression, no round trip
            if RESET_ZERO:
                membrane = beta * membrane * (1.0 - spikes) + current
            else:
                membrane = beta * membrane - spikes * threshold + current
            spikes = tl.where(membrane - threshold > 0.0, 1.0, 0.0)

            tl.store(spike_ptr + base + offsets, spikes, mask=mask)
            tl.store(membrane_ptr + base + offsets, membrane, mask=mask)

    @triton.jit  # type: ignore[untyped-decorator]
    def _lif_backward_kernel(  # type: ignore[no-untyped-def]
        grad_spike_ptr, grad_membrane_out_ptr,
        membrane_ptr, spike_ptr,
        grad_current_ptr,
        n_elements, timesteps,
        beta, threshold, slope,
        BLOCK: tl.constexpr,
        RESET_ZERO: tl.constexpr,
    ):
        """Reverse-time sweep, carrying dL/dU in a register.

        The recurrence, with `U[t] = beta * U[t-1] * (1 - S[t-1]) + I[t]` and
        `S[t] = Theta(U[t] - threshold)`, gives `U[t]` three paths to the loss:

          1. directly, if the membrane trace is an output          -> gUout[t]
          2. through `S[t]`, via the surrogate derivative
          3. through `U[t+1]`, via the decay factor `beta*(1-S[t])`

        and `S[t]` has two of its own: the spike output, and `U[t+1]` through
        the **reset** term, whose derivative is `-beta * U[t]`.

        That reset path is easy to miss and its absence is not subtle: leaving
        it out gave relative gradient errors up to 0.94 against the reference.
        Both terms are here, and `spikes` is indexed at `t` rather than `t-1`
        because the decay from `t+1` back to `t` is gated by the spike emitted
        *at* `t`.
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements

        # dL/dU[t+1], carried backwards in a register.
        carry = tl.zeros([BLOCK], dtype=tl.float32)

        for i in range(timesteps):
            t = timesteps - 1 - i
            base = t * n_elements

            membrane = tl.load(membrane_ptr + base + offsets, mask=mask, other=0.0)
            spikes = tl.load(spike_ptr + base + offsets, mask=mask, other=0.0)
            grad_spike = tl.load(grad_spike_ptr + base + offsets, mask=mask,
                                 other=0.0)
            grad_membrane_out = tl.load(grad_membrane_out_ptr + base + offsets,
                                        mask=mask, other=0.0)

            shifted = membrane - threshold
            denominator = 1.0 + slope * tl.abs(shifted)
            surrogate = 1.0 / (denominator * denominator)

            # Everything reaching S[t]: the spike output, plus the reset
            # path into U[t+1]. Both derivatives differ by convention:
            #   zero:      dU[t+1]/dS[t] = -beta * U[t]
            #              dU[t+1]/dU[t] =  beta * (1 - S[t])
            #   subtract:  dU[t+1]/dS[t] = -threshold
            #              dU[t+1]/dU[t] =  beta
            if RESET_ZERO:
                grad_spike_total = grad_spike - carry * beta * membrane
                decay_path = carry * beta * (1.0 - spikes)
            else:
                grad_spike_total = grad_spike - carry * threshold
                decay_path = carry * beta

            # Everything reaching U[t]: the membrane output, the surrogate
            # path through S[t], and the decay path into U[t+1].
            grad_membrane = (grad_membrane_out
                             + grad_spike_total * surrogate
                             + decay_path)

            # dU[t]/dI[t] is 1, so dL/dI[t] is exactly dL/dU[t].
            tl.store(grad_current_ptr + base + offsets, grad_membrane, mask=mask)
            carry = grad_membrane


class _FusedLIF(torch.autograd.Function):
    """Autograd wrapper around the two kernels."""

    @staticmethod
    def forward(ctx: Any, current: torch.Tensor, beta: float,
                threshold: float, slope: float, reset: str
                ) -> tuple[torch.Tensor, torch.Tensor]:
        timesteps, batch, neurons = current.shape
        n_elements = batch * neurons
        current = current.contiguous()

        spikes = torch.empty_like(current)
        membrane = torch.empty_like(current)

        block = 1024
        grid = (triton.cdiv(n_elements, block),)
        _lif_forward_kernel[grid](
            current, spikes, membrane,
            n_elements, timesteps,
            beta, threshold,
            BLOCK=block,
            RESET_ZERO=(reset == "zero"),
        )

        ctx.save_for_backward(membrane, spikes)
        ctx.beta, ctx.threshold, ctx.slope = beta, threshold, slope
        ctx.reset = reset
        ctx.shape = (timesteps, batch, neurons)
        return spikes, membrane

    @staticmethod
    def backward(ctx: Any, grad_spikes: torch.Tensor,
                 grad_membrane: torch.Tensor
                 ) -> tuple[torch.Tensor, None, None, None, None]:
        membrane, spikes = ctx.saved_tensors
        timesteps, batch, neurons = ctx.shape
        n_elements = batch * neurons

        grad_current = torch.empty_like(membrane)
        block = 1024
        grid = (triton.cdiv(n_elements, block),)
        _lif_backward_kernel[grid](
            grad_spikes.contiguous(), grad_membrane.contiguous(),
            membrane, spikes,
            grad_current,
            n_elements, timesteps,
            ctx.beta, ctx.threshold, ctx.slope,
            BLOCK=block,
            RESET_ZERO=(ctx.reset == "zero"),
        )
        return grad_current, None, None, None, None


def fused_lif(
    current: torch.Tensor,
    *,
    beta: float = 0.95,
    threshold: float = 1.0,
    slope: float = 25.0,
    reset: str = "zero",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused LIF over a `(T, B, N)` input current.

    Falls back to the reference loop when Triton is unavailable or the tensor
    is on CPU, so calling code does not have to branch. The fallback is
    numerically identical, just slower, and `TRITON_AVAILABLE` says which path
    ran.

    Args:
        current: Input current, shape `(timesteps, batch, neurons)`.
        beta: Membrane decay in `[0, 1)`.
        threshold: Firing threshold.
        slope: Surrogate gradient steepness.
        reset: "zero" or "subtract". Compiled as a constexpr, so each
            convention gets its own kernel. snnTorch defaults to "subtract";
            pass it to match.

    Returns:
        `(spikes, membrane)`, both `(T, B, N)`.
    """
    if current.dim() != 3:
        raise ValueError(
            f"expected (timesteps, batch, neurons), got {tuple(current.shape)}")
    if not 0.0 <= beta < 1.0:
        raise ValueError(f"beta must be in [0, 1), got {beta}")
    if reset not in ("zero", "subtract"):
        raise ValueError(f"reset must be 'zero' or 'subtract', got {reset!r}")

    if not TRITON_AVAILABLE or not current.is_cuda:
        return lif_sequence(current, beta=beta, threshold=threshold,
                            slope=slope, reset=reset)

    result = _FusedLIF.apply(  # type: ignore[no-untyped-call]
        current, beta, threshold, slope, reset)
    return cast("tuple[torch.Tensor, torch.Tensor]", result)
