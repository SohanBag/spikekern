"""Tests for spikekern.

One property dominates: **the fused kernel must compute what the reference
computes, forward and backward.** A kernel that is fast and subtly wrong is
worse than no kernel, because the network still trains, still converges to
something, and nothing reports the difference.

The backward tests earned their place. The first version of the backward kernel
omitted the reset path — the gradient flowing from `U[t+1]` back through
`S[t]`, whose derivative is `-beta*U[t]` for reset-to-zero and `-threshold` for
subtract. The forward pass was bit-exact throughout, and gradients were wrong by
a relative 0.94.
"""

from __future__ import annotations

import pytest
import torch

from spikekern import (
    TRITON_AVAILABLE,
    fused_lif,
    lif_sequence,
    lif_step,
    spike,
    triton_unavailable_reason,
)

CUDA = torch.cuda.is_available()
needs_gpu = pytest.mark.skipif(
    not (CUDA and TRITON_AVAILABLE),
    reason=f"needs CUDA and Triton ({triton_unavailable_reason() or 'no device'})",
)
RESETS = ("zero", "subtract")


def current(timesteps: int = 32, batch: int = 8, neurons: int = 128,
            *, device: str = "cpu", seed: int = 0) -> torch.Tensor:
    """Input current strong enough that neurons actually fire.

    Scale matters: too small and nothing crosses threshold, so the spike
    comparison passes trivially on two all-zero tensors.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(timesteps, batch, neurons, generator=generator) * 0.6
    return x.to(device)


# ===========================================================================
# the reference, on its own terms
# ===========================================================================


def test_reference_actually_spikes():
    """A test comparing two silent networks proves nothing."""
    spikes, _ = lif_sequence(current(), beta=0.9)
    rate = spikes.mean().item()
    assert 0.01 < rate < 0.9, f"firing rate {rate:.3f} is degenerate"


def test_reference_membrane_resets_after_a_spike():
    """Reset-to-zero must actually zero the decay term."""
    strong = torch.full((1, 1, 1), 5.0)
    spikes, membrane = lif_sequence(strong, beta=0.9, reset="zero")
    assert spikes[0, 0, 0].item() == 1.0
    assert membrane[0, 0, 0].item() == pytest.approx(5.0)


def test_subtract_and_zero_are_different_dynamics():
    """If these agreed, the reset parameter would be doing nothing."""
    x = current()
    zero_spikes, _ = lif_sequence(x, beta=0.9, reset="zero")
    sub_spikes, _ = lif_sequence(x, beta=0.9, reset="subtract")
    assert not torch.equal(zero_spikes, sub_spikes)


def test_surrogate_gradient_is_largest_at_threshold():
    """The point of the fast sigmoid: gradient concentrated near firing."""
    membrane = torch.tensor([-5.0, 0.0, 1.0, 2.0, 8.0], requires_grad=True)
    spike(membrane, threshold=1.0, slope=25.0).sum().backward()
    gradient = membrane.grad
    assert gradient is not None
    assert gradient.argmax().item() == 2, "peak is not at the threshold"
    assert gradient[0] < gradient[2] and gradient[-1] < gradient[2]


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_beta_outside_the_unit_interval_is_rejected(bad):  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="beta must be"):
        lif_sequence(current(4, 2, 4), beta=bad)


def test_wrong_rank_is_rejected():
    with pytest.raises(ValueError, match="timesteps, batch, neurons"):
        lif_sequence(torch.zeros(8, 4))


def test_unknown_reset_is_rejected():
    with pytest.raises(ValueError, match="reset must be"):
        lif_sequence(current(4, 2, 4), reset="decay")


def test_lif_step_rejects_an_unknown_reset():
    zeros = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="reset must be"):
        lif_step(zeros, zeros, zeros, reset="nonsense")


# ===========================================================================
# the fused kernel against the reference
# ===========================================================================


@needs_gpu
@pytest.mark.cuda
@pytest.mark.parametrize("reset", RESETS)
def test_forward_matches_the_reference_exactly(reset):  # type: ignore[no-untyped-def]
    x = current(device="cuda")
    fused_spikes, fused_membrane = fused_lif(x, beta=0.9, reset=reset)
    ref_spikes, ref_membrane = lif_sequence(x, beta=0.9, reset=reset)

    assert torch.equal(fused_spikes, ref_spikes), "spike trains differ"
    assert (fused_membrane - ref_membrane).abs().max().item() < 1e-5


@needs_gpu
@pytest.mark.cuda
@pytest.mark.parametrize("reset", RESETS)
@pytest.mark.parametrize("shape", [(8, 4, 64), (32, 8, 128), (64, 16, 256)])
def test_backward_matches_the_reference(reset, shape):  # type: ignore[no-untyped-def]
    """The test that caught the missing reset path.

    Without that term the relative error was 0.94, so the tolerance here is
    tight enough to fail loudly rather than absorb a wrong derivative.
    """
    x = current(*shape, device="cuda")

    a = x.clone().requires_grad_(True)
    spikes, membrane = lif_sequence(a, beta=0.9, reset=reset)
    (spikes.sum() + 0.5 * membrane.sum()).backward()

    b = x.clone().requires_grad_(True)
    fused_spikes, fused_membrane = fused_lif(b, beta=0.9, reset=reset)
    (fused_spikes.sum() + 0.5 * fused_membrane.sum()).backward()

    assert a.grad is not None and b.grad is not None
    scale = max(a.grad.abs().max().item(), 1e-12)
    relative = (a.grad - b.grad).abs().max().item() / scale
    assert relative < 1e-4, f"relative gradient error {relative:.3e}"


@needs_gpu
@pytest.mark.cuda
def test_gradient_is_not_trivially_zero():
    """A backward test passes vacuously if both gradients are zero."""
    x = current(device="cuda")
    a = x.clone().requires_grad_(True)
    spikes, membrane = fused_lif(a, beta=0.9)
    (spikes.sum() + 0.5 * membrane.sum()).backward()

    assert a.grad is not None
    assert a.grad.abs().mean().item() > 1e-3


@needs_gpu
@pytest.mark.cuda
@pytest.mark.parametrize("timesteps", [1, 2, 7, 33])
def test_awkward_timestep_counts(timesteps):  # type: ignore[no-untyped-def]
    """Including T=1, where the recurrence never runs."""
    x = current(timesteps, 4, 96, device="cuda")
    fused_spikes, _ = fused_lif(x, beta=0.9)
    ref_spikes, _ = lif_sequence(x, beta=0.9)
    assert torch.equal(fused_spikes, ref_spikes)


@needs_gpu
@pytest.mark.cuda
def test_a_size_that_does_not_divide_the_block():
    """Masked lanes must not contribute. BLOCK is 1024; 1000 is not a multiple."""
    x = current(16, 5, 200, device="cuda")   # 5 * 200 = 1000 elements
    fused_spikes, fused_membrane = fused_lif(x, beta=0.9)
    ref_spikes, ref_membrane = lif_sequence(x, beta=0.9)
    assert torch.equal(fused_spikes, ref_spikes)
    assert (fused_membrane - ref_membrane).abs().max().item() < 1e-5


@needs_gpu
@pytest.mark.cuda
def test_non_contiguous_input_is_handled():
    """A transposed view must not be read as if it were contiguous."""
    base = current(32, 16, 128, device="cuda")
    view = base.transpose(1, 2).transpose(1, 2)   # same values, new strides
    fused_spikes, _ = fused_lif(view, beta=0.9)
    ref_spikes, _ = lif_sequence(base, beta=0.9)
    assert torch.equal(fused_spikes, ref_spikes)


# ===========================================================================
# against snnTorch, the library people actually use
# ===========================================================================


@needs_gpu
@pytest.mark.cuda
@pytest.mark.parametrize("reset", RESETS)
def test_matches_snntorch(reset):  # type: ignore[no-untyped-def]
    """Drop-in equivalence with snn.Leaky, in both reset conventions.

    snnTorch defaults to "subtract"; a kernel that only matched "zero" would
    silently change the dynamics for most people who swapped it in.
    """
    snn = pytest.importorskip("snntorch")

    x = current(device="cuda")
    lif = snn.Leaky(beta=0.9, threshold=1.0, reset_mechanism=reset).cuda()
    membrane = torch.zeros_like(x[0])
    spikes, membranes = [], []
    for t in range(x.shape[0]):
        emitted, membrane = lif(x[t], membrane)
        spikes.append(emitted)
        membranes.append(membrane)

    native_spikes = torch.stack(spikes)
    native_membrane = torch.stack(membranes)
    fused_spikes, fused_membrane = fused_lif(x, beta=0.9, reset=reset)

    assert torch.equal(native_spikes, fused_spikes)
    assert (native_membrane - fused_membrane).abs().max().item() < 1e-5


# ===========================================================================
# the fallback
# ===========================================================================


def test_cpu_input_falls_back_to_the_reference():
    """No branch required of the caller, and the result is identical."""
    x = current(16, 4, 64)
    fused_spikes, fused_membrane = fused_lif(x, beta=0.9)
    ref_spikes, ref_membrane = lif_sequence(x, beta=0.9)
    assert torch.equal(fused_spikes, ref_spikes)
    assert torch.equal(fused_membrane, ref_membrane)


def test_cpu_fallback_is_differentiable():
    x = current(8, 2, 32).requires_grad_(True)
    spikes, membrane = fused_lif(x, beta=0.9)
    (spikes.sum() + 0.5 * membrane.sum()).backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_fused_validates_its_arguments():
    with pytest.raises(ValueError, match="reset must be"):
        fused_lif(current(4, 2, 4), reset="bogus")
    with pytest.raises(ValueError, match="beta must be"):
        fused_lif(current(4, 2, 4), beta=1.0)
    with pytest.raises(ValueError, match="timesteps, batch, neurons"):
        fused_lif(torch.zeros(4, 4))


def test_unavailable_reason_is_a_string():
    assert isinstance(triton_unavailable_reason(), str)
    if TRITON_AVAILABLE:
        assert triton_unavailable_reason() == ""
