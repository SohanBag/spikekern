"""spikekern: fused Triton kernels for LIF spiking neuron dynamics."""

from __future__ import annotations

from .fused import TRITON_AVAILABLE, fused_lif, triton_unavailable_reason
from .reference import SurrogateSpike, lif_sequence, lif_step, spike

__version__ = "0.1.0"

__all__ = [
    "TRITON_AVAILABLE",
    "SurrogateSpike",
    "__version__",
    "fused_lif",
    "lif_sequence",
    "lif_step",
    "spike",
    "triton_unavailable_reason",
]
