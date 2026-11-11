from flax import linen as nn


class PhaseSeparated(nn.Module):
    r"""Combine separate models for log amplitude and phase.

    Both models must return real values. 
    The combined model returns the log wavefunction:

    .. math::

        \log \psi(x) = A(x) + i\phi(x).

    Thus the sampling probability is proportional to ``exp(2 * amplitude(x))``.
    """

    amplitude: nn.Module
    """Model returning the log amplitude."""

    phase: nn.Module
    """Model returning the phase in radians."""

    def __call__(self, x):
        log_amplitude = self.amplitude(x)
        phase = self.phase(x)
        return log_amplitude + 1j * phase
