from functools import partial

import jax
from jax import numpy as jnp

from netket.utils import struct

from .base import Sampler, SamplerState

class GibbsSamplerState(SamplerState):
    """Current visible configurations and random key."""

    sigma: jax.Array
    rng: jax.Array

    def __init__(self, sigma, rng):
        self.sigma = sigma
        self.rng = rng
        super().__init__()

class GibbsSampler(Sampler):

    # n_chains, sweep_size are additional to the required sampler argument (hilbert), optional 
    n_chains: int = struct.field(pytree_node=False)
    sweep_size: int = struct.field(pytree_node=False)

    # default machine_pow (the coefficient alpha of the exp argument in pdf) = 2,
    # not passed to super()
    def __init__(self, hilbert, *, n_chains=16, sweep_size=1):
        super().__init__(hilbert)
        self.n_chains = n_chains
        self.sweep_size = sweep_size

    def _init_state(self, machine, params, seed) -> SamplerState:
        key, init_key = jax.random.split(seed)
        sigma = self.hilbert.random_state(
            init_key, size=self.n_chains, dtype=self.dtype
            )
        return GibbsSamplerState(sigma, key)
