from functools import partial

import jax
from jax import numpy as jnp

from netket.utils import struct

from .base import Sampler, SamplerState

def _sample_spins(key, field, dtype):
    """Draw independent +/-1 spins with P(+1) = sigmoid(2 * field)."""
    up = jax.random.bernoulli(key, jax.nn.sigmoid(2 * field)) # array of bools, independent trials according to sigmoid pdf
    return jnp.where(up, 1, -1).astype(dtype) # flip spins when up (trial outcome) = True

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

    def _init_state(self, machine, params, key) -> SamplerState:
        key, init_key = jax.random.split(key)
        sigma = self.hilbert.random_state(
            init_key, size=self.n_chains, dtype=self.dtype
            )
        return GibbsSamplerState(sigma, key)

    def _reset(self, model, variables, state):
        return state

    @partial(
        jax.jit, static_argnames=("model", "chain_length", "return_log_probabilities")
    )
    def _sample_chain(
        self, model, variables, state, chain_length, return_log_probabilities=False
    ):
        # unpack nested dict 
        params = variables["params"]["amplitude"] # pick only amplitude model params
        weights = params["Dense"]["kernel"] # W matrix
        hidden_bias = params["Dense"].get("bias", 0) # b vector, default zero if hidden bias disabled in RBM
        visible_bias = params.get("visible_bias", 0) # a vector, default zero if visible " " "

        def block_update(_, state): # _ is unused loop index for jax.lax.fori_loop

            key, key_h1, key_h2, key_s = jax.random.split(state.rng, 4)

            hidden_field = jnp.dot(state.sigma, weights) + hidden_bias 
            # ^i.e. hidden theta_j "angle" variables

            hidden_1 = _sample_spins(key_h1, hidden_field, weights.dtype)
            hidden_2 = _sample_spins(key_h2, hidden_field, weights.dtype)

            # two hidden copies needed because |psi|^2 is double sum over h, h'
            visible_field = 2 * visible_bias + jnp.dot(
                hidden_1 + hidden_2, weights.T,
            ) # visible "angle" variables in sigmoid pdf

            sigma = _sample_spins(key_s, visible_field, self.dtype)

            # return updated GibbsSamplerState
            return state.replace(sigma=sigma, rng=key)

        def save_sample(state, _):
            """Loop block_update over state sweep_size times"""
            state = jax.lax.fori_loop(0, self.sweep_size, block_update, state)
            return state, state.sigma

        state, samples = jax.lax.scan(save_sample, state, None, length=chain_length)
        samples = jnp.swapaxes(samples, 0, 1)  # (chains, length, sites)

        if return_log_probabilities:
            amplitude = model.amplitude.apply(
                {"params": variables["params"]["amplitude"]}, samples
            )
            return (samples, 2 * amplitude), state

        return samples, state

