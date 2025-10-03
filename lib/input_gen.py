import jax
import jax.numpy as jnp
from flax.struct import dataclass
from functools import partial

from .base import Xs_Generator


class Normal(Xs_Generator):
    def __init__(self, nd: int, ns: int):
        super().__init__()
        self.nd = nd
        self.ns = ns

    @partial(jax.jit, static_argnames=('self', 'n_patterns'))
    def _generate(self, rng, n_patterns: int):
        x = jax.random.normal(rng, (n_patterns, self.nd, self.ns))
        return x