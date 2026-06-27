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
    
class Normal_random_scaled(Xs_Generator):
    def __init__(self, nd: int, ns: int, rng: jax.Array):
        super().__init__()
        self.nd = nd
        self.ns = ns
        self.scale_mat = jax.random.normal(rng, (nd, ns))
        self.scale_mat = self.scale_mat / jnp.linalg.norm(self.scale_mat, axis=-1, ord=2, keepdims=True) * jnp.sqrt(ns)
        self.scale_mat = jnp.abs(self.scale_mat)

    @partial(jax.jit, static_argnames=('self', 'n_patterns'))
    def _generate(self, rng, n_patterns: int):
        rng1, rng2 = jax.random.split(rng)
        x = jax.random.normal(rng1, (n_patterns, self.nd, self.ns))
        x = x * self.scale_mat[None, :, :]
        return x

class Normal_Corr_across_Dendrites(Xs_Generator):
    def __init__(self, nd, ns, coeff, key):
        """coeff: the ratio between number of axons and number of synapses per dendrite."""
        super().__init__()
        self.nd = nd
        self.ns = ns
        self.coeff = coeff
        self.nA = int(ns*coeff) # number of axons
        self.key, subkey = jax.random.split(key)
        self.conn_idx_mat = self._get_connectivity_idx_mat(subkey)

    def _get_connectivity_idx_mat(self, key) -> jnp.ndarray:
        key_list = jax.random.split(key, self.nd)
        fun = lambda key: jax.random.choice(key, jnp.arange(self.nA), (self.ns, ), replace=False) # each axon is chosen only once per dendrite
        conn_mat = jax.vmap(fun, in_axes=0)(key_list) # dim = (nd, ns)
        return conn_mat

    @partial(jax.jit, static_argnames=('self', 'n_patterns'))
    def _generate(self, rng, n_patterns: int):
        axon_inputs = jax.random.normal(rng, (n_patterns, self.nA))
        return axon_inputs[:, self.conn_idx_mat]
    
class Sparse_Binomial(Xs_Generator):
    def __init__(self, nd, ns, fA=0.1, mu=4, n_burst=4, key=jax.random.PRNGKey(0)):
        """n_burst: the maximum number of spikes in a burst
        mu: the expected number of spikes in a burst
        fA: the sparsity of the input signal
        rescale: rescale each element so that it has unit variance"""
        super().__init__()
        self.nd = nd
        self.ns = ns
        self.fA = fA
        self.mu = mu
        self.n_burst = n_burst
        self.key = key
        self.rescale_factor = 1/jnp.sqrt(fA*(mu*(1-mu/n_burst) + mu*mu*(1-fA)))

    @partial(jax.jit, static_argnames=('self', 'n_patterns'))
    def _generate(self, rng, n_patterns: int):
        key1, key2 = jax.random.split(rng)
        xs1 = jax.random.choice(key1, jnp.array([0., 1.]), (n_patterns, self.nd, self.ns), p=jnp.array([1-self.fA, self.fA]))
        xs2 = jax.random.binomial(key2, self.n_burst, self.mu/self.n_burst, (n_patterns, self.nd, self.ns))
        xs = xs1*xs2 - self.fA*self.mu
        xs = xs*self.rescale_factor
        return xs

class Normal_VariableNs(Xs_Generator):
    def __init__(self, nd: int, ns_max: int):
        super().__init__()
        self.nd = nd
        self.ns_max = ns_max

    def get_mask(self, ns_array: jax.Array):
        """ns_array: shape (nd, ), each element indicates the number of synapses for that dendrite."""
        mask = jnp.zeros((self.nd, self.ns_max))
        for i in range(self.nd):
            mask = mask.at[i, :ns_array[i]].set(1.0)
        return mask # shape (nd, ns_max)

    @partial(jax.jit, static_argnames=('self', 'n_patterns'))
    def _generate(self, rng, n_patterns: int, mask: jax.Array):
        x = jax.random.normal(rng, (n_patterns, self.nd, self.ns_max))
        x = x * mask[None, :, :]
        return x

