import jax
import jax.numpy as jnp
from flax.struct import dataclass
from functools import partial

from .base import Neuron


# region neurontype1

@dataclass
class HyperConfig1:
    nd: int # number of dendrites
    ns: int # number of synapses per dendrite
    n: float # (average) number of selected dendrites for each update
    b: float # threshold for dendritic nonlinearity
    bk: float # target dendritic activation
    beta: float # weight decay rate

@dataclass
class State1:
    w: jnp.ndarray # shape (nd, ns)

class Neuron_Decay_Int(Neuron): 
    def __init__(self, hyperparameters: HyperConfig1):
        super().__init__(hyperparameters)

    def init_state(self, rng):
        w = jax.random.normal(rng, (self.hyper.nd, self.hyper.ns)) / jnp.sqrt(self.hyper.ns)
        return State1(w=w)
    
    @partial(jax.jit, static_argnames=('self',))
    def _forward(self, state: State1, x: jnp.ndarray): # x: (B, nd, ns)
        w = state.w # (nd, ns)
        u = jnp.sum(w * x, axis=-1) # (B, nd)
        z = jax.nn.relu(u - self.hyper.b) # (B, nd)
        y = jnp.sum(z, axis=-1) # (B,)
        return y
    
    @partial(jax.jit, static_argnames=('self',))
    def _update(self, state: State1, x: jnp.ndarray): # x: (nd, ns)
        w = state.w # (nd, ns)
        beta, bk = self.hyper.b, self.hyper.beta, self.hyper.bk
        overlaps = jnp.sum(w*x, axis=-1) # dim = (nd, )
        updated_overlap, updated_idx = jax.lax.top_k(overlaps, self.ndR)
        x_len_updated2 = jnp.linalg.norm(x[updated_idx], axis=-1, ord=2, keepdims=True)**2
        updated_w = w[updated_idx] + jnp.heaviside(bk-updated_overlap, 0.).reshape((-1, 1))*(x[updated_idx]/x_len_updated2*(bk-(1-beta/self.ns)*updated_overlap).reshape((-1, 1)) - beta/self.ns*w[updated_idx]) # dim = (ndR, ns)
        w = w.at[updated_idx].set(updated_w)
        new_state = State1(w=w)
        return new_state
    
# class Neuron_Decay_NonInt(Neuron_Decay_Int):
#     def __init__(self, hyperparameters: HyperConfig1):
#         super().__init__(hyperparameters)

#     def _update(self, state: State1, x: jnp.ndarray): # x: (nd, ns)
#         w = state.w # (nd, ns)
#         b, bk, beta = self.hyper.b, self.hyper.bk, self.hyper.beta
#         la = b - jnp.sqrt(2.)*jax.scipy.special.erfcinv(2*self.hyper.n/self.hyper.nd) # lambda
#         w, b, kappa, beta, la = params["w"], params["b"], params["kappa"], params["beta"], params["la"]
#         x_len2 = jnp.linalg.norm(x, axis=-1, ord=2, keepdims=True)**2
#         overlaps = jnp.sum(w*x, axis=-1) # dim = (nd, )
#         mask = jnp.heaviside(kappa+b-overlaps, 0)*jnp.heaviside(overlaps-b+la, 0.) # dim = (nd, )
#         w = w + mask.reshape((-1, 1))*(x/x_len2*(b+kappa-(1-beta/self.ns)*overlaps).reshape((-1, 1)) - beta/self.ns*w) # dim = (ndR, ns)
#         params["w"] = w
#         return params

# endregion neurontype1

