import jax
import jax.numpy as jnp
from flax.struct import dataclass
from functools import partial

from .base import Neuron


# region neurontype1

# standard neuron with weight decay

@dataclass
class HyperConfig1:
    nd: int # number of dendrites
    ns: int # number of synapses per dendrite
    n: float # (average) number of selected dendrites for each update
    b: float # threshold for dendritic nonlinearity
    bk: float # target dendritic activation
    beta: float # weight decay rate

@dataclass
class HyperDynConfig1:
    b: float # threshold for dendritic nonlinearity
    bk: float # target dendritic activation
    beta: float # weight decay rate

@dataclass
class HyperDynConfig1_SynapticBound(HyperDynConfig1):
    reset_from: float # if the absolute weight exceeds this value, it is reset to reset_to
    reset_to: float # the value to which the weight is reset if it exceeds reset_from

@dataclass
class State1:
    w: jnp.ndarray # shape (nd, ns)

class Neuron_Decay_Int(Neuron[HyperConfig1, State1, HyperDynConfig1]): 
    def __init__(self, hyperparameters: HyperConfig1):
        super().__init__(hyperparameters)

    @partial(jax.jit, static_argnames=('self',))
    def init_state(self, rng, hyperD: HyperDynConfig1): 
        w = jax.random.normal(rng, (self.hyper.nd, self.hyper.ns)) / jnp.sqrt(self.hyper.ns)
        return State1(w=w)
    
    @partial(jax.jit, static_argnames=('self',))
    def _forward(self, hyperD: HyperDynConfig1, state: State1, x: jnp.ndarray): # x: (B, nd, ns)
        w = state.w # (nd, ns)
        u = jnp.sum(w * x, axis=-1) # (B, nd)
        z = jax.nn.relu(u - hyperD.b) # (B, nd)
        y = jnp.sum(z, axis=-1) # (B,)
        return y
    
    @partial(jax.jit, static_argnames=('self',))
    def _update(self, hyperD: HyperDynConfig1, state: State1, x: jnp.ndarray): # x: (nd, ns)
        w = state.w # (nd, ns)
        ns, n = self.hyper.ns, self.hyper.n
        beta, bk = hyperD.beta, hyperD.bk
        overlaps = jnp.sum(w*x, axis=-1) # dim = (nd, )
        updated_overlap, updated_idx = jax.lax.top_k(overlaps, n)
        x_len_updated2 = jnp.linalg.norm(x[updated_idx], axis=-1, ord=2, keepdims=True)**2
        updated_w = w[updated_idx] + jnp.heaviside(bk-updated_overlap, 0.)[:, None] * ( x[updated_idx]/x_len_updated2*(bk-(1-beta/ns)*updated_overlap)[:, None]- beta/ns*w[updated_idx] ) # dim = (n, ns)
        w = w.at[updated_idx].set(updated_w)
        new_state = State1(w=w)
        return new_state
    
class Neuron_Decay_NonInt(Neuron_Decay_Int):
    def __init__(self, hyperparameters: HyperConfig1):
        super().__init__(hyperparameters)

    def _update(self, hyperD: HyperDynConfig1, state: State1, x: jnp.ndarray): # x: (nd, ns)
        w = state.w # (nd, ns)
        n, nd, ns = self.hyper.n, self.hyper.nd, self.hyper.ns
        bk, beta = hyperD.bk, hyperD.beta
        bla = jnp.sqrt(2.)*jax.scipy.special.erfinv(1-2*n/nd) # bias - lambda
        x_len2 = jnp.linalg.norm(x, axis=-1, ord=2, keepdims=True)**2
        overlaps = jnp.sum(w*x, axis=-1) # dim = (nd, )
        mask = jnp.heaviside(bk-overlaps, 0)*jnp.heaviside(overlaps-bla, 0.) # dim = (nd, )
        w = w + mask[:, None] * ( x/x_len2*(bk-(1-beta/ns)*overlaps)[:, None] - beta/ns*w ) # dim = (nd, ns)
        new_state = State1(w=w)
        return new_state


class Neuron_Decay_Int_SynapticBound(Neuron_Decay_Int):
    def __init__(self, hyperparameters: HyperConfig1):
        super().__init__(hyperparameters)

    @partial(jax.jit, static_argnames=('self',))
    def _update(self, hyperD: HyperDynConfig1_SynapticBound, state: State1, x: jnp.ndarray): # x: (nd, ns)
        w = state.w # (nd, ns)
        ns, n = self.hyper.ns, self.hyper.n
        beta, bk, reset_from, reset_to = hyperD.beta, hyperD.bk, hyperD.reset_from, hyperD.reset_to
        reset_from, reset_to = reset_from/jnp.sqrt(ns), reset_to/jnp.sqrt(ns)
        overlaps = jnp.sum(w*x, axis=-1) # dim = (nd, )
        updated_overlap, updated_idx = jax.lax.top_k(overlaps, n)
        x_len_updated2 = jnp.linalg.norm(x[updated_idx], axis=-1, ord=2, keepdims=True)**2
        updated_w = w[updated_idx] + jnp.heaviside(bk-updated_overlap, 0.)[:, None] * ( x[updated_idx]/x_len_updated2*(bk-(1-beta/ns)*updated_overlap)[:, None]- beta/ns*w[updated_idx] ) # dim = (n, ns)
        updated_w = jnp.where(jnp.abs(updated_w) > reset_from, reset_to*jnp.sign(updated_w), updated_w)
        w = w.at[updated_idx].set(updated_w)
        new_state = State1(w=w)
        return new_state
    
class Neuron_Decay_Int_NoNonLinearity(Neuron_Decay_Int):
    def __init__(self, hyperparameters: HyperConfig1):
        super().__init__(hyperparameters)

    @partial(jax.jit, static_argnames=('self',))
    def _forward(self, hyperD: HyperDynConfig1, state: State1, x: jnp.ndarray): # x: (B, nd, ns)
        w = state.w # (nd, ns)
        u = jnp.sum(w * x, axis=-1) # (B, nd)
        y = jnp.sum(u, axis=-1) # (B,)
        return y

# endregion neurontype1

# region neurontype2

# standard neuron with weight decay and adaptive thresholds: b -> b+shift, bk -> bk+shift
# note that the update does not depend on b, so b_scan is still possible. 

@dataclass
class HyperConfig2:
    nd: int # number of dendrites
    ns: int # number of synapses per dendrite
    n: float # (average) number of selected dendrites for each update
    b: float # threshold for dendritic nonlinearity
    bk: float # target dendritic activation
    beta: float # weight decay rate
    gamma: float # decay factor for frequency estimation, recommended value: 1-f/k = 1- f/2, where f is the dendrite's expected update frequency, then <freq>^2 / Var[freq] is roughly 2k.

    delta_th: float # if freq is outside [f-delta_th, f+delta_th], then shift is updated. recommended value: delta_th=2*sqrt(var(freq)) = 2*sqrt(f(1-f)(1-gamma)/(1+gamma))
    delta: float # the step size of shift update

@dataclass
class HyperDynConfig2:
    b: float # threshold for dendritic nonlinearity
    bk: float # target dendritic activation
    beta: float # weight decay rate
    gamma: float # decay factor for frequency estimation, recommended value: 1-f/k = 1- f/2, where f is the dendrite's expected update frequency, then <freq>^2 / Var[freq] is roughly 2k.
    delta_th: float # if freq is outside [f-delta_th, f+delta_th], then shift is updated. recommended value: delta_th=2*sqrt(var(freq)) = 2*sqrt(f(1-f)(1-gamma)/(1+gamma))
    delta: float # the step size of shift update

@dataclass
class State2:
    w: jnp.ndarray # shape (nd, ns)
    shift: jnp.ndarray # shape (nd,), additional shift applied to w*x
    freq: jnp.ndarray # shape (nd,), estimated frequency of updates for each dendrite

class Neuron_Decay_Int2(Neuron[HyperConfig2, State2, HyperDynConfig2]): 
    def __init__(self, hyperparameters: HyperConfig2):
        super().__init__(hyperparameters)

    @partial(jax.jit, static_argnames=('self',))
    def init_state(self, rng, hyperD: HyperDynConfig2): 
        w = jax.random.normal(rng, (self.hyper.nd, self.hyper.ns)) / jnp.sqrt(self.hyper.ns)
        shift = jnp.zeros(self.hyper.nd)
        freq = jnp.ones(self.hyper.nd) * (self.hyper.n / self.hyper.nd) # initial frequency estimate
        return State2(w=w, shift=shift, freq=freq)

    @partial(jax.jit, static_argnames=('self',))
    def _forward(self, hyperD: HyperDynConfig2, state: State2, x: jnp.ndarray): # x: (B, nd, ns)
        w = state.w # (nd, ns)
        shift = state.shift # (nd,)
        u = jnp.sum(w * x, axis=-1) # (B, nd)
        z = jax.nn.relu(u - hyperD.b - shift) # (B, nd)
        y = jnp.sum(z, axis=-1) # (B,)
        return y
    
    @partial(jax.jit, static_argnames=('self',))
    def _update(self, hyperD: HyperDynConfig2, state: State2, x: jnp.ndarray): # x: (nd, ns)
        w, shift, freq = state.w, state.shift, state.freq # (nd, ns), (nd,), (nd,)
        ns, nd, n = self.hyper.ns, self.hyper.nd, self.hyper.n
        beta, bk, gamma, delta_th, delta = hyperD.beta, hyperD.bk, hyperD.gamma, hyperD.delta_th, hyperD.delta

        f_exp = n/nd
        shift_updateQ = jnp.abs(freq-f_exp) > delta_th
        shift += shift_updateQ * jnp.sign(freq-f_exp) * delta
        shift = shift - jnp.mean(shift) # keep mean(shift)=0

        overlaps = jnp.sum(w*x, axis=-1) # (nd, )
        _, updated_idx = jax.lax.top_k(overlaps-shift, n)
        updated_overlap = overlaps[updated_idx] # (n, )
        updated_shift = shift[updated_idx] # (n, )
        x_len_updated2 = jnp.linalg.norm(x[updated_idx], axis=-1, ord=2, keepdims=True)**2
        updated_w = w[updated_idx] + jnp.heaviside(bk+updated_shift-updated_overlap, 0.)[:, None] * ( x[updated_idx]/x_len_updated2*(bk+updated_shift-(1-beta/ns)*updated_overlap)[:, None]- beta/ns*w[updated_idx] ) # dim = (n, ns)
        w = w.at[updated_idx].set(updated_w)

        freq = freq * gamma
        freq = freq.at[updated_idx].add(1-gamma)

        new_state = State2(w=w, shift=shift, freq=freq)
        return new_state

# endregion neurontype2