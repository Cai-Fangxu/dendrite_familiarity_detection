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
    age_max: int # maximum age for each dendrite, when max is reached, the age is set back to 0. 
    gamma: float # decay rate for var(w*x) estimation
    shift: float # dendritic threshold b and bk are shited by shift*sqrt(var(w*x))

@dataclass
class HyperDynConfig2:
    b: float # threshold for dendritic nonlinearity
    bk: float # target dendritic activation
    beta: float # weight decay rate
    age_max: int # maximum age for each dendrite, when max is reached, the age is set back to 0
    gamma: float # decay rate for var(w*x) estimation
    shift: float # dendritic threshold b and bk are shited by shift*sqrt(var(w*x))


@dataclass
class State2:
    w: jnp.ndarray # shape (nd, ns)
    wx2: jnp.ndarray # shape (nd,), running estimate of E[(w*x)^2] for each dendrite
    age: jnp.ndarray # shape (nd,), number of updates since the last time the dendrite is selected

class Neuron_Decay_Int2(Neuron[HyperConfig2, State2, HyperDynConfig2]): 
    def __init__(self, hyperparameters: HyperConfig2):
        super().__init__(hyperparameters)

    @partial(jax.jit, static_argnames=('self',))
    def init_state(self, rng, hyperD: HyperDynConfig2): 
        w = jax.random.normal(rng, (self.hyper.nd, self.hyper.ns)) / jnp.sqrt(self.hyper.ns)
        wx2 = jnp.ones(self.hyper.nd, dtype=jnp.float32) # initial estimate of E[(w*x)^2]
        age = jnp.full((self.hyper.nd,), hyperD.age_max, dtype=jnp.int32) # set initial age to age_max, so that all dendrites can be selected at the first update
        return State2(w=w, wx2=wx2, age=age)

    @partial(jax.jit, static_argnames=('self',))
    def _forward(self, hyperD: HyperDynConfig2, state: State2, x: jnp.ndarray): # x: (B, nd, ns)
        w = state.w # (nd, ns)
        wx2 = state.wx2 # (nd,)
        th_shift = hyperD.shift * jnp.sqrt(wx2) # (nd,)
        u = jnp.sum(w * x, axis=-1) # (B, nd)
        z = jax.nn.relu(u - hyperD.b - th_shift) # (B, nd)
        y = jnp.sum(z, axis=-1) # (B,)
        return y
    
    @partial(jax.jit, static_argnames=('self',))
    def _update(self, hyperD: HyperDynConfig2, state: State2, x: jnp.ndarray): # x: (nd, ns)
        w, wx2, age = state.w, state.wx2, state.age # (nd, ns), (nd,), (nd,)
        ns, nd, n = self.hyper.ns, self.hyper.nd, self.hyper.n
        beta, bk, age_max, gamma, shift = hyperD.beta, hyperD.bk, hyperD.age_max, hyperD.gamma, hyperD.shift

        th_shift = shift * jnp.sqrt(wx2) # (nd,)
        overlaps = jnp.sum(w*x, axis=-1) # (nd, )
        overlaps_tmp = overlaps - (age!=age_max)*1e3 # if age != age_max, set overlaps to a very small value, so that it won't be selected
        _, updated_idx = jax.lax.top_k(overlaps_tmp-th_shift, n)
        updated_overlap = overlaps[updated_idx] # (n, )
        updated_shift = th_shift[updated_idx] # (n, )
        x_len_updated2 = jnp.linalg.norm(x[updated_idx], axis=-1, ord=2, keepdims=True)**2
        updated_w = w[updated_idx] + jnp.heaviside(bk+updated_shift-updated_overlap, 0.)[:, None] * ( x[updated_idx]/x_len_updated2*(bk+updated_shift-(1-beta/ns)*updated_overlap)[:, None]- beta/ns*w[updated_idx] ) # dim = (n, ns)
        updated_w = jnp.clip(updated_w, -3./jnp.sqrt(ns), 3./jnp.sqrt(ns)) # clip weights to avoid numerical issues
        w = w.at[updated_idx].set(updated_w)

        # update age
        age = jnp.minimum(age+1, age_max)
        age = age.at[updated_idx].set(0)
        # update wx2
        wx2 = gamma*wx2 + (1-gamma)*(overlaps**2)
        return State2(w=w, wx2=wx2, age=age)

# endregion neurontype2