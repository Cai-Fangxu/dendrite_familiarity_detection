import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass
from flax.struct import dataclasses
from functools import partial
from typing import Tuple

from .base import Neuron, Xs_Generator
from .neurons import HyperConfig1, State1, HyperDynConfig1, HyperConfig2, State2, HyperDynConfig2

# region drivers-for-Type1-neurons

class Driver1_bScan:
    def __init__(
            self, 
            neuron: Neuron[HyperConfig1, State1, HyperDynConfig1], 
            xs_generator: Xs_Generator,
            rng: jax.Array, 
            hyperD: HyperDynConfig1 = None,
            initial_state: State1 = None, 
            resample_every: int = 1000,
            n_tested_patterns: int = 1000, # n_tested_patterns should be a multiple of resample_every
            ):
        self.neuron = neuron
        self.xs_generator = xs_generator
        self.rng = rng
        self.hyperD = hyperD
        self.state = initial_state
        self.resample_every = resample_every
        self.n_tested_patterns = n_tested_patterns

        self.transient_record = None
        self.steady_record = None
        self.upper99_s = None

    def re_init(self, rng: jax.Array, hyperD: HyperDynConfig1=None, initial_state: State1 = None):
        """reset the driver without triggering recompilation"""
        self.state = initial_state
        self.rng = rng
        self.hyperD = hyperD
        self.transient_record = None
        self.steady_record = None
        self.upper99_s = None

    def _update_no_record(self, hyperD: HyperDynConfig1, state: State1, xs: jax.Array): # xs: (resample_every, nd, ns)
        """ Update the state by xs without recording the outputs"""
        def step_fn(carry, x): # carry = state
            new_carry = self.neuron.update(hyperD, carry, x)
            return new_carry, None
        new_state, _ = jax.lax.scan(step_fn, state, xs)
        return new_state

    def _update_record1(self, hyperD: HyperDynConfig1, state: State1, bs: jax.Array, xs: jax.Array, x0s: jax.Array): # bs: (bs_size,), xs: (resample_every, nd, ns), x0s: (n_tested_patterns, nd, ns)
        """ Run simulation for `resample_every` steps and record the outputs"""
        def step_fn(carry, x): # carry = state
            new_carry = self.neuron.update(hyperD, carry, x)
            w = new_carry.w # (nd, ns)
            us = jnp.sum(x0s * w, axis=-1) # (n_tested_patterns, nd)
            z = jax.nn.relu(us[None, :, :] - bs[:, None, None]) # (bs_size, n_tested_patterns, nd)
            y = jnp.sum(z, axis=-1) # (bs_size, n_tested_patterns)
            return new_carry, y
        new_state, ys = jax.lax.scan(step_fn, state, xs) # ys: (resample_every, bs_size, n_tested_patterns)
        return new_state, ys

    def _update_record3_sparse(self, hyperD: HyperDynConfig1, state: State1, bs: jax.Array, xs: jax.Array, x0s: jax.Array, test_every: int): # bs: (bs_size,), x0s: (n_tested_patterns, nd, ns)
        """ Run simulation for `resample_every` steps and record the outputs"""
        n_tested_patterns = x0s.shape[0]
        bs_size = bs.shape[0]
        def step_fn1(state, x): # carry = state
            new_state = self.neuron.update(hyperD, state, x)
            w = new_state.w # (nd, ns)
            us = jnp.sum(x0s * w, axis=-1) # (n_tested_patterns, nd)
            z = jax.nn.relu(us[None, :, :] - bs[:, None, None]) # (bs_size, n_tested_patterns, nd)
            y = jnp.sum(z, axis=-1) # (bs_size, n_tested_patterns)
            return new_state, y
        def step_fn2(state, x): # carry = state
            new_state = self.neuron.update(hyperD, state, x)
            y = jnp.full((bs_size, n_tested_patterns), -100.)
            return new_state, y
        def step_fn(carry, y): # carry = (state, idx)
            state, idx = carry
            new_state, y = jax.lax.cond(idx % test_every == 0, step_fn1, step_fn2, state, y)
            return (new_state, idx+1), y
        (new_state, _), ys = jax.lax.scan(step_fn, (state, 0), xs) # ys: (resample_every, bs_size, n_tested_patterns)
        return new_state, ys

    @partial(jax.jit, static_argnames=('self', 'n_steps'))
    def init_phase(self, rng: jax.Array, hyperD: HyperDynConfig1, state: State1, n_steps: int) -> State1:
        """ Initialize the state by `n_steps` patterns. 
            `n_steps` should be a multiple of resample_every.
        """
        n_big_steps = n_steps // self.resample_every
        rngs = jax.random.split(rng, n_big_steps)
        def step_fn(carry, rng): # carry = state
            xs = self.xs_generator.generate(rng, self.resample_every) # (resample_every, nd, ns)
            new_carry = self._update_no_record(hyperD, carry, xs)
            return new_carry, None
        new_state, _ = jax.lax.scan(step_fn, state, rngs)
        return new_state

    @partial(jax.jit, static_argnames=('self', 'n_steps'))
    def record_phase1(self, rng: jax.Array, hyperD: HyperDynConfig1, state: State1, n_steps: jax.Array, bs: jax.Array, x0s: jax.Array) -> Tuple[State1, jax.Array]: # bs: (bs_size,), x0s: (n_tested_patterns, nd, ns)
        """ Run simulation for `n_steps`, and for each step, test the state by `n_tested_patterns` patterns and record the outputs.
        `n_steps` should be a multiple of `resample_every`."""
        x0s_padded_len = (self.resample_every - self.n_tested_patterns) % self.resample_every
        x0s_padded = jnp.pad(x0s, ((0, x0s_padded_len), (0,0), (0,0)), mode='constant', constant_values=0.) # (n_tested_patterns + x0s_padded_len, nd, ns), first dim is a multiple of resample_every

        n_big_steps = n_steps // self.resample_every
        rngs = jax.random.split(rng, n_big_steps)
        def step_fn(carry, rng): # carry = (state, idx)
            state, idx = carry

            # generate xs
            xs = self.xs_generator.generate(rng, self.resample_every) # (resample_every, nd, ns)
            tmp_x0_padded = jax.lax.dynamic_slice(x0s_padded, (idx*self.resample_every, 0, 0), (self.resample_every, x0s.shape[1], x0s.shape[2])) # (resample_every, nd, ns)
            mask_array = ( jnp.arange(self.resample_every)+idx*self.resample_every ) < self.n_tested_patterns
            xs = jnp.where(mask_array[:, None, None], tmp_x0_padded, xs) # (resample_every, nd, ns)

            new_state, ys = self._update_record1(hyperD, state, bs, xs, x0s)
            return (new_state, idx+1), ys

        (new_state, _), ys = jax.lax.scan(step_fn, (state, 0), rngs) # ys: (n_big_steps, resample_every, bs_size, n_tested_patterns)
        return new_state, ys.reshape(-1, ys.shape[-2], ys.shape[-1]) # (n_steps, bs_size, n_tested_patterns)

    def no_record_phase2(self, rng: jax.Array, hyperD: HyperDynConfig1, state: State1, n_steps: int) -> State1:
        """ Continue the simulation for `n_steps` without recording the outputs.
            `n_steps` should be a multiple of resample_every.
        """
        n_big_steps = n_steps // self.resample_every
        rngs = jax.random.split(rng, n_big_steps)
        def step_fn(carry, rng): # carry = state
            xs = self.xs_generator.generate(rng, self.resample_every) # (resample_every, nd, ns)
            new_carry = self._update_no_record(hyperD, carry, xs)
            return new_carry, None
        new_state, _ = jax.lax.scan(step_fn, state, rngs)
        return new_state
    
    @partial(jax.jit, static_argnames=('self', 'n_steps', 'test_every'))
    def steady_record_phase3(self, rng: jax.Array, hyperD: HyperDynConfig1, state: State1, n_steps: int, test_every: int, bs: jax.Array, x0s: jax.Array) -> Tuple[State1, jax.Array]: # bs: (bs_size,), x0s: (n_tested_patterns, nd, ns)
        """ Continue the simulation for `n_steps`, and for every `test_every` steps, test the state by `n_tested_patterns` patterns and record the outputs.
            `n_steps` should be a multiple of resample_every.
            This phase should be used in the steady state. 
        """
        n_big_steps = n_steps // self.resample_every
        rngs = jax.random.split(rng, n_big_steps)
        def step_fn(carry, rng): # carry = state
            xs = self.xs_generator.generate(rng, self.resample_every) # (resample_every, nd, ns)
            new_carry, ys = self._update_record3_sparse(hyperD, carry, bs, xs, x0s, test_every)
            return new_carry, ys
        new_state, ys = jax.lax.scan(step_fn, state, rngs) # ys: (n_big_steps, resample_every, bs_size, n_tested_patterns)
        return new_state, ys.reshape(-1, ys.shape[-2], ys.shape[-1]) # (n_steps, bs_size, n_tested_patterns)
    
    @partial(jax.jit, static_argnames=('self', ))
    def get_upper99(self, steady_record: jax.Array):
        """ steady_record: (n_steps, bs_size, n_tested_patterns) """
        eps = 1e-5
        n_steps, n_tested_patterns = steady_record.shape[0], steady_record.shape[2]
        tmp = jnp.transpose(steady_record, (1, 0, 2)) # (bs_size, n_steps, n_tested_patterns)
        tmp = tmp.reshape(tmp.shape[0], -1) # (bs_size, n_steps*n_tested_patterns)
        k99 = max(0, int(np.ceil(0.99 * n_steps*n_tested_patterns)) - 1)
        upper99_s = jnp.partition(tmp, k99, axis=1)[:, k99] + eps # (bs_size,)
        return upper99_s

    def run(self, 
            n_init_steps: int,
            n_steps_phase1: int, # must be larger than n_tested_patterns
            n_steps_phase2: int,
            n_steps_phase3: int,
            test_every_phase3: int,
            bs: jax.Array,
            ):
        if self.state is None:
            self.rng, rng1, rng2 = jax.random.split(self.rng, 3)
            self.state = self.neuron.init_state(rng1, self.hyperD)
            self.state = self.init_phase(rng2, self.hyperD, self.state, n_init_steps)

        self.rng, rng1, rng2, rng3, rng4 = jax.random.split(self.rng, 5)
        x0s = self.xs_generator.generate(rng1, self.n_tested_patterns) # (n_tested_patterns, nd, ns)
        self.state, transient_record = self.record_phase1(rng2, self.hyperD, self.state, n_steps_phase1, bs, x0s)
        self.state = self.no_record_phase2(rng3, self.hyperD, self.state, n_steps_phase2)
        self.state, steady_record = self.steady_record_phase3(rng4, self.hyperD, self.state, n_steps_phase3, test_every_phase3, bs, x0s)

        # exclude empty records in steady_record
        mask = jnp.any(jnp.isclose(steady_record, -100), axis=(1, 2)) # (n_steps_phase3,)
        steady_record = steady_record[~mask] # (n_recorded_steps, bs_size, n_tested_patterns)

        self.upper99_s = np.array(self.get_upper99(steady_record)) # (bs_size,)

        self.transient_record = np.array(transient_record) # (n_steps_phase1, bs_size, n_tested_patterns)
        self.steady_record = np.array(steady_record) # (n_recorded_steps, bs_size, n_tested_patterns)

        # realign the transient_record so that the row index corresponds to the delay
        for i in range(self.n_tested_patterns):
            self.transient_record[:, :, i] = np.roll(self.transient_record[:, :, i], -i, axis=0)
        self.transient_record = self.transient_record[:n_steps_phase1-self.n_tested_patterns]

class Driver1_bScan_activationScan(Driver1_bScan):
    def __init__(self, neuron, xs_generator, rng, hyperD = None, initial_state = None, resample_every = 1000, n_tested_patterns = 1000):
        super().__init__(neuron, xs_generator, rng, hyperD, initial_state, resample_every, n_tested_patterns)


    def _update_record1(self, hyperD: HyperDynConfig1, state: State1, bs: jax.Array, window_lens: jax.Array, xs: jax.Array, x0s: jax.Array): # bs: (bs_size,), window_lens: (win_size,), xs: (resample_every, nd, ns), x0s: (n_tested_patterns, nd, ns)
        """ Run simulation for `resample_every` steps and record the outputs"""
        def step_fn(carry, x): # carry = state
            new_carry = self.neuron.update(hyperD, carry, x)
            w = new_carry.w # (nd, ns)
            us = jnp.sum(x0s * w, axis=-1) # (n_tested_patterns, nd)
            z = jax.nn.relu(us[None, :, :] - bs[:, None, None]) # (bs_size, n_tested_patterns, nd)
            z = z[:, None, :, :] - jax.nn.relu(z[:, None, :, :] - window_lens[None, :, None, None]) # (bs_size, win_size, n_tested_patterns, nd)
            y = jnp.sum(z, axis=-1) # (bs_size, win_size, n_tested_patterns)
            y = y.reshape(-1, y.shape[-1]) # (bs_size*win_size, n_tested_patterns)
            return new_carry, y
        new_state, ys = jax.lax.scan(step_fn, state, xs) # ys: (resample_every, bs_size*win_size, n_tested_patterns)
        return new_state, ys

    def _update_record3_sparse(self, hyperD: HyperDynConfig1, state: State1, bs: jax.Array, window_lens: jax.Array, xs: jax.Array, x0s: jax.Array, test_every: int): # bs: (bs_size,), window_lens: (win_size,), x0s: (n_tested_patterns, nd, ns)
        """ Run simulation for `resample_every` steps and record the outputs"""
        n_tested_patterns = x0s.shape[0]
        bs_size = bs.shape[0]
        win_size = window_lens.shape[0]
        def step_fn1(state, x): # carry = state
            new_state = self.neuron.update(hyperD, state, x)
            w = new_state.w # (nd, ns)
            us = jnp.sum(x0s * w, axis=-1) # (n_tested_patterns, nd)
            z = jax.nn.relu(us[None, :, :] - bs[:, None, None]) # (bs_size, n_tested_patterns, nd)
            z = z[:, None, :, :] - jax.nn.relu(z[:, None, :, :] - window_lens[None, :, None, None]) # (bs_size, win_size, n_tested_patterns, nd)
            y = jnp.sum(z, axis=-1) # (bs_size, win_size, n_tested_patterns)
            y = y.reshape(-1, y.shape[-1]) # (bs_size*win_size, n_tested_patterns)
            return new_state, y
        def step_fn2(state, x): # carry = state
            new_state = self.neuron.update(hyperD, state, x)
            y = jnp.full((bs_size*win_size, n_tested_patterns), -100.)
            return new_state, y
        def step_fn(carry, y): # carry = (state, idx)
            state, idx = carry
            new_state, y = jax.lax.cond(idx % test_every == 0, step_fn1, step_fn2, state, y)
            return (new_state, idx+1), y
        (new_state, _), ys = jax.lax.scan(step_fn, (state, 0), xs) # ys: (resample_every, bs_size*win_size, n_tested_patterns)
        return new_state, ys
    
    @partial(jax.jit, static_argnames=('self', 'n_steps'))
    def record_phase1(self, rng: jax.Array, hyperD: HyperDynConfig1, state: State1, n_steps: jax.Array, bs: jax.Array, window_lens: jax.Array, x0s: jax.Array) -> Tuple[State1, jax.Array]: # bs: (bs_size,), window_lens: (win_size,), x0s: (n_tested_patterns, nd, ns)
        """ Run simulation for `n_steps`, and for each step, test the state by `n_tested_patterns` patterns and record the outputs.
        `n_steps` should be a multiple of `resample_every`."""
        x0s_padded_len = (self.resample_every - self.n_tested_patterns) % self.resample_every
        x0s_padded = jnp.pad(x0s, ((0, x0s_padded_len), (0,0), (0,0)), mode='constant', constant_values=0.) # (n_tested_patterns + x0s_padded_len, nd, ns), first dim is a multiple of resample_every

        n_big_steps = n_steps // self.resample_every
        rngs = jax.random.split(rng, n_big_steps)
        def step_fn(carry, rng): # carry = (state, idx)
            state, idx = carry

            # generate xs
            xs = self.xs_generator.generate(rng, self.resample_every) # (resample_every, nd, ns)
            tmp_x0_padded = jax.lax.dynamic_slice(x0s_padded, (idx*self.resample_every, 0, 0), (self.resample_every, x0s.shape[1], x0s.shape[2])) # (resample_every, nd, ns)
            mask_array = ( jnp.arange(self.resample_every)+idx*self.resample_every ) < self.n_tested_patterns
            xs = jnp.where(mask_array[:, None, None], tmp_x0_padded, xs) # (resample_every, nd, ns)

            new_state, ys = self._update_record1(hyperD, state, bs, window_lens, xs, x0s)
            return (new_state, idx+1), ys

        (new_state, _), ys = jax.lax.scan(step_fn, (state, 0), rngs) # ys: (n_big_steps, resample_every, bs_size*win_size, n_tested_patterns)
        return new_state, ys.reshape(-1, ys.shape[-2], ys.shape[-1]) # (n_steps, bs_size*win_size, n_tested_patterns)
    
    @partial(jax.jit, static_argnames=('self', 'n_steps', 'test_every'))
    def steady_record_phase3(self, rng: jax.Array, hyperD: HyperDynConfig1, state: State1, n_steps: int, test_every: int, bs: jax.Array, window_lens: jax.Array, x0s: jax.Array) -> Tuple[State1, jax.Array]: # bs: (bs_size,), window_lens: (win_size,), x0s: (n_tested_patterns, nd, ns)
        """ Continue the simulation for `n_steps`, and for every `test_every` steps, test the state by `n_tested_patterns` patterns and record the outputs.
            `n_steps` should be a multiple of resample_every.
            This phase should be used in the steady state. 
        """
        n_big_steps = n_steps // self.resample_every
        rngs = jax.random.split(rng, n_big_steps)
        def step_fn(carry, rng): # carry = state
            xs = self.xs_generator.generate(rng, self.resample_every) # (resample_every, nd, ns)
            new_carry, ys = self._update_record3_sparse(hyperD, carry, bs, window_lens, xs, x0s, test_every)
            return new_carry, ys
        new_state, ys = jax.lax.scan(step_fn, state, rngs) # ys: (n_big_steps, resample_every, bs_size*win_size, n_tested_patterns)
        return new_state, ys.reshape(-1, ys.shape[-2], ys.shape[-1]) # (n_steps, bs_size*win_size, n_tested_patterns)
    
    def run(self, 
            n_init_steps: int,
            n_steps_phase1: int, # must be larger than n_tested_patterns
            n_steps_phase2: int,
            n_steps_phase3: int,
            test_every_phase3: int,
            bs: jax.Array,
            window_lens: jax.Array,
            ):
        if self.state is None:
            self.rng, rng1, rng2 = jax.random.split(self.rng, 3)
            self.state = self.neuron.init_state(rng1, self.hyperD)
            self.state = self.init_phase(rng2, self.hyperD, self.state, n_init_steps)

        self.rng, rng1, rng2, rng3, rng4 = jax.random.split(self.rng, 5)
        x0s = self.xs_generator.generate(rng1, self.n_tested_patterns) # (n_tested_patterns, nd, ns)
        self.state, transient_record = self.record_phase1(rng2, self.hyperD, self.state, n_steps_phase1, bs, window_lens, x0s)
        self.state = self.no_record_phase2(rng3, self.hyperD, self.state, n_steps_phase2)
        self.state, steady_record = self.steady_record_phase3(rng4, self.hyperD, self.state, n_steps_phase3, test_every_phase3, bs, window_lens, x0s)

        self.transient_record = np.array(transient_record) # (n_steps_phase1, bs_size*win_size, n_tested_patterns)
        self.steady_record = np.array(steady_record) # (n_steps_phase3, bs_size*win_size, n_tested_patterns)

        # realign the transient_record so that the row index corresponds to the delay
        for i in range(self.n_tested_patterns):
            self.transient_record[:, :, i] = np.roll(self.transient_record[:, :, i], -i, axis=0)
        self.transient_record = self.transient_record[:n_steps_phase1-self.n_tested_patterns]

        # exclude empty records in steady_record
        mask = np.any(np.isclose(self.steady_record, -100), axis=(1, 2)) # (n_steps_phase3,)
        self.steady_record = self.steady_record[~mask] # (n_recorded_steps, bs_size, n_tested_patterns)

class Driver1_noNonlinearity(Driver1_bScan):
    def __init__(self, neuron, xs_generator, rng, hyperD = None, initial_state = None, resample_every = 1000, n_tested_patterns = 1000):
        super().__init__(neuron, xs_generator, rng, hyperD, initial_state, resample_every, n_tested_patterns)

    def _update_record1(self, hyperD: HyperDynConfig1, state: State1, bs: jax.Array, xs: jax.Array, x0s: jax.Array): # bs: (bs_size,), xs: (resample_every, nd, ns), x0s: (n_tested_patterns, nd, ns)
        """ Run simulation for `resample_every` steps and record the outputs"""
        def step_fn(carry, x): # carry = state
            new_carry = self.neuron.update(hyperD, carry, x)
            w = new_carry.w # (nd, ns)
            us = jnp.sum(x0s * w, axis=-1) # (n_tested_patterns, nd)
            y = jnp.sum(us, axis=-1) # (n_tested_patterns, )
            y = y.reshape(1, -1) # (1, n_tested_patterns)
            return new_carry, y
        new_state, ys = jax.lax.scan(step_fn, state, xs) # ys: (resample_every, bs_size, n_tested_patterns)
        return new_state, ys

    def _update_record3_sparse(self, hyperD: HyperDynConfig1, state: State1, bs: jax.Array, xs: jax.Array, x0s: jax.Array, test_every: int): # bs: (bs_size,), x0s: (n_tested_patterns, nd, ns)
        """ Run simulation for `resample_every` steps and record the outputs"""
        n_tested_patterns = x0s.shape[0]
        bs_size = bs.shape[0]
        def step_fn1(state, x): # carry = state
            new_state = self.neuron.update(hyperD, state, x)
            w = new_state.w # (nd, ns)
            us = jnp.sum(x0s * w, axis=-1) # (n_tested_patterns, nd)
            y = jnp.sum(us, axis=-1) # (n_tested_patterns, )
            y = y.reshape(1, -1) # (1, n_tested_patterns)
            return new_state, y
        def step_fn2(state, x): # carry = state
            new_state = self.neuron.update(hyperD, state, x)
            y = jnp.full((1, n_tested_patterns), -100.)
            return new_state, y
        def step_fn(carry, y): # carry = (state, idx)
            state, idx = carry
            new_state, y = jax.lax.cond(idx % test_every == 0, step_fn1, step_fn2, state, y)
            return (new_state, idx+1), y
        (new_state, _), ys = jax.lax.scan(step_fn, (state, 0), xs) # ys: (resample_every, bs_size, n_tested_patterns)
        return new_state, ys

class Driver1_NoisyRecall_SynapseNoise(Driver1_bScan):
    def __init__(self, neuron, xs_generator, rng, noise_level=0.0, hyperD = None, initial_state = None, resample_every = 1000, n_tested_patterns = 1000):
        super().__init__(neuron, xs_generator, rng, hyperD, initial_state, resample_every, n_tested_patterns)
        self.noise_level = noise_level

    def _update_record1(self, hyperD: HyperDynConfig1, state: State1, bs: jax.Array, xs: jax.Array, x0s: jax.Array): # bs: (bs_size,), xs: (resample_every, nd, ns), x0s: (n_tested_patterns, nd, ns)
        """ Run simulation for `resample_every` steps and record the outputs"""
        nd = xs.shape[1]
        rng = state.rng
        rng_noise, new_rng = jax.random.split(rng)
        noise_arr = jax.random.normal(rng_noise, (self.resample_every, self.n_tested_patterns, nd)) * self.noise_level # (resample_every, n_tested_patterns, nd)
        state = dataclasses.replace(state, rng=new_rng)

        def step_fn(carry, inp): # carry = state, inp = (x, noise)
            x, noise = inp # x: (nd, ns), noise: (n_tested_patterns, nd)
            new_carry = self.neuron.update(hyperD, carry, x)
            w = new_carry.w # (nd, ns)
            w_len = jnp.linalg.norm(w, axis=-1) # (nd,)
            us = jnp.sum(x0s * w, axis=-1) # (n_tested_patterns, nd)
            us = us + noise * w_len[None, :] # (n_tested_patterns, nd)
            z = jax.nn.relu(us[None, :, :] - bs[:, None, None]) # (bs_size, n_tested_patterns, nd)
            y = jnp.sum(z, axis=-1) # (bs_size, n_tested_patterns)
            return new_carry, y
        new_state, ys = jax.lax.scan(step_fn, state, (xs, noise_arr)) # ys: (resample_every, bs_size, n_tested_patterns)
        return new_state, ys
    
    def _update_record3_sparse(self, hyperD: HyperDynConfig1, state: State1, bs: jax.Array, xs: jax.Array, x0s: jax.Array, test_every: int): # bs: (bs_size,), x0s: (n_tested_patterns, nd, ns)
        """ Run simulation for `resample_every` steps and record the outputs"""
        n_tested_patterns = x0s.shape[0]
        bs_size = bs.shape[0]

        nd = xs.shape[1]
        rng = state.rng
        rng_noise, new_rng = jax.random.split(rng)
        noise_arr = jax.random.normal(rng_noise, (self.resample_every, self.n_tested_patterns, nd)) * self.noise_level # (resample_every, n_tested_patterns, nd)
        state = dataclasses.replace(state, rng=new_rng)

        def step_fn1(state, inp): # carry = state, inp = (x, noise)
            x, noise = inp # x: (nd, ns), noise: (n_tested_patterns, nd)
            new_state = self.neuron.update(hyperD, state, x)
            w = new_state.w # (nd, ns)
            w_len = jnp.linalg.norm(w, axis=-1) # (nd,)
            us = jnp.sum(x0s * w, axis=-1) # (n_tested_patterns, nd)
            us = us + noise * w_len[None, :] # (n_tested_patterns, nd)
            z = jax.nn.relu(us[None, :, :] - bs[:, None, None]) # (bs_size, n_tested_patterns, nd)
            y = jnp.sum(z, axis=-1) # (bs_size, n_tested_patterns)
            return new_state, y
        def step_fn2(state, inp): # carry = state, inp = (x, noise)
            x, noise = inp # x: (nd, ns), noise: (n_tested_patterns, nd)
            new_state = self.neuron.update(hyperD, state, x)
            y = jnp.full((bs_size, n_tested_patterns), -100.)
            return new_state, y
        def step_fn(carry, y): # carry = (state, idx)
            state, idx = carry
            new_state, y = jax.lax.cond(idx % test_every == 0, step_fn1, step_fn2, state, y)
            return (new_state, idx+1), y
        (new_state, _), ys = jax.lax.scan(step_fn, (state, 0), (xs, noise_arr)) # ys: (resample_every, bs_size, n_tested_patterns)
        return new_state, ys

class Driver1_NoisyRecall_DendriteNoise(Driver1_bScan):
    """noise created during dendrite transmission. noise is scaled by kappa, the noise is only added when the activation level is above b."""
    def __init__(self, neuron, xs_generator, rng, noise_level=0.0, hyperD = None, initial_state = None, resample_every = 1000, n_tested_patterns = 1000):
        super().__init__(neuron, xs_generator, rng, hyperD, initial_state, resample_every, n_tested_patterns)
        self.noise_level = noise_level

    def _update_record1(self, hyperD: HyperDynConfig1, state: State1, bs: jax.Array, xs: jax.Array, x0s: jax.Array): # bs: (bs_size,), xs: (resample_every, nd, ns), x0s: (n_tested_patterns, nd, ns)
        """ Run simulation for `resample_every` steps and record the outputs"""
        nd = xs.shape[1]
        rng = state.rng
        rng_noise, new_rng = jax.random.split(rng)
        noise_arr = jax.random.normal(rng_noise, (self.resample_every, self.n_tested_patterns, nd)) * self.noise_level # (resample_every, n_tested_patterns, nd)
        state = dataclasses.replace(state, rng=new_rng)
        kappa_s = hyperD.bk - bs # (bs_size,)

        def step_fn(carry, inp): # carry = state, inp = (x, noise)
            x, noise = inp # x: (nd, ns), noise: (n_tested_patterns, nd)
            new_carry = self.neuron.update(hyperD, carry, x)
            w = new_carry.w # (nd, ns)
            w_len = jnp.linalg.norm(w, axis=-1) # (nd,)
            us = jnp.sum(x0s * w, axis=-1) # (n_tested_patterns, nd)
            z = jax.nn.relu(us[None, :, :] - bs[:, None, None]) # (bs_size, n_tested_patterns, nd)
            mask = jnp.heaviside(z, 0) # (bs_size, n_tested_patterns, nd)
            z += mask*noise[None, :, :]*kappa_s[:, None, None]*w_len[None, None, :] # (bs_size, n_tested_patterns, nd)
            y = jnp.sum(z, axis=-1) # (bs_size, n_tested_patterns)
            return new_carry, y
        new_state, ys = jax.lax.scan(step_fn, state, (xs, noise_arr)) # ys: (resample_every, bs_size, n_tested_patterns)
        return new_state, ys
    
    def _update_record3_sparse(self, hyperD: HyperDynConfig1, state: State1, bs: jax.Array, xs: jax.Array, x0s: jax.Array, test_every: int): # bs: (bs_size,), x0s: (n_tested_patterns, nd, ns)
        """ Run simulation for `resample_every` steps and record the outputs"""
        n_tested_patterns = x0s.shape[0]
        bs_size = bs.shape[0]

        nd = xs.shape[1]
        rng = state.rng
        rng_noise, new_rng = jax.random.split(rng)
        noise_arr = jax.random.normal(rng_noise, (self.resample_every, self.n_tested_patterns, nd)) * self.noise_level # (resample_every, n_tested_patterns, nd)
        state = dataclasses.replace(state, rng=new_rng)
        kappa_s = hyperD.bk - bs # (bs_size,)

        def step_fn1(state, inp): # carry = state, inp = (x, noise)
            x, noise = inp # x: (nd, ns), noise: (n_tested_patterns, nd)
            new_state = self.neuron.update(hyperD, state, x)
            w = new_state.w # (nd, ns)
            w_len = jnp.linalg.norm(w, axis=-1) # (nd,)
            us = jnp.sum(x0s * w, axis=-1) # (n_tested_patterns, nd)
            z = jax.nn.relu(us[None, :, :] - bs[:, None, None]) # (bs_size, n_tested_patterns, nd)
            mask = jnp.heaviside(z, 0) # (bs_size, n_tested_patterns, nd)
            z += mask*noise[None, :, :]*kappa_s[:, None, None]*w_len[None, None, :] # (bs_size, n_tested_patterns, nd
            y = jnp.sum(z, axis=-1) # (bs_size, n_tested_patterns)
            return new_state, y
        def step_fn2(state, inp): # carry = state, inp = (x, noise)
            x, noise = inp # x: (nd, ns), noise: (n_tested_patterns, nd)
            new_state = self.neuron.update(hyperD, state, x)
            y = jnp.full((bs_size, n_tested_patterns), -100.)
            return new_state, y
        def step_fn(carry, y): # carry = (state, idx)
            state, idx = carry
            new_state, y = jax.lax.cond(idx % test_every == 0, step_fn1, step_fn2, state, y)
            return (new_state, idx+1), y
        (new_state, _), ys = jax.lax.scan(step_fn, (state, 0), (xs, noise_arr)) # ys: (resample_every, bs_size, n_tested_patterns)
        return new_state, ys

class Driver1_NoisyRecall_SomaNoise(Driver1_bScan):
    """noise added to the soma is proportional to the norm of the synaptic weight vector, about sqrt(nd)"""
    def __init__(self, neuron, xs_generator, rng, noise_level=0.0, hyperD = None, initial_state = None, resample_every = 1000, n_tested_patterns = 1000):
        super().__init__(neuron, xs_generator, rng, hyperD, initial_state, resample_every, n_tested_patterns)
        self.noise_level = noise_level

    def _update_record1(self, hyperD: HyperDynConfig1, state: State1, bs: jax.Array, xs: jax.Array, x0s: jax.Array): # bs: (bs_size,), xs: (resample_every, nd, ns), x0s: (n_tested_patterns, nd, ns)
        """ Run simulation for `resample_every` steps and record the outputs"""
        nd = xs.shape[1]
        rng = state.rng
        rng_noise, new_rng = jax.random.split(rng)
        # noise_arr = jax.random.normal(rng_noise, (self.resample_every, self.n_tested_patterns)) * self.noise_level # (resample_every, n_tested_patterns)
        noise_arr = jax.random.normal(rng_noise, (self.resample_every, self.n_tested_patterns)) * self.noise_level / jnp.sqrt(nd) # (resample_every, n_tested_patterns)
        state = dataclasses.replace(state, rng=new_rng)

        def step_fn(carry, inp): # carry = state, inp = (x, noise)
            x, noise = inp # x: (nd, ns), noise: (n_tested_patterns, )
            new_carry = self.neuron.update(hyperD, carry, x)
            w = new_carry.w # (nd, ns)
            w_len = jnp.linalg.norm(w) # scalar
            us = jnp.sum(x0s * w, axis=-1) # (n_tested_patterns, nd)
            z = jax.nn.relu(us[None, :, :] - bs[:, None, None]) # (bs_size, n_tested_patterns, nd)
            y = jnp.sum(z, axis=-1) # (bs_size, n_tested_patterns)
            y = y + noise[None, :] * w_len # (bs_size, n_tested_patterns)
            return new_carry, y
        new_state, ys = jax.lax.scan(step_fn, state, (xs, noise_arr)) # ys: (resample_every, bs_size, n_tested_patterns)
        return new_state, ys
    
    def _update_record3_sparse(self, hyperD: HyperDynConfig1, state: State1, bs: jax.Array, xs: jax.Array, x0s: jax.Array, test_every: int): # bs: (bs_size,), x0s: (n_tested_patterns, nd, ns)
        """ Run simulation for `resample_every` steps and record the outputs"""
        n_tested_patterns = x0s.shape[0]
        bs_size = bs.shape[0]

        nd = xs.shape[1]
        rng = state.rng
        rng_noise, new_rng = jax.random.split(rng)
        # noise_arr = jax.random.normal(rng_noise, (self.resample_every, self.n_tested_patterns)) * self.noise_level # (resample_every, n_tested_patterns)
        noise_arr = jax.random.normal(rng_noise, (self.resample_every, self.n_tested_patterns)) * self.noise_level / jnp.sqrt(nd) # (resample_every, n_tested_patterns)
        state = dataclasses.replace(state, rng=new_rng)

        def step_fn1(state, inp): # carry = state, inp = (x, noise)
            x, noise = inp # x: (nd, ns), noise: (n_tested_patterns, )
            new_state = self.neuron.update(hyperD, state, x)
            w = new_state.w # (nd, ns)
            w_len = jnp.linalg.norm(w) # scalar
            us = jnp.sum(x0s * w, axis=-1) # (n_tested_patterns, nd)
            z = jax.nn.relu(us[None, :, :] - bs[:, None, None]) # (bs_size, n_tested_patterns, nd)
            y = jnp.sum(z, axis=-1) # (bs_size, n_tested_patterns)
            y = y + noise[None, :] * w_len # (bs_size, n_tested_patterns)
            return new_state, y
        def step_fn2(state, inp): # carry = state, inp = (x, noise)
            x, noise = inp # x: (nd, ns), noise: (n_tested_patterns, )
            new_state = self.neuron.update(hyperD, state, x)
            y = jnp.full((bs_size, n_tested_patterns), -100.)
            return new_state, y
        def step_fn(carry, y): # carry = (state, idx)
            state, idx = carry
            new_state, y = jax.lax.cond(idx % test_every == 0, step_fn1, step_fn2, state, y)
            return (new_state, idx+1), y
        (new_state, _), ys = jax.lax.scan(step_fn, (state, 0), (xs, noise_arr)) # ys: (resample_every, bs_size, n_tested_patterns)
        return new_state, ys

class Driver1_NoisyRecall_SomaNoise2(Driver1_bScan):
    """noise added to the soma is proportional to n*kappa"""
    def __init__(self, neuron, xs_generator, rng, noise_level=0.0, hyperD = None, initial_state = None, resample_every = 1000, n_tested_patterns = 1000):
        super().__init__(neuron, xs_generator, rng, hyperD, initial_state, resample_every, n_tested_patterns)
        self.noise_level = noise_level

    def _update_record1(self, hyperD: HyperDynConfig1, state: State1, bs: jax.Array, xs: jax.Array, x0s: jax.Array): # bs: (bs_size,), xs: (resample_every, nd, ns), x0s: (n_tested_patterns, nd, ns)
        """ Run simulation for `resample_every` steps and record the outputs"""
        nd = xs.shape[1]
        rng = state.rng
        rng_noise, new_rng = jax.random.split(rng)
        noise_arr = jax.random.normal(rng_noise, (self.resample_every, self.n_tested_patterns)) * self.noise_level # (resample_every, n_tested_patterns)
        state = dataclasses.replace(state, rng=new_rng)

        bk = hyperD.bk
        kappa_s = bk - bs # (bs_size,)
        n = self.neuron.hyper.n

        def step_fn(carry, inp): # carry = state, inp = (x, noise)
            x, noise = inp # x: (nd, ns), noise: (n_tested_patterns, )
            new_carry = self.neuron.update(hyperD, carry, x)
            w = new_carry.w # (nd, ns)
            us = jnp.sum(x0s * w, axis=-1) # (n_tested_patterns, nd)
            z = jax.nn.relu(us[None, :, :] - bs[:, None, None]) # (bs_size, n_tested_patterns, nd)
            y = jnp.sum(z, axis=-1) # (bs_size, n_tested_patterns)
            y = y + noise[None, :] * n * kappa_s[:, None] # (bs_size, n_tested_patterns)
            return new_carry, y
        new_state, ys = jax.lax.scan(step_fn, state, (xs, noise_arr)) # ys: (resample_every, bs_size, n_tested_patterns)
        return new_state, ys
    
    def _update_record3_sparse(self, hyperD: HyperDynConfig1, state: State1, bs: jax.Array, xs: jax.Array, x0s: jax.Array, test_every: int): # bs: (bs_size,), x0s: (n_tested_patterns, nd, ns)
        """ Run simulation for `resample_every` steps and record the outputs"""
        n_tested_patterns = x0s.shape[0]
        bs_size = bs.shape[0]

        nd = xs.shape[1]
        rng = state.rng
        rng_noise, new_rng = jax.random.split(rng)
        noise_arr = jax.random.normal(rng_noise, (self.resample_every, self.n_tested_patterns)) * self.noise_level # (resample_every, n_tested_patterns)
        state = dataclasses.replace(state, rng=new_rng)

        bk = hyperD.bk
        kappa_s = bk - bs # (bs_size,)
        n = self.neuron.hyper.n

        def step_fn1(state, inp): # carry = state, inp = (x, noise)
            x, noise = inp # x: (nd, ns), noise: (n_tested_patterns, )
            new_state = self.neuron.update(hyperD, state, x)
            w = new_state.w # (nd, ns)
            us = jnp.sum(x0s * w, axis=-1) # (n_tested_patterns, nd)
            z = jax.nn.relu(us[None, :, :] - bs[:, None, None]) # (bs_size, n_tested_patterns, nd)
            y = jnp.sum(z, axis=-1) # (bs_size, n_tested_patterns)
            y = y + noise[None, :] * n * kappa_s[:, None] # (bs_size, n_tested_patterns)
            return new_state, y
        def step_fn2(state, inp): # carry = state, inp = (x, noise)
            x, noise = inp # x: (nd, ns), noise: (n_tested_patterns, )
            new_state = self.neuron.update(hyperD, state, x)
            y = jnp.full((bs_size, n_tested_patterns), -100.)
            return new_state, y
        def step_fn(carry, y): # carry = (state, idx)
            state, idx = carry
            new_state, y = jax.lax.cond(idx % test_every == 0, step_fn1, step_fn2, state, y)
            return (new_state, idx+1), y
        (new_state, _), ys = jax.lax.scan(step_fn, (state, 0), (xs, noise_arr)) # ys: (resample_every, bs_size, n_tested_patterns)
        return new_state, ys

class Driver1_NoisyRecall_SomaNoise3(Driver1_bScan):
    """noise added to the soma is scaled by theta_Y, the maximum output is clipped by max_Y*theta_Y"""
    def __init__(self, neuron, xs_generator, rng, noise_level=0.0, max_Y=3.0, hyperD = None, initial_state = None, resample_every = 1000, n_tested_patterns = 1000):
        super().__init__(neuron, xs_generator, rng, hyperD, initial_state, resample_every, n_tested_patterns)
        self.noise_level = noise_level
        self.max_Y = max_Y
    
    @partial(jax.jit, static_argnames=('self', ))
    def add_noise_and_clip(self, ys: jax.Array, upper99_s: jax.Array, rng: jax.Array):
        """ ys: (n_steps, bs_size, n_tested_patterns), upper99_s: (bs_size,) """
        noise = jax.random.normal(rng, ys.shape) * self.noise_level * upper99_s[None, :, None] # (n_steps, bs_size, n_tested_patterns)
        ys = jnp.clip(ys, a_min=0.0, a_max=self.max_Y * upper99_s[None, :, None]) # (n_steps, bs_size, n_tested_patterns)
        ys = ys + noise
        return ys

    def run(self, 
            n_init_steps: int,
            n_steps_phase1: int, # must be larger than n_tested_patterns
            n_steps_phase2: int,
            n_steps_phase3: int,
            test_every_phase3: int,
            bs: jax.Array,
            ):
        if self.state is None:
            self.rng, rng1, rng2 = jax.random.split(self.rng, 3)
            self.state = self.neuron.init_state(rng1, self.hyperD)
            self.state = self.init_phase(rng2, self.hyperD, self.state, n_init_steps)

        self.rng, rng1, rng2, rng3, rng4 = jax.random.split(self.rng, 5)
        x0s = self.xs_generator.generate(rng1, self.n_tested_patterns) # (n_tested_patterns, nd, ns)
        self.state, transient_record = self.record_phase1(rng2, self.hyperD, self.state, n_steps_phase1, bs, x0s)
        self.state = self.no_record_phase2(rng3, self.hyperD, self.state, n_steps_phase2)
        self.state, steady_record = self.steady_record_phase3(rng4, self.hyperD, self.state, n_steps_phase3, test_every_phase3, bs, x0s)

        # exclude empty records in steady_record
        mask = jnp.any(jnp.isclose(steady_record, -100), axis=(1, 2)) # (n_steps_phase3,)
        steady_record = steady_record[~mask] # (n_recorded_steps, bs_size, n_tested_patterns)

        self.rng, noise_rng1, noise_rng2 = jax.random.split(self.rng, 3)
        upper99_s = self.get_upper99(steady_record) # (bs_size,)
        transient_record = self.add_noise_and_clip(transient_record, upper99_s, noise_rng1)
        steady_record = self.add_noise_and_clip(steady_record, upper99_s, noise_rng2)
        self.upper99_s = self.get_upper99(steady_record) # (bs_size,)

        self.transient_record = np.array(transient_record) # (n_steps_phase1, bs_size, n_tested_patterns)
        self.steady_record = np.array(steady_record) # (n_recorded_steps, bs_size, n_tested_patterns)
        self.upper99_s = np.array(self.upper99_s) # (bs_size,)

        # realign the transient_record so that the row index corresponds to the delay
        for i in range(self.n_tested_patterns):
            self.transient_record[:, :, i] = np.roll(self.transient_record[:, :, i], -i, axis=0)
        self.transient_record = self.transient_record[:n_steps_phase1-self.n_tested_patterns]

class Driver1_NoisyRecall_DendriteSomaNoise(Driver1_NoisyRecall_DendriteNoise):
    def __init__(self, neuron, xs_generator, rng, dendrite_noise_level=0, soma_noise_level=0, hyperD=None, initial_state=None, resample_every=1000, n_tested_patterns=1000):
        super().__init__(neuron, xs_generator, rng, dendrite_noise_level, hyperD, initial_state, resample_every, n_tested_patterns)
        self.soma_noise_level = soma_noise_level

    @partial(jax.jit, static_argnames=('self', ))
    def add_soma_noise(self, ys: jax.Array, upper99_s: jax.Array, rng: jax.Array):
        """ ys: (n_steps, bs_size, n_tested_patterns), upper99_s: (bs_size,) """
        noise = jax.random.normal(rng, ys.shape) * self.soma_noise_level * upper99_s[None, :, None] # (n_steps, bs_size, n_tested_patterns)
        ys = ys + noise
        return ys
    
    def run(self, 
            n_init_steps: int,
            n_steps_phase1: int, # must be larger than n_tested_patterns
            n_steps_phase2: int,
            n_steps_phase3: int,
            test_every_phase3: int,
            bs: jax.Array,
            ):
        if self.state is None:
            self.rng, rng1, rng2 = jax.random.split(self.rng, 3)
            self.state = self.neuron.init_state(rng1, self.hyperD)
            self.state = self.init_phase(rng2, self.hyperD, self.state, n_init_steps)

        self.rng, rng1, rng2, rng3, rng4 = jax.random.split(self.rng, 5)
        x0s = self.xs_generator.generate(rng1, self.n_tested_patterns) # (n_tested_patterns, nd, ns)
        self.state, transient_record = self.record_phase1(rng2, self.hyperD, self.state, n_steps_phase1, bs, x0s)
        self.state = self.no_record_phase2(rng3, self.hyperD, self.state, n_steps_phase2)
        self.state, steady_record = self.steady_record_phase3(rng4, self.hyperD, self.state, n_steps_phase3, test_every_phase3, bs, x0s)

        # exclude empty records in steady_record
        mask = jnp.any(jnp.isclose(steady_record, -100), axis=(1, 2)) # (n_steps_phase3,)
        steady_record = steady_record[~mask] # (n_recorded_steps, bs_size, n_tested_patterns)

        self.rng, noise_rng1, noise_rng2 = jax.random.split(self.rng, 3)
        upper99_s = self.get_upper99(steady_record) # (bs_size,)
        transient_record = self.add_soma_noise(transient_record, upper99_s, noise_rng1)
        steady_record = self.add_soma_noise(steady_record, upper99_s, noise_rng2)
        self.upper99_s = self.get_upper99(steady_record) # (bs_size,)

        self.transient_record = np.array(transient_record) # (n_steps_phase1, bs_size, n_tested_patterns)
        self.steady_record = np.array(steady_record) # (n_recorded_steps, bs_size, n_tested_patterns)
        self.upper99_s = np.array(self.upper99_s) # (bs_size,)

        # realign the transient_record so that the row index corresponds to the delay
        for i in range(self.n_tested_patterns):
            self.transient_record[:, :, i] = np.roll(self.transient_record[:, :, i], -i, axis=0)
        self.transient_record = self.transient_record[:n_steps_phase1-self.n_tested_patterns]

class Driver1_NoisyRecall_SomaNoise3Scan(Driver1_bScan):
    """ Scan different noise levels during recall """
    def __init__(self, neuron, xs_generator, rng, hyperD = None, initial_state = None, resample_every = 1000, n_tested_patterns = 1000):
        super().__init__(neuron, xs_generator, rng, hyperD, initial_state, resample_every, n_tested_patterns)

    @partial(jax.jit, static_argnames=('self', ))
    def get_upper99(self, steady_record: jax.Array):
        """ steady_record: (n_steps, bs_size, n_tested_patterns) """
        eps = 1e-5
        n_steps, n_tested_patterns = steady_record.shape[0], steady_record.shape[2]
        tmp = jnp.transpose(steady_record, (1, 0, 2)) # (bs_size, n_steps, n_tested_patterns)
        tmp = tmp.reshape(tmp.shape[0], -1) # (bs_size, n_steps*n_tested_patterns)
        k99 = max(0, int(np.ceil(0.99 * n_steps*n_tested_patterns)) - 1)
        upper99_s = jnp.partition(tmp, k99, axis=1)[:, k99] + eps # (bs_size,)
        return upper99_s

    @partial(jax.jit, static_argnames=('self', ))
    def add_noises(self, ys: jax.Array, noise_levels: jax.Array, upper99_s: jax.Array, rng: jax.Array):
        """ ys: (n_steps, 1, n_tested_patterns), upper99_s: (1, ), noise_levels: (n_levels,) """
        noise = jax.random.normal(rng, ys.shape)[:, :, None, :] * noise_levels[None, None, :, None] * upper99_s[None, :, None, None] # (n_steps, 1, n_levels, n_tested_patterns)
        noise = noise[:, 0, :, :] # (n_steps, n_levels, n_tested_patterns)
        ys = ys[:, 0][:, None, :] + noise # (n_steps, n_levels, n_tested_patterns) 
        return ys
    
    def run(self, 
            n_init_steps: int,
            n_steps_phase1: int, # must be larger than n_tested_patterns
            n_steps_phase2: int,
            n_steps_phase3: int,
            test_every_phase3: int,
            noise_levels: jax.Array, # (n_levels,)
            ):
        if self.state is None:
            self.rng, rng1, rng2 = jax.random.split(self.rng, 3)
            self.state = self.neuron.init_state(rng1, self.hyperD)
            self.state = self.init_phase(rng2, self.hyperD, self.state, n_init_steps)

        b = self.neuron.hyper.b
        bs = jnp.array([b]) # (1,)

        self.rng, rng1, rng2, rng3, rng4 = jax.random.split(self.rng, 5)
        x0s = self.xs_generator.generate(rng1, self.n_tested_patterns) # (n_tested_patterns, nd, ns)
        self.state, transient_record = self.record_phase1(rng2, self.hyperD, self.state, n_steps_phase1, bs, x0s)
        self.state = self.no_record_phase2(rng3, self.hyperD, self.state, n_steps_phase2)
        self.state, steady_record = self.steady_record_phase3(rng4, self.hyperD, self.state, n_steps_phase3, test_every_phase3, bs, x0s)

        # exclude empty records in steady_record
        mask = jnp.any(jnp.isclose(steady_record, -100), axis=(1, 2)) # (n_steps_phase3,)
        steady_record = steady_record[~mask] # (n_recorded_steps, 1, n_tested_patterns)

        self.rng, noise_rng1, noise_rng2 = jax.random.split(self.rng, 3)
        upper99_s = self.get_upper99(steady_record) # (1,)
        transient_record = self.add_noises(transient_record, noise_levels, upper99_s, noise_rng1) # (n_steps, n_levels, n_tested_patterns)
        steady_record = self.add_noises(steady_record, noise_levels, upper99_s, noise_rng2) # (n_steps, n_levels, n_tested_patterns)
        self.upper99_s = self.get_upper99(steady_record) # (n_levels,)

        self.transient_record = np.array(transient_record) # (n_steps_phase1, n_levels, n_tested_patterns)
        self.steady_record = np.array(steady_record) # (n_recorded_steps, n_levels, n_tested_patterns)
        self.upper99_s = np.array(self.upper99_s) # (n_levels,)

        # realign the transient_record so that the row index corresponds to the delay
        for i in range(self.n_tested_patterns):
            self.transient_record[:, :, i] = np.roll(self.transient_record[:, :, i], -i, axis=0)
        self.transient_record = self.transient_record[:n_steps_phase1-self.n_tested_patterns]

class Driver1_NoisyRecall_DendriteNoiseScan(Driver1_bScan):
    def __init__(self, neuron, xs_generator, rng, hyperD = None, initial_state = None, resample_every = 1000, n_tested_patterns = 1000):
        super().__init__(neuron, xs_generator, rng, hyperD, initial_state, resample_every, n_tested_patterns)

    def _update_record1(self, hyperD: HyperDynConfig1, state: State1, noise_levels: jax.Array, xs: jax.Array, x0s: jax.Array): # bs: (bs_size,), xs: (resample_every, nd, ns), x0s: (n_tested_patterns, nd, ns)
        """ Run simulation for `resample_every` steps and record the outputs"""
        nd = xs.shape[1]
        rng = state.rng
        rng_noise, new_rng = jax.random.split(rng)
        noise_arr = jax.random.normal(rng_noise, (self.resample_every, self.n_tested_patterns, nd)) # (resample_every, n_tested_patterns, nd)
        state = dataclasses.replace(state, rng=new_rng)
        bk, b = hyperD.bk, hyperD.b
        kappa = bk - b

        def step_fn(carry, inp): # carry = state, inp = (x, noise)
            x, noise = inp # x: (nd, ns), noise: (n_tested_patterns, nd)
            new_carry = self.neuron.update(hyperD, carry, x)
            w = new_carry.w # (nd, ns)
            w_len = jnp.linalg.norm(w, axis=-1) # (nd,)
            us = jnp.sum(x0s * w, axis=-1) # (n_tested_patterns, nd)
            z = jax.nn.relu(us - b) # (n_tested_patterns, nd)
            mask = jnp.heaviside(z, 0)
            z = z[None, :, :] + (mask*noise*kappa*w_len[None, :])[None, :, :] * noise_levels[:, None, None] # (n_noise_levels, n_tested_patterns, nd)
            y = jnp.sum(z, axis=-1) # (n_noise_levels, n_tested_patterns)
            return new_carry, y
        new_state, ys = jax.lax.scan(step_fn, state, (xs, noise_arr)) # ys: (resample_every, n_noise_levels, n_tested_patterns)
        return new_state, ys
    
    def _update_record3_sparse(self, hyperD: HyperDynConfig1, state: State1, noise_levels: jax.Array, xs: jax.Array, x0s: jax.Array, test_every: int): # bs: (bs_size,), x0s: (n_tested_patterns, nd, ns)
        """ Run simulation for `resample_every` steps and record the outputs"""
        n_tested_patterns = x0s.shape[0]
        n_noise_levels = noise_levels.shape[0]

        nd = xs.shape[1]
        rng = state.rng
        rng_noise, new_rng = jax.random.split(rng)
        noise_arr = jax.random.normal(rng_noise, (self.resample_every, self.n_tested_patterns, nd)) # (resample_every, n_tested_patterns, nd)
        state = dataclasses.replace(state, rng=new_rng)
        bk, b = hyperD.bk, hyperD.b
        kappa = bk - b

        def step_fn1(state, inp): # carry = state, inp = (x, noise)
            x, noise = inp # x: (nd, ns), noise: (n_tested_patterns, nd)
            new_state = self.neuron.update(hyperD, state, x)
            w = new_state.w # (nd, ns)
            w_len = jnp.linalg.norm(w, axis=-1) # (nd,)
            us = jnp.sum(x0s * w, axis=-1) # (n_tested_patterns, nd)
            z = jax.nn.relu(us-b) # (n_tested_patterns, nd)
            mask = jnp.heaviside(z, 0) # (n_tested_patterns, nd)
            z = z[None, :, :] + (mask*noise*kappa*w_len[None, :])[None, :, :] * noise_levels[:, None, None] # (n_noise_levels, n_tested_patterns, nd)
            y = jnp.sum(z, axis=-1) # (n_noise_levels, n_tested_patterns)
            return new_state, y
        def step_fn2(state, inp): # carry = state, inp = (x, noise)
            x, noise = inp # x: (nd, ns), noise: (n_tested_patterns, nd)
            new_state = self.neuron.update(hyperD, state, x)
            y = jnp.full((n_noise_levels, n_tested_patterns), -100.)
            return new_state, y
        def step_fn(carry, y): # carry = (state, idx)
            state, idx = carry
            new_state, y = jax.lax.cond(idx % test_every == 0, step_fn1, step_fn2, state, y)
            return (new_state, idx+1), y
        (new_state, _), ys = jax.lax.scan(step_fn, (state, 0), (xs, noise_arr)) # ys: (resample_every, n_noise_levels, n_tested_patterns)
        return new_state, ys
    
    def run(self, 
            n_init_steps: int,
            n_steps_phase1: int, # must be larger than n_tested_patterns
            n_steps_phase2: int,
            n_steps_phase3: int,
            test_every_phase3: int,
            noise_levels: jax.Array,
            ):
        if self.state is None:
            self.rng, rng1, rng2 = jax.random.split(self.rng, 3)
            self.state = self.neuron.init_state(rng1, self.hyperD)
            self.state = self.init_phase(rng2, self.hyperD, self.state, n_init_steps)

        self.rng, rng1, rng2, rng3, rng4 = jax.random.split(self.rng, 5)
        x0s = self.xs_generator.generate(rng1, self.n_tested_patterns) # (n_tested_patterns, nd, ns)
        self.state, transient_record = self.record_phase1(rng2, self.hyperD, self.state, n_steps_phase1, noise_levels, x0s) # note here noise_levels is passed as bs
        self.state = self.no_record_phase2(rng3, self.hyperD, self.state, n_steps_phase2)
        self.state, steady_record = self.steady_record_phase3(rng4, self.hyperD, self.state, n_steps_phase3, test_every_phase3, noise_levels, x0s) # note here noise_levels is passed as bs

        # exclude empty records in steady_record
        mask = jnp.any(jnp.isclose(steady_record, -100), axis=(1, 2)) # (n_steps_phase3,)
        steady_record = steady_record[~mask] # (n_recorded_steps, n_noise_levels, n_tested_patterns)
        self.upper99_s = np.array(self.get_upper99(steady_record)) # (n_noise_levels,)

        self.transient_record = np.array(transient_record) # (n_steps_phase1, n_noise_levels, n_tested_patterns)
        self.steady_record = np.array(steady_record) # (n_recorded_steps, n_noise_levels, n_tested_patterns)

        # realign the transient_record so that the row index corresponds to the delay
        for i in range(self.n_tested_patterns):
            self.transient_record[:, :, i] = np.roll(self.transient_record[:, :, i], -i, axis=0)
        self.transient_record = self.transient_record[:n_steps_phase1-self.n_tested_patterns]
    
class Driver1_thetaY_record(Driver1_bScan):

    def _update_record1(self, hyperD: HyperDynConfig1, state: State1, bs: jax.Array, xs: jax.Array, x0s: jax.Array): # bs: (bs_size,), xs: (resample_every, nd, ns), x0s: (n_tested_patterns, nd, ns)
        """ Run simulation for `resample_every` steps and record the outputs"""
        bs_size = bs.shape[0]
        def step_fn(carry, x): # carry = state
            new_carry = self.neuron.update(hyperD, carry, x)
            return new_carry, None
        new_state, _ = jax.lax.scan(step_fn, state, xs) # ys: (resample_every, bs_size, n_tested_patterns)
        return new_state, jnp.zeros((self.resample_every, bs_size, self.n_tested_patterns))

    def run(self, 
            n_init_steps: int,
            n_steps_phase1: int, # must be larger than n_tested_patterns
            n_steps_phase2: int,
            n_steps_phase3: int,
            test_every_phase3: int,
            bs: jax.Array,
            ):
        if self.state is None:
            self.rng, rng1, rng2 = jax.random.split(self.rng, 3)
            self.state = self.neuron.init_state(rng1, self.hyperD)
            self.state = self.init_phase(rng2, self.hyperD, self.state, n_init_steps)

        self.rng, rng1, rng2, rng3, rng4 = jax.random.split(self.rng, 5)
        x0s = self.xs_generator.generate(rng1, self.n_tested_patterns) # (n_tested_patterns, nd, ns)
        self.state, _ = self.record_phase1(rng2, self.hyperD, self.state, n_steps_phase1, bs, x0s)
        self.state = self.no_record_phase2(rng3, self.hyperD, self.state, n_steps_phase2)
        self.state, steady_record = self.steady_record_phase3(rng4, self.hyperD, self.state, n_steps_phase3, test_every_phase3, bs, x0s)

        # exclude empty records in steady_record
        mask = jnp.any(jnp.isclose(steady_record, -100), axis=(1, 2)) # (n_steps_phase3,)
        steady_record = steady_record[~mask] # (n_recorded_steps, bs_size, n_tested_patterns)

        self.upper99_s = np.array(self.get_upper99(steady_record)) # (bs_size,)

class Driver1_DendriteNoise_thetaY_record(Driver1_NoisyRecall_DendriteNoise):
    def _update_record1(self, hyperD: HyperDynConfig1, state: State1, bs: jax.Array, xs: jax.Array, x0s: jax.Array): # bs: (bs_size,), xs: (resample_every, nd, ns), x0s: (n_tested_patterns, nd, ns)
        """ Run simulation for `resample_every` steps and record the outputs"""
        nd = xs.shape[1]
        rng = state.rng
        rng_noise, new_rng = jax.random.split(rng)
        noise_arr = jax.random.normal(rng_noise, (self.resample_every, self.n_tested_patterns, nd)) * self.noise_level # (resample_every, n_tested_patterns, nd)
        state = dataclasses.replace(state, rng=new_rng)
        kappa_s = hyperD.bk - bs # (bs_size,)

        def step_fn(carry, inp): # carry = state, inp = (x, noise)
            x, noise = inp # x: (nd, ns), noise: (n_tested_patterns, nd)
            new_carry = self.neuron.update(hyperD, carry, x)
            return new_carry, None
        new_state, _ = jax.lax.scan(step_fn, state, (xs, noise_arr)) # ys: (resample_every, bs_size, n_tested_patterns)
        return new_state, jnp.zeros((self.resample_every, bs.shape[0], self.n_tested_patterns))
    
    def run(self, 
            n_init_steps: int,
            n_steps_phase1: int, # must be larger than n_tested_patterns
            n_steps_phase2: int,
            n_steps_phase3: int,
            test_every_phase3: int,
            bs: jax.Array,
            ):
        if self.state is None:
            self.rng, rng1, rng2 = jax.random.split(self.rng, 3)
            self.state = self.neuron.init_state(rng1, self.hyperD)
            self.state = self.init_phase(rng2, self.hyperD, self.state, n_init_steps)

        self.rng, rng1, rng2, rng3, rng4 = jax.random.split(self.rng, 5)
        x0s = self.xs_generator.generate(rng1, self.n_tested_patterns) # (n_tested_patterns, nd, ns)
        self.state, _ = self.record_phase1(rng2, self.hyperD, self.state, n_steps_phase1, bs, x0s)
        self.state = self.no_record_phase2(rng3, self.hyperD, self.state, n_steps_phase2)
        self.state, steady_record = self.steady_record_phase3(rng4, self.hyperD, self.state, n_steps_phase3, test_every_phase3, bs, x0s)

        # exclude empty records in steady_record
        mask = jnp.any(jnp.isclose(steady_record, -100), axis=(1, 2)) # (n_steps_phase3,)
        steady_record = steady_record[~mask] # (n_recorded_steps, bs_size, n_tested_patterns)

        self.upper99_s = np.array(self.get_upper99(steady_record)) # (bs_size,)

class Driver1_wLens_record(Driver1_bScan):
    def _update_record3_sparse(self, hyperD: HyperDynConfig1, state: State1, bs: None, xs: jax.Array, x0s: None, test_every: int): # bs: (bs_size,), x0s: None
        """ Run simulation for `resample_every` steps and record the outputs"""
        # bs_size = bs.shape[0]
        nd = self.neuron.hyper.nd
        def step_fn1(state, x): # carry = state
            new_state = self.neuron.update(hyperD, state, x)
            w = new_state.w # (nd, ns)
            w_lens = jnp.linalg.norm(w, axis=-1) # (nd,)
            return new_state, w_lens # (nd,)
        def step_fn2(state, x): # carry = state
            new_state = self.neuron.update(hyperD, state, x)
            return new_state, jnp.full((nd,), -100.) # (nd,)
        def step_fn(carry, x): # carry = (state, idx)
            state, idx = carry
            new_state, y = jax.lax.cond(idx % test_every == 0, step_fn1, step_fn2, state, x)
            return (new_state, idx+1), y
        (new_state, _), ys = jax.lax.scan(step_fn, (state, 0), xs) # ys: (resample_every, nd)
        return new_state, ys
    
    @partial(jax.jit, static_argnames=('self', 'n_steps', 'test_every'))
    def steady_record_phase3(self, rng: jax.Array, hyperD: HyperDynConfig1, state: State1, n_steps: int, test_every: int, bs: jax.Array, x0s: jax.Array) -> Tuple[State1, jax.Array]: # bs: (bs_size,), x0s: (n_tested_patterns, nd, ns)
        """ Continue the simulation for `n_steps`, and for every `test_every` steps, analyze the state and record the outputs.
            `n_steps` should be a multiple of resample_every.
            This phase should be used in the steady state. 
        """
        n_big_steps = n_steps // self.resample_every
        rngs = jax.random.split(rng, n_big_steps)
        def step_fn(carry, rng): # carry = state
            xs = self.xs_generator.generate(rng, self.resample_every) # (resample_every, nd, ns)
            new_carry, ys = self._update_record3_sparse(hyperD, carry, None, xs, None, test_every)
            return new_carry, ys
        new_state, ys = jax.lax.scan(step_fn, state, rngs) # ys: (n_big_steps, resample_every, nd)
        return new_state, ys.reshape(-1, ys.shape[-1]) # (n_steps, nd)
    
    def run(self, 
            n_init_steps: int,
            n_steps_phase2: int,
            n_steps_phase3: int,
            test_every_phase3: int,
            bs: jax.Array,
            ):
        if self.state is None:
            self.rng, rng1, rng2 = jax.random.split(self.rng, 3)
            self.state = self.neuron.init_state(rng1, self.hyperD)
            self.state = self.init_phase(rng2, self.hyperD, self.state, n_init_steps)

        self.rng, rng1, rng2, rng3, rng4 = jax.random.split(self.rng, 5)
        self.state = self.no_record_phase2(rng3, self.hyperD, self.state, n_steps_phase2)
        self.state, steady_record = self.steady_record_phase3(rng4, self.hyperD, self.state, n_steps_phase3, test_every_phase3, None, None)

        # exclude empty records in steady_record
        mask = jnp.any(jnp.isclose(steady_record, -100), axis=(1, )) # (n_steps_phase3,)
        steady_record = steady_record[~mask] # (n_recorded_steps, nd)

        self.steady_record = np.array(steady_record) # (n_recorded_steps, bs_size, n_tested_patterns)

class Driver1_WeightCorr_record(Driver1_wLens_record):
    def _update_record3_sparse(self, hyperD: HyperDynConfig1, state: State1, bs: None, xs: jax.Array, x0s: None, test_every: int): # bs: (bs_size,), x0s: None
        """ Run simulation for `resample_every` steps and record the outputs"""
        # bs_size = bs.shape[0]
        nd = self.neuron.hyper.nd
        conn_mat = self.xs_generator.conn_idx_mat # (nd, ns)
        def step_fn1(state, x): # carry = state
            new_state = self.neuron.update(hyperD, state, x)
            w = new_state.w # (nd, ns)
            w_lens = jnp.linalg.norm(w, axis=-1) # (nd,)
            w_A = jnp.zeros((nd, self.xs_generator.nA)) # (nd, nA)
            w_A = w_A.at[jnp.arange(nd)[:, None], conn_mat].set(w) # (nd, nA)
            w_A = w_A / w_lens[:, None] # (nd, nA)
            corr_mat = w_A @ w_A.T # (nd, nd)
            corr_mat = corr_mat.at[jnp.diag_indices_from(corr_mat)].set(0.0)
            mean_corr = jnp.sum(corr_mat)/nd/(nd-1) # scalar
            return new_state, jnp.array([mean_corr]) # (1, )
        def step_fn2(state, x): # carry = state
            new_state = self.neuron.update(hyperD, state, x)
            return new_state, jnp.array([-100.]) # (1, )
        def step_fn(carry, x): # carry = (state, idx)
            state, idx = carry
            new_state, y = jax.lax.cond(idx % test_every == 0, step_fn1, step_fn2, state, x)
            return (new_state, idx+1), y
        (new_state, _), ys = jax.lax.scan(step_fn, (state, 0), xs) # ys: (resample_every, 1)
        return new_state, ys

# endregion drivers-for-Type1-neurons


# region drivers-for-Type2-neurons

class Driver2_bScan:
    def __init__(
            self, 
            neuron: Neuron[HyperConfig2, State2, HyperDynConfig2], 
            xs_generator: Xs_Generator,
            rng: jax.Array, 
            hyperD: HyperDynConfig2 = None,
            initial_state: State2 = None, 
            resample_every: int = 1000,
            n_tested_patterns: int = 1000, # n_tested_patterns should be a multiple of resample_every
            ):
        self.neuron = neuron
        self.xs_generator = xs_generator
        self.rng = rng
        self.hyperD = hyperD
        self.state = initial_state
        self.resample_every = resample_every
        self.n_tested_patterns = n_tested_patterns

        self.transient_record = None
        self.steady_record = None

    def _update_no_record(self, hyperD: HyperDynConfig2, state: State2, xs: jax.Array): # xs: (resample_every, nd, ns)
        """ Update the state by xs without recording the outputs"""
        def step_fn(carry, x): # carry = state
            new_carry = self.neuron.update(hyperD, carry, x)
            return new_carry, None
        new_state, _ = jax.lax.scan(step_fn, state, xs)
        return new_state

    def _update_record1(self, hyperD: HyperDynConfig2, state: State2, bs: jax.Array, xs: jax.Array, x0s: jax.Array): # bs: (bs_size,), xs: (resample_every, nd, ns), x0s: (n_tested_patterns, nd, ns)
        """ Run simulation for `resample_every` steps and record the outputs"""
        def step_fn(carry, x): # carry = state
            new_carry = self.neuron.update(hyperD, carry, x)
            w, shift = new_carry.w, new_carry.shift # (nd, ns), (nd,)
            us = jnp.sum(x0s * w, axis=-1) # (n_tested_patterns, nd)
            z = jax.nn.relu(us[None, :, :] - bs[:, None, None] - shift[None, None, :]) # (bs_size, n_tested_patterns, nd)
            y = jnp.sum(z, axis=-1) # (bs_size, n_tested_patterns)
            return new_carry, y
        new_state, ys = jax.lax.scan(step_fn, state, xs) # ys: (resample_every, bs_size, n_tested_patterns)
        return new_state, ys

    def _update_record3_sparse(self, hyperD: HyperDynConfig2, state: State2, bs: jax.Array, xs: jax.Array, x0s: jax.Array, test_every: int): # bs: (bs_size,), x0s: (n_tested_patterns, nd, ns)
        """ Run simulation for `resample_every` steps and record the outputs"""
        n_tested_patterns = x0s.shape[0]
        bs_size = bs.shape[0]
        def step_fn1(state, x): # carry = state
            new_state = self.neuron.update(hyperD, state, x)
            w, shift = new_state.w, new_state.shift # (nd, ns), (nd,)
            us = jnp.sum(x0s * w, axis=-1) # (n_tested_patterns, nd)
            z = jax.nn.relu(us[None, :, :] - bs[:, None, None] - shift[None, None, :]) # (bs_size, n_tested_patterns, nd)
            y = jnp.sum(z, axis=-1) # (bs_size, n_tested_patterns)
            return new_state, y
        def step_fn2(state, x): # carry = state
            new_state = self.neuron.update(hyperD, state, x)
            y = jnp.full((bs_size, n_tested_patterns), -100.)
            return new_state, y
        def step_fn(carry, y): # carry = (state, idx)
            state, idx = carry
            new_state, y = jax.lax.cond(idx % test_every == 0, step_fn1, step_fn2, state, y)
            return (new_state, idx+1), y
        (new_state, _), ys = jax.lax.scan(step_fn, (state, 0), xs) # ys: (resample_every, bs_size, n_tested_patterns)
        return new_state, ys

    @partial(jax.jit, static_argnames=('self', 'n_steps'))
    def init_phase(self, rng: jax.Array, hyperD: HyperDynConfig2, state: State2, n_steps: int) -> State2:
        """ Initialize the state by `n_steps` patterns. 
            `n_steps` should be a multiple of resample_every.
        """
        n_big_steps = n_steps // self.resample_every
        rngs = jax.random.split(rng, n_big_steps)
        def step_fn(carry, rng): # carry = state
            xs = self.xs_generator.generate(rng, self.resample_every) # (resample_every, nd, ns)
            new_carry = self._update_no_record(hyperD, carry, xs)
            return new_carry, None
        new_state, _ = jax.lax.scan(step_fn, state, rngs)
        return new_state

    @partial(jax.jit, static_argnames=('self', 'n_steps'))
    def record_phase1(self, rng: jax.Array, hyperD: HyperDynConfig2, state: State2, n_steps: jax.Array, bs: jax.Array, x0s: jax.Array) -> Tuple[State2, jax.Array]: # bs: (bs_size,), x0s: (n_tested_patterns, nd, ns)
        """ Run simulation for `n_steps`, and for each step, test the state by `n_tested_patterns` patterns and record the outputs.
        `n_steps` should be a multiple of `resample_every`."""
        x0s_padded_len = (self.resample_every - self.n_tested_patterns) % self.resample_every
        x0s_padded = jnp.pad(x0s, ((0, x0s_padded_len), (0,0), (0,0)), mode='constant', constant_values=0.) # (n_tested_patterns + x0s_padded_len, nd, ns), first dim is a multiple of resample_every

        n_big_steps = n_steps // self.resample_every
        rngs = jax.random.split(rng, n_big_steps)
        def step_fn(carry, rng): # carry = (state, idx)
            state, idx = carry

            # generate xs
            xs = self.xs_generator.generate(rng, self.resample_every) # (resample_every, nd, ns)
            tmp_x0_padded = jax.lax.dynamic_slice(x0s_padded, (idx*self.resample_every, 0, 0), (self.resample_every, x0s.shape[1], x0s.shape[2])) # (resample_every, nd, ns)
            mask_array = ( jnp.arange(self.resample_every)+idx*self.resample_every ) < self.n_tested_patterns
            xs = jnp.where(mask_array[:, None, None], tmp_x0_padded, xs) # (resample_every, nd, ns)

            new_state, ys = self._update_record1(hyperD, state, bs, xs, x0s)
            return (new_state, idx+1), ys

        (new_state, _), ys = jax.lax.scan(step_fn, (state, 0), rngs) # ys: (n_big_steps, resample_every, bs_size, n_tested_patterns)
        return new_state, ys.reshape(-1, ys.shape[-2], ys.shape[-1]) # (n_steps, bs_size, n_tested_patterns)

    def no_record_phase2(self, rng: jax.Array, hyperD: HyperDynConfig2, state: State2, n_steps: int) -> State2:
        """ Continue the simulation for `n_steps` without recording the outputs.
            `n_steps` should be a multiple of resample_every.
        """
        n_big_steps = n_steps // self.resample_every
        rngs = jax.random.split(rng, n_big_steps)
        def step_fn(carry, rng): # carry = state
            xs = self.xs_generator.generate(rng, self.resample_every) # (resample_every, nd, ns)
            new_carry = self._update_no_record(hyperD, carry, xs)
            return new_carry, None
        new_state, _ = jax.lax.scan(step_fn, state, rngs)
        return new_state
    
    @partial(jax.jit, static_argnames=('self', 'n_steps', 'test_every'))
    def steady_record_phase3(self, rng: jax.Array, hyperD: HyperDynConfig2, state: State2, n_steps: int, test_every: int, bs: jax.Array, x0s: jax.Array) -> Tuple[State2, jax.Array]: # bs: (bs_size,), x0s: (n_tested_patterns, nd, ns)
        """ Continue the simulation for `n_steps`, and for every `test_every` steps, test the state by `n_tested_patterns` patterns and record the outputs.
            `n_steps` should be a multiple of resample_every.
            This phase should be used in the steady state. 
        """
        n_big_steps = n_steps // self.resample_every
        rngs = jax.random.split(rng, n_big_steps)
        def step_fn(carry, rng): # carry = state
            xs = self.xs_generator.generate(rng, self.resample_every) # (resample_every, nd, ns)
            new_carry, ys = self._update_record3_sparse(hyperD, carry, bs, xs, x0s, test_every)
            return new_carry, ys
        new_state, ys = jax.lax.scan(step_fn, state, rngs) # ys: (n_big_steps, resample_every, bs_size, n_tested_patterns)
        return new_state, ys.reshape(-1, ys.shape[-2], ys.shape[-1]) # (n_steps, bs_size, n_tested_patterns)
    
    def run(self, 
            n_init_steps: int,
            n_steps_phase1: int, # must be larger than n_tested_patterns
            n_steps_phase2: int,
            n_steps_phase3: int,
            test_every_phase3: int,
            bs: jax.Array,
            ):
        if self.state is None:
            self.rng, rng1, rng2 = jax.random.split(self.rng, 3)
            self.state = self.neuron.init_state(rng1, self.hyperD)
            self.state = self.init_phase(rng2, self.hyperD, self.state, n_init_steps)

        self.rng, rng1, rng2, rng3, rng4 = jax.random.split(self.rng, 5)
        x0s = self.xs_generator.generate(rng1, self.n_tested_patterns) # (n_tested_patterns, nd, ns)
        self.state, transient_record = self.record_phase1(rng2, self.hyperD, self.state, n_steps_phase1, bs, x0s)
        self.state = self.no_record_phase2(rng3, self.hyperD, self.state, n_steps_phase2)
        self.state, steady_record = self.steady_record_phase3(rng4, self.hyperD, self.state, n_steps_phase3, test_every_phase3, bs, x0s)

        self.transient_record = np.array(transient_record) # (n_steps_phase1, bs_size, n_tested_patterns)
        self.steady_record = np.array(steady_record) # (n_steps_phase3, bs_size, n_tested_patterns)

        # realign the transient_record so that the row index corresponds to the delay
        for i in range(self.n_tested_patterns):
            self.transient_record[:, :, i] = np.roll(self.transient_record[:, :, i], -i, axis=0)
        self.transient_record = self.transient_record[:n_steps_phase1-self.n_tested_patterns]

        # exclude empty records in steady_record
        mask = np.any(np.isclose(self.steady_record, -100), axis=(1, 2)) # (n_steps_phase3,)
        self.steady_record = self.steady_record[~mask] # (n_recorded_steps, bs_size, n_tested_patterns)


class Driver2_shift_freq_record:
    def __init__(
            self, 
            neuron: Neuron[HyperConfig2, State2, HyperDynConfig2], 
            xs_generator: Xs_Generator,
            rng: jax.Array, 
            hyperD: HyperDynConfig2 = None,
            initial_state: State2 = None,
            resample_every: int = 1000,
            ):
        self.neuron = neuron
        self.xs_generator = xs_generator
        self.rng = rng
        self.hyperD = hyperD
        self.state = initial_state
        self.resample_every = resample_every
        self.freq_record = None
        self.shift_record = None

    def _update_record(self, hyperD: HyperDynConfig2, state: State2, xs: jax.Array): # xs: (resample_every, nd, ns)
        """ Update the state by xs without recording the outputs"""
        def step_fn(carry, x): # carry = state
            new_carry = self.neuron.update(hyperD, carry, x)
            return new_carry, (new_carry.freq, new_carry.shift)
        new_state, records = jax.lax.scan(step_fn, state, xs)
        return new_state, records # records: ((resample_every, nd), (resample_every, nd))
    
    @partial(jax.jit, static_argnames=('self', 'n_steps'))
    def update_record(self, rng: jax.Array, hyperD: HyperDynConfig2, state: State2, n_steps: int) -> State2:
        """ Initialize the state by `n_steps` patterns. 
            `n_steps` should be a multiple of resample_every.
        """
        n_big_steps = n_steps // self.resample_every
        rngs = jax.random.split(rng, n_big_steps)
        def step_fn(carry, rng): # carry = state
            xs = self.xs_generator.generate(rng, self.resample_every) # (resample_every, nd, ns)
            new_carry, records = self._update_record(hyperD, carry, xs)
            return new_carry, records # records: ((resample_every, nd), (resample_every, nd))
        new_state, records = jax.lax.scan(step_fn, state, rngs) # records: ((n_big_steps, resample_every, nd), (n_big_steps, resample_every, nd))
        return new_state, records

    def run(self, n_steps: int):
        if self.state is None:
            self.rng, rng1 = jax.random.split(self.rng, 2)
            self.state = self.neuron.init_state(rng1, self.hyperD)

        self.rng, rng1 = jax.random.split(self.rng, 2)
        self.state, records = self.update_record(rng1, self.hyperD, self.state, n_steps)
        self.freq_record = np.array(records[0].reshape(-1, records[0].shape[-1])) # (n_steps, nd)
        self.shift_record = np.array(records[1].reshape(-1, records[1].shape[-1])) # (n_steps, nd)

# endregion drivers-for-Type2-neurons