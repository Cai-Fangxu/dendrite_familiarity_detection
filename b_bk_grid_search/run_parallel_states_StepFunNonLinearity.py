"""
Grid search for hyperparameters b and bk
This script is almost identical to run.py, except that it uses the Heaviside step function for dendritic non-linearity. 
The actual base neuron model used can vary according to need.  
"""

import sys
sys.path.insert(0, '..')


import os
import gc
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

from lib import neurons, input_gen, driver, utils

# Define neuron and driver classes for the Heaviside step function non-linearity
####################################################################
class MyNeuron(neurons.Neuron_Decay_Int_Variable_n_min1):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)

    @partial(jax.jit, static_argnames=('self',))
    def _forward(self, hyperD: neurons.HyperDynConfig1, state: neurons.State1, x: jnp.ndarray): # x: (B, nd, ns)
        w = state.w # (nd, ns)
        u = jnp.sum(w * x, axis=-1) # (B, nd)
        z = jnp.heaviside(u - hyperD.b, 0.0) # (B, nd)
        y = jnp.sum(z, axis=-1) # (B,)
        return y
    
class MyDriver(driver.Driver1_parallel_states):
    def __init__(self, neuron, xs_generator, rng, hyperD, initial_state=None, resample_every=1000, n_tested_patterns=1000):
        super().__init__(neuron, xs_generator, rng, hyperD, initial_state, resample_every, n_tested_patterns)

    def _update_record1(self, hyperD: neurons.HyperDynConfig1, state: neurons.State1, bs: jax.Array, xs: jax.Array, x0s: jax.Array): # bs=None (not used), xs: (resample_every, nd, ns), x0s: (n_tested_patterns, nd, ns)
        """ Run simulation for `resample_every` steps and record the outputs"""
        neuron_update_vmapped = jax.vmap(self.neuron.update, in_axes=(0, 0, None))
        bs = hyperD.b # bs: (np,)
        def step_fn(carry, x): # carry = state
            new_carry = neuron_update_vmapped(hyperD, carry, x)
            w = new_carry.w # (np, nd, ns)
            us = jnp.sum(x0s[None, :, :, :] * w[:, None, :, :], axis=-1) # (np, n_tested_patterns, nd)
            z = jnp.heaviside(us - bs[:, None, None], 0.0) # (np, n_tested_patterns, nd)
            y = jnp.sum(z, axis=-1) # (np, n_tested_patterns)
            return new_carry, y
        new_state, ys = jax.lax.scan(step_fn, state, xs) # ys: (resample_every, np, n_tested_patterns)
        return new_state, ys
    
    def _update_record3_sparse(self, hyperD: neurons.HyperDynConfig1, state: neurons.State1, bs: jax.Array, xs: jax.Array, x0s: jax.Array, test_every: int): # bs=None (not used), x0s: (n_tested_patterns, nd, ns)
        """ Run simulation for `resample_every` steps and record the outputs"""
        neuron_update_vmapped = jax.vmap(self.neuron.update, in_axes=(0, 0, None))
        bs = hyperD.b # bs: (np,)
        n_tested_patterns = x0s.shape[0]
        n_parallel_states = bs.shape[0] # np
        def step_fn1(state, x): # carry = state
            new_state = neuron_update_vmapped(hyperD, state, x)
            w = new_state.w # (np, nd, ns)
            us = jnp.sum(x0s[None, :, :, :] * w[:, None, :, :], axis=-1) # (np, n_tested_patterns, nd)
            z = jnp.heaviside(us - bs[:, None, None], 0.0) # (np, n_tested_patterns, nd)
            y = jnp.sum(z, axis=-1) # (np, n_tested_patterns)
            return new_state, y
        def step_fn2(state, x): # carry = state
            new_state = neuron_update_vmapped(hyperD, state, x)
            y = jnp.full((n_parallel_states, n_tested_patterns), -100.)
            return new_state, y
        def step_fn(carry, y): # carry = (state, idx)
            state, idx = carry
            new_state, y = jax.lax.cond(idx % test_every == 0, step_fn1, step_fn2, state, y)
            return (new_state, idx+1), y
        (new_state, _), ys = jax.lax.scan(step_fn, (state, 0), xs) # ys: (resample_every, np, n_tested_patterns)
        return new_state, ys

####################################################################

tmp_data_folder = "../tmp_data/"
n_repeat = 6

bs_size = 9
b_spacing = 0.15

# each row of params list is: nd, ns, bk, bias_center, n, seed, *betas
params_list = np.genfromtxt(f"{tmp_data_folder}params_list_tmp.csv", delimiter=',')
n_lines = params_list.shape[0]

old_nd, old_n = 0, 0
for line_idx in range(n_lines):
    tmp = params_list[line_idx]
    nd, ns, bk, bias_center, n, seed, *betas = tmp
    seed = int(seed)
    nd = int(nd)
    ns = int(ns)
    n = int(n)
    rng = jax.random.PRNGKey(seed)
    print(" ", flush=True)
    print(f"#############Line{line_idx+1}/{n_lines}#####################", flush=True)
    print(f"nd: {nd}; n: {n}; bk: {bk:.3f}", flush=True)

    bs = jnp.linspace(bias_center-b_spacing*(bs_size//2), bias_center+b_spacing*(bs_size//2), bs_size)
    betas = jnp.array(betas)

    # Initialization phase. Recompile only when nd or n changes.
    rng, subkey = jax.random.split(rng)
    if nd != old_nd or n != old_n:
        old_nd, old_n = nd, n
        hyper = neurons.HyperConfig1(nd=nd, ns=ns, n=n, b=bias_center, bk=bk, beta=None) # the values of b, bk, beta here are just placeholders here, they will be overwritten in the hyperD. 
        hyperD = neurons.HyperDynConfig1(b=bs, bk=jnp.full(bs_size, bk), beta=betas) # all attributes have a leading dimension of bs_size
        # neuron = neurons.Neuron_Decay_Int_Variable_n(hyperparameters=hyper)
        neuron = MyNeuron(hyperparameters=hyper)
        xs_gen = input_gen.Normal(nd=nd, ns=ns)
        # subkey, subkey2 = jax.random.split(subkey)
        # xs_gen = input_gen.Normal_Corr_across_Dendrites(nd=nd, ns=ns, coeff=1, key=subkey2)
        my_driver = MyDriver(neuron=neuron, xs_generator=xs_gen, rng=subkey, hyperD=hyperD, initial_state=None, resample_every=1000, n_tested_patterns=1000)
    else:
        hyperD = neurons.HyperDynConfig1(b=bs, bk=jnp.full(bs_size, bk), beta=betas)
        my_driver.re_init(rng=subkey, hyperD=hyperD, initial_state=None)

    # Run simulation and get results
    capa_list = np.zeros((bs_size, n_repeat))
    for repeat_idx in range(n_repeat):
        print(repeat_idx+1, end='    ', flush=True)
        my_driver.run(
            n_init_steps=10000,
            n_steps_phase1=3000,
            n_steps_phase2=20000,
            n_steps_phase3=10000,
            test_every_phase3=50,
            bs=None   # the argument is not used in this driver
        )  # no need to re-initialize for different repeats

        capas = utils.capacity_numba(my_driver.transient_record, my_driver.steady_record)  # (bs_size,)
        capa_list[:, repeat_idx] = capas

    tmp_data = np.zeros((bs_size, 3 + n_repeat))  # (bs_size, 3+n_repeat)
    tmp_data[:, 0] = bs
    tmp_data[:, 1] = bk
    tmp_data[:, 2] = betas
    tmp_data[:, 3:] = capa_list
    folder_path = f"{tmp_data_folder}/int_variable_n_min1/nd{nd}_ns{ns}/"
    os.makedirs(folder_path, exist_ok=True)
    with open(f"{folder_path}n{n}.csv", 'a') as f:
        np.savetxt(f, tmp_data, delimiter=',', fmt=['%.3f', '%.3f', '%.6f']+['%.2f' for _ in range(n_repeat)])


    # Clear memory after each iteration
    my_driver.transient_record = None
    my_driver.steady_record = None
    del tmp_data, capa_list, capas
    gc.collect()
    
    # Periodically clear JAX cache to prevent compilation cache buildup
    if (line_idx + 1) % 20 == 0:
        jax.clear_caches()