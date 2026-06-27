"""
Grid search for hyperparameters b and bk
"""

import sys
sys.path.insert(0, '..')


import os
import gc
import numpy as np
import jax
import jax.numpy as jnp

from lib import neurons, input_gen, driver, utils

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
    betas = jnp.full(bs_size, -1.) # dummy betas since they are not used in this model

    # Initialization phase. Recompile only when nd or n changes.
    rng, subkey = jax.random.split(rng)
    if nd != old_nd or n != old_n:
        ns_max = int(1.5*ns)
        ns_min = int(0.5*ns)
        if nd!= old_nd:
            ns_array_list = jax.random.randint(subkey, (n_repeat, nd,), minval=ns_min, maxval=ns_max+1) # this is shared for across n, bk and b values, different ns_array is used for different repeats and different nd values.
            l_array_list = jnp.sqrt(ns_array_list)
            xs_gen = input_gen.Normal_VariableNs(nd, ns_max)
            ns_mask = xs_gen.get_mask(ns_array_list[0]) # shape (nd, ns_max), used to mask the inputs
        old_nd, old_n = nd, n

        hyper = neurons.HyperConfig2(nd=nd, ns=ns, ns_max=ns_max, n=n, b=bias_center, bk=bk, beta=None, l_array=l_array_list[0], ns_mask=ns_mask) # the values that also exist in hyperD are just placeholders here, they will be overwritten in the hyperD. 
        hyperD = neurons.HyperDynConfig2(b=bs, bk=jnp.full(bs_size, bk), beta=betas, l_array=jnp.repeat(l_array_list[0][None, :], bs_size, axis=0), ns_mask=jnp.repeat(ns_mask[None, :, :], bs_size, axis=0)) # all attributes have a leading dimension of bs_size
        neuron = neurons.Neuron_Decay_Int_VariableNs(hyperparameters=hyper)
        my_driver = driver.Driver2_parallel_states(neuron=neuron, xs_generator=xs_gen, rng=subkey, hyperD=hyperD, initial_state=None, resample_every=1000, n_tested_patterns=1000)

    # Run simulation and get results
    capa_list = np.zeros((bs_size, n_repeat))
    for repeat_idx in range(n_repeat):
        print(repeat_idx+1, end='    ', flush=True)

        ns_mask = xs_gen.get_mask(ns_array_list[repeat_idx]) # shape (nd, ns_max), used to mask the inputs
        hyperD = neurons.HyperDynConfig2(b=bs, bk=jnp.full(bs_size, bk), beta=betas, l_array=jnp.repeat(l_array_list[repeat_idx][None, :], bs_size, axis=0), ns_mask=jnp.repeat(ns_mask[None, :, :], bs_size, axis=0)) # all attributes have a leading dimension of bs_size
        my_driver.re_init(rng=subkey, hyperD=hyperD, initial_state=None)

        my_driver.run(
            n_init_steps=10000,
            n_steps_phase1=3000,
            n_steps_phase2=20000,
            n_steps_phase3=10000,
            test_every_phase3=50,
            bs=None   # the argument is not used in this driver
        )  # no need to re-initialize for different repeats

        print("Calculating capacities...", end=' ', flush=True)

        capas = utils.capacity_numba(my_driver.transient_record, my_driver.steady_record, my_driver.upper99_s)  # (bs_size,)
        capa_list[:, repeat_idx] = capas

    tmp_data = np.zeros((bs_size, 3 + n_repeat))  # (bs_size, 3+n_repeat)
    tmp_data[:, 0] = bs
    tmp_data[:, 1] = bk
    tmp_data[:, 2] = betas
    tmp_data[:, 3:] = capa_list
    folder_path = f"{tmp_data_folder}/int_variableNs/nd{nd}_ns{ns}/"
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