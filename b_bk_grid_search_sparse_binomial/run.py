"""
Grid search for hyperparameters b and bk
"""

import sys
sys.path.insert(0, '..')


import os
import numpy as np
import jax
import jax.numpy as jnp

from lib import neurons, input_gen, driver, utils

tmp_data_folder = "../tmp_data/"
n_repeat = 6

bs_size = 13
b_spacing = 0.15

params_list = np.genfromtxt(f"{tmp_data_folder}params_list_tmp.csv", delimiter=',')
n_lines = params_list.shape[0]

old_nd, old_n = 0, 0
for line_idx in range(n_lines):
    tmp = params_list[line_idx]
    nd, ns, bk, bias_center, n, beta, seed = tmp
    seed = int(seed)
    nd = int(nd)
    ns = int(ns)
    n = int(n)
    rng = jax.random.PRNGKey(seed)
    print(" ", flush=True)
    print(f"#############Line{line_idx+1}/{n_lines}#####################", flush=True)
    print(f"nd: {nd}; n: {n}; bk: {bk:.3f}", flush=True)

    bs = jnp.linspace(bias_center-b_spacing*(bs_size//2), bias_center+b_spacing*(bs_size//2), bs_size)

    # Initialization phase. Recompile only when nd or n changes.
    rng, subkey = jax.random.split(rng)
    if nd != old_nd or n != old_n:
        old_nd, old_n = nd, n
        hyper = neurons.HyperConfig1(nd=nd, ns=ns, n=n, b=bias_center, bk=bk, beta=beta) # the values of b, bk, beta here are just placeholders here, they will be overwritten in the hyperD. 
        hyperD = neurons.HyperDynConfig1_SynapticBound(b=bias_center, bk=bk, beta=beta, reset_from=3.0, reset_to=3.0)
        neuron = neurons.Neuron_Decay_Int_SynapticBound(hyperparameters=hyper)
        # xs_gen = input_gen.Normal(nd=nd, ns=ns)
        subkey, subkey2 = jax.random.split(subkey)
        xs_gen = input_gen.Sparse_Binomial(nd=nd, ns=ns, fA=0.15, mu=4, n_burst=4, key=subkey2)
        my_driver = driver.Driver1_bScan(neuron=neuron, xs_generator=xs_gen, rng=subkey, hyperD=hyperD, initial_state=None, resample_every=1000, n_tested_patterns=1000)
    else:
        hyperD = neurons.HyperDynConfig1_SynapticBound(b=bias_center, bk=bk, beta=beta, reset_from=3.0, reset_to=3.0)
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
            bs=bs
        )  # no need to re-initialize for different repeats

        capas = utils.capacity_numba(my_driver.transient_record, my_driver.steady_record)  # (bs_size,)
        capa_list[:, repeat_idx] = capas

    tmp_data = np.zeros((bs_size, 3 + n_repeat))  # (bs_size, 3+n_repeat)
    tmp_data[:, 0] = bs
    tmp_data[:, 1] = bk
    tmp_data[:, 2] = beta
    tmp_data[:, 3:] = capa_list
    folder_path = f"{tmp_data_folder}nd{nd}_ns{ns}/"
    os.makedirs(folder_path, exist_ok=True)
    with open(f"{folder_path}n{n}.csv", 'a') as f:
        np.savetxt(f, tmp_data, delimiter=',', fmt=['%.3f', '%.3f', '%.6f']+['%.2f' for _ in range(n_repeat)])
