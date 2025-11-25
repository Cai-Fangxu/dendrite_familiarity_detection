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

noise_levels = jnp.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])  # (n_levels,)
n_noise_levels = noise_levels.shape[0]

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

    if line_idx < 520:
        coeff = "infty"
    elif line_idx < 1040:
        coeff = 4
    elif line_idx < 1560:
        coeff = 4
    else:
        coeff = 4

    # Initialization phase. Recompile only when nd or n changes.
    rng, subkey = jax.random.split(rng)
    if nd != old_nd or n != old_n:
        old_nd, old_n = nd, n
        hyper = neurons.HyperConfig1(nd=nd, ns=ns, n=n, b=bias_center, bk=bk, beta=beta) # the values of b, bk, beta here are just placeholders here, they will be overwritten in the hyperD. 
        hyperD = neurons.HyperDynConfig1(b=bias_center, bk=bk, beta=beta)
        neuron = neurons.Neuron_Decay_Int(hyperparameters=hyper)
        # neuron = neurons.Neuron_Decay_NonInt(hyperparameters=hyper)
        if coeff == "infty":
            xs_gen = input_gen.Normal(nd=nd, ns=ns)
        else:
            subkey, subkey2 = jax.random.split(subkey)
            xs_gen = input_gen.Normal_Corr_across_Dendrites(nd=nd, ns=ns, coeff=coeff, key=subkey2)
        # my_driver = driver.Driver1_NoisyRecall_SomaNoise3Scan(neuron=neuron, xs_generator=xs_gen, rng=subkey, hyperD=hyperD, initial_state=None, resample_every=1000, n_tested_patterns=1000)
        my_driver = driver.Driver1_NoisyRecall_DendriteNoiseScan(neuron=neuron, xs_generator=xs_gen, rng=subkey, hyperD=hyperD, initial_state=None, resample_every=1000, n_tested_patterns=1000)
    else:
        hyperD = neurons.HyperDynConfig1(b=bias_center, bk=bk, beta=beta)
        my_driver.re_init(rng=subkey, hyperD=hyperD, initial_state=None)

    # Run simulation and get results
    capa_list = np.zeros((n_noise_levels, n_repeat))
    for repeat_idx in range(n_repeat):
        print(repeat_idx+1, end='    ', flush=True)
        my_driver.run(
            n_init_steps=10000,
            n_steps_phase1=3000,
            n_steps_phase2=20000,
            n_steps_phase3=10000,
            test_every_phase3=50,
            noise_levels=noise_levels
        )  # no need to re-initialize for different repeats

        capas = utils.capacity_numba(my_driver.transient_record, my_driver.steady_record, upper99_s=my_driver.upper99_s)  # (n_noise_levels,)
        capa_list[:, repeat_idx] = capas

    tmp_data = np.zeros((n_noise_levels, 4 + n_repeat))  # (n_noise_levels, 4+n_repeat)
    tmp_data[:, 0] = noise_levels
    tmp_data[:, 1] = bias_center
    tmp_data[:, 2] = bk
    tmp_data[:, 3] = beta
    tmp_data[:, 4:] = capa_list

    if line_idx < 520:
        folder_path = f"{tmp_data_folder}coeff_infty/nd{nd}_ns{ns}/"
    elif line_idx < 1040:
        folder_path = f"{tmp_data_folder}coeff_4/dendrite_noise_0.2_soma_noise_0.1/nd{nd}_ns{ns}/"
    elif line_idx < 1560:
        folder_path = f"{tmp_data_folder}coeff_4/dendrite_noise_0.25_soma_noise_0.1/nd{nd}_ns{ns}/"
    else:
        folder_path = f"{tmp_data_folder}coeff_4/dendrite_noise_0.3_soma_noise_0.1/nd{nd}_ns{ns}/"
    os.makedirs(folder_path, exist_ok=True)
    with open(f"{folder_path}n{n}.csv", 'a') as f:
        np.savetxt(f, tmp_data, delimiter=',', fmt=['%.3f', '%.3f', '%.3f', '%.6f']+['%.2f' for _ in range(n_repeat)])

    # Clear memory after each iteration
    my_driver.transient_record = None
    my_driver.steady_record = None
    del tmp_data, capa_list, capas
    gc.collect()
    
    # Periodically clear JAX cache to prevent compilation cache buildup
    if (line_idx + 1) % 5 == 0:
        jax.clear_caches()