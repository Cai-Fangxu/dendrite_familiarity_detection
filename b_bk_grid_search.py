"""
Grid search for hyperparameters b and bk
"""

import sys
import numpy as np
import jax
import jax.numpy as jnp

from lib import neurons, input_gen, driver, utils

tmp_data_folder = "../tmp_data_new/"
n_repeat = 6

params_list = np.genfromtxt(f"{tmp_data_folder}params_list_tmp.csv", delimiter=',')
params_list = params_list[idx]

print("params=",  params_list, flush=True)

nd, ns, bk, bias_center, ndR, beta, seed = params_list
seed = int(seed)
nd = int(nd)
ns = int(ns)
ndR = int(ndR)
rng = np.random.default_rng(seed)