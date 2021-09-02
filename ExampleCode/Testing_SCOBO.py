#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 13:49:07 2021

@author: danielmckenzie
"""

from scobo_optimizer import SCOBOoptimizer
import numpy as np
from oracle import Oracle
import matplotlib.pyplot as plt
from benchmarkfunctions import SparseQuadratic, MaxK

# Defining the function
n_def = 2000
s_exact = 20
noise_amp = 0.001
func = SparseQuadratic(n_def, s_exact, noise_amp)
query_budget = int(1e5)
m = 100  # Should always be larger than s_exact
x0 = 100*np.random.randn(n_def)
step_size = 0.5
r = 0.1
max_iters = int(2e2)

# Define the comparison oracle
oracle = Oracle(func)

Opt = SCOBOoptimizer(oracle, step_size, query_budget, x0, r, m, s_exact, objfunc=func)

for i in range(max_iters):
    print(i)
    err = Opt.step()
    print(err)

plt.semilogy(Opt.function_vals)
plt.show()