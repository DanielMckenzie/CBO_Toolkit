# invokes the SignOPT algorithm class.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 11:51:58 2021

@author: danielmckenzie
"""

from Algorithms.SignOPT2 import SignOPT
import numpy as np
from ExampleCode.oracle import Oracle
import matplotlib.pyplot as plt
from ExampleCode.benchmarkfunctions import SparseQuadratic

# Defining the function
n_def = 2000
s_exact = 200
noise_amp = 0.001
func = SparseQuadratic(n_def, s_exact, noise_amp)
function_budget = int(1e5)
m = 100
x0 = np.random.randn(n_def)
step_size = 0.2
r = 0.1
# max_iters = int(2e4)
max_iters = int(10000)

# Define the comparison oracle
oracle = Oracle(func)

Opt = SignOPT(oracle, function_budget, x0, m, step_size, r, debug=False, function=func)

for i in range(max_iters - 1):
    print(i)
    Opt.step()
    
plt.semilogy(Opt.f_vals)
plt.show()

print('number of function vals: ', len(Opt.f_vals))
    
