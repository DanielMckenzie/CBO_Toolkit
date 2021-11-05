# invokes the SignOPT algorithm class.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 11:51:58 2021

@author: danielmckenzie
"""

from Algorithms.SignOPT2 import SignOPT
import numpy as np
from ExampleCode.oracle import Oracle
import matplotlib.pyplot as plt
from ExampleCode.benchmarkfunctions import SparseQuadratic, MaxK

# Defining the function
n_def = 2000
s_exact = 200
noise_amp = 0.001
# objective functions.
obj_func_1 = SparseQuadratic(n_def, s_exact, noise_amp)
obj_func_2 = MaxK(n_def, s_exact, noise_amp)
# invoke.
function_budget = int(1e5)
m = 100
x0 = np.random.randn(n_def)
step_size = 0.2
r = 0.1
# max_iters = int(2e4)
# max_iters = int(10000)

# Define the comparison oracle.
oracle = Oracle(obj_func_1)

Opt = SignOPT(oracle, function_budget, x0, m, step_size, r, debug=False, function=obj_func_1)
# step.
termination = False
i = 0
while termination is False:
    # optimization step.
    solution, func_value, termination, queries = Opt.step()
    # print('step')
    print('current value at ' + str(i) + ': ', func_value[-1])
# plot the decreasing function.
plt.plot(func_value)
plt.show()
plt.close()
# log x-axis.
plt.semilogy(func_value)
plt.show()
plt.close()

print('number of values: ', len(func_value))
print('number of oracle queries: ', queries)
# ---------




'''
for i in range(max_iters - 1):
    print(i)
    Opt.step()

plt.semilogy(Opt.f_vals)
plt.show()

print('number of function vals: ', len(Opt.f_vals))
'''

