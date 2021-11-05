#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 12:58:51 2021

@author: danielmckenzie
"""

from Algorithms.CMA_2 import CMA
import numpy as np
from ExampleCode.oracle import Oracle
import matplotlib.pyplot as plt
from ExampleCode.benchmarkfunctions import SparseQuadratic, MaxK

# Defining the function
n_def = 100
s_exact = 20
noise_amp = 0.001
func = SparseQuadratic(n_def, s_exact, noise_amp)
query_budget = int(1e9)
m = 100
#x0 = 100 * np.random.randn(n_def)
x0 = np.random.randn(n_def)
step_size = 0.2
r = 0.1
max_iters = int(1000)
lam = 10
mu = 5
sigma = 0.5

# Define the comparison oracle
oracle = Oracle(func)

all_func_vals = [func(x0)]

Opt = CMA(oracle, query_budget, x0, lam, mu, sigma, function=func)
# step.
termination = False
prev_evals = 0
i = 0
while termination is False:
    # optimization step.
    solution, func_value, termination, queries = Opt.step()
    # print('step')
    print('current value at ' + str(i) + ': ', func_value[-1])
    if i > 1:
        if np.abs(func_value[-1] - all_func_vals[-1]) < 1e-6:
            all_func_vals.append(func_value[-1])
            termination = False
            break
        all_func_vals.append(func_value[-1])
    i += 1
# print the solution.
print('\n')
print('solution: ', solution)
# plot the decreasing function.
plt.plot(func_value)
plt.show()
plt.close()
# log x-axis.
plt.semilogy(func_value)
plt.show()
plt.close()

print('\n')
print('number of values: ', len(func_value))
print('number of oracle queries: ', queries)


# __________________________________________________
'''
for i in range(max_iters):
    # print(i)
    val = Opt.step()
    print(str(i) + ': ' + str(val))
    
    # pseudocode.
    # _________
    
    if i > 1:
        if np.abs(val - all_func_vals[-1]) < 1e-6:
            all_func_vals.append(val)
            break
    all_func_vals.append(val)


plt.plot(all_func_vals)
plt.show()
plt.close()
plt.semilogy(all_func_vals)
plt.show()
plt.close()
'''

# ----------------------------------------------
# this is GLD's invocation:
"""
# step.
termination = False
prev_evals = 0
while termination is False:
    # optimization step.
    solution, func_value, termination = stp1.step()
    # print('step')
    print('current value: ', func_value[-1])
# print the solution.
print('\n')
print('solution: ', solution)
# plot the decreasing function.
plt.plot(func_value)
"""


