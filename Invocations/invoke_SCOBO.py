#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 13:49:07 2021

@author: danielmckenzie
"""

from Algorithms.scobo_optimizer import SCOBOoptimizer
import numpy as np
from ExampleCode.oracle import Oracle
import matplotlib.pyplot as plt
from ExampleCode.benchmarkfunctions import SparseQuadratic, MaxK

# Defining the function
n_def = 200
s_exact = 20
noise_amp = 0.001
func = SparseQuadratic(n_def, s_exact, noise_amp)
query_budget = int(1e5)
m = 100  # Should always be larger than s_exact
#x0 = 100*np.random.randn(n_def)
x0 = np.random.randn(n_def)
step_size = 0.01
r = 0.1
# max_iters = int(2e2)
# max_iters = 1000

# Define the comparison oracle
oracle = Oracle(func)

Opt = SCOBOoptimizer(oracle, step_size, query_budget, x0, r, m, s_exact, objfunc=func)
# step.
termination = False
prev_evals = 0
while termination is False:
    # optimization step.
    solution, func_value, termination, queries = Opt.step()
    # print('step')
    print('current value at ' + str(prev_evals) + ': ', func_value[-1])
    prev_evals += 1
# print the solution.
print('\n')
print('solution: ', solution)
# plot the decreasing function.
plt.plot(func_value)
plt.show()
plt.close()
plt.semilogy(func_value)
plt.show()
plt.close()

print('number of values: ', len(func_value))
print('number of oracle queries: ', queries)





'''
for i in range(max_iters):
    print(i)
    err, func_values, termination, queries = Opt.step()
    print(err)


plt.semilogy(func_values)
plt.show()
plt.close()
plt.plot(func_values)
plt.show()
plt.close()
'''
