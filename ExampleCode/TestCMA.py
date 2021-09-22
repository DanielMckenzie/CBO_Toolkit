#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 12:58:51 2021

@author: danielmckenzie
"""


from CMA_2 import CMA
import numpy as np
from oracle import Oracle
import matplotlib.pyplot as plt
from benchmarkfunctions import SparseQuadratic, MaxK

# Defining the function
n_def = 100
s_exact = 20
noise_amp = 0.0
func = SparseQuadratic(n_def, s_exact, noise_amp)
query_budget = int(1e5)
m = 100
x0 = 100*np.random.randn(n_def)
step_size = 0.2
r = 0.1
max_iters = int(1000)
lam = 10
mu = 5
sigma = 0.5

# Define the comparison oracle
oracle = Oracle(func)

all_func_vals = []

Opt = CMA(oracle, query_budget, x0, lam, mu, sigma, function=func)

for i in range(max_iters):
    # print(i)
    val = Opt.step()
    print(val)
    all_func_vals.append(val)
    
plt.semilogy(all_func_vals)
plt.show()
    
