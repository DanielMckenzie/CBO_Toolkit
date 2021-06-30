# isha slavin.
# week 2 - task 4.
# this code is currently not running;
# it is my work so far on implementing a comparison - based version of gradient-less descent algorithm.
# talk to Daniel about this on Wednesday.

from oracle import Oracle
import numpy as np
import pandas as pd
import math
import random
from benchmarkfunctions import SparseQuadratic, MaxK
from matplotlib import pyplot as plt

# let's write a function called GLD_algorithm().
'''
INPUTS:
    - defined_func (objective function; inputted into Oracle).
    - max_iter (# of iterations).
    - x_0 (starting point).
    - D (sampling distribution).
    - R (max. search radius).
    - r (min. search radius).
'''


def GLD_algorithm(defined_func, max_iter, x_0, D, R, r):
    n = len(D)
    # D is sampling distribution (of dimension n, I believe).
    K = math.log(R/r, 10)
    K = math.ceil(K)
    print('K: ', K)
    list_of_xt = []
    f_vals = []
    x_t = x_0
    # x = x_0
    print(x_t)
    list_of_xt.append(x_t)
    f_vals.append(defined_func(x_t))
    for t in range(max_iter):
        # print(t)
        # x_t = x_0[t]
        v_list = [x_t]
        for k in range(int(K)):
            r_k = 2 ** -k
            r_k = r_k * R
            # print('r_k: ', r_k)
            r_k_D = np.dot(r_k, D)
            random_dir = random.randint(0, n - 1)
            v_k = r_k_D[random_dir]
            # print('vk: ', v_k)
            next_el = x_t + v_k
            v_list.append(next_el)
        if len(v_list) == 0:
            list_of_xt.append(x_t)
            f_vals.append(defined_func(x_t))
            continue
        while len(v_list) >= 2:
            new_instance_1 = Oracle(defined_func)
            # print('0:', v_list[0])
            # print('1:', v_list[1])
            first_comparison = new_instance_1(v_list[0], v_list[1])
            if first_comparison == +1:
                # 0th elem is smaller.
                v_list.pop(1)
            elif first_comparison == -1:
                # 1st elem is smaller.
                v_list.pop(0)
            else:
                # function values are equal with elements 0 and 1 of list.
                # choose one at random to drop.
                rand_choice = random.choice([0, 1])
                v_list.pop(rand_choice)
        list_length = len(v_list)
        if list_length == 1:
            # print(t)
            x_t = v_list[0]
            list_of_xt.append(x_t)
            f_vals.append(defined_func(x_t))
            continue
    return x_t, f_vals


# sample invocation.
print('\n')
print('*********')
# ---------
n_def = 20000  # problem dimension.
s_exact = 200  # True sparsity.
noise_amp = 0.001  # noise amplitude.
# ---------
# initialize objective functions.
obj_func_1 = SparseQuadratic(n_def, s_exact, noise_amp)
# obj_func_2 = MaxK(n_def, s_exact, noise_amp)
max_iterations = 10000
x_0_ = np.random.rand(n_def)
# Gaussian.
D_ = np.random.randn(n_def)
R_ = 10
r_ = .01
gld_1 = GLD_algorithm(obj_func_1, max_iterations, x_0_, D_, R_, r_)
# gld_1 = GLD_algorithm(obj_func_2, max_iterations, x_0_, D_, R_, r_)

# print(gld_1)
answer = gld_1[0]
print('solution: ', answer)
func_vals = gld_1[1]

plt.semilogy(func_vals)
plt.show()




