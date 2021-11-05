# isha slavin.
# Week 2 - TASK 3.
# an attempt to implement the Stochastic 3 Point code as an optimizer in optimizer.py.

import numpy as np
from ExampleCode.base import BaseOptimizer
from ExampleCode.oracle import Oracle
from ExampleCode.benchmarkfunctions import SparseQuadratic, MaxK
import pandas as pd
import random
import matplotlib.pyplot as plt
from numpy import linalg as LA
from ExampleCode.utils import random_sampling_directions, multiple_comparisons_oracle_2


class STPOptimizer(BaseOptimizer):
    """
    INPUTS:
        1. direction_vector_type: (type = INT) method to calculate direction vectors at each step.
            INPUT = 0: original (randomly choose element from 0 to n-1, have sparse n-dim vec. with 1 at random index).
            INPUT = 1: gaussian (randomly chosen elements of vector size n).
            INPUT = 2: uniform from sphere (randomly chosen elements of vector size n... normalized).
            INPUT = 3: rademacher (vector of length n with elements -1 or 1, 50% chance of each).
        2. n: (type = INT) dimension of vectors.
        3. a_k: (type = FLOAT) step_size; used in generating x+ and x-. (ex.: a_k = .1.)
        4. defined_func: (type = FUNC) objective function; inputted into Oracle class for function evaluations.
        5. function_budget: (type = INT) total number of function evaluations allowed.
    """

    def __init__(self, oracle, direction_vector_type, x_0, n, a_k, defined_func, function_budget):
        super().__init__()

        self.function_evals = 0
        self.direction_vector_type = direction_vector_type
        self.n = n
        self.a_k = a_k
        self.defined_func = defined_func
        self.function_budget = function_budget

        self.list_of_sk = []
        self.x_0 = x_0
        # print('x0: ', self.x_0)
        self.f_vals = [defined_func(x_0)]
        # must be a comparison oracle.
        self.oracle = oracle

    def step(self):
        n = self.n
        a_k = self.a_k
        # step of optimizer.
        # ---------
        # 1. generate an initial x_0.
        if self.function_evals == 0:
            ### DM: Rewrote in a slightly more efficient way
            '''
            x_k = np.random.randn(n)
            '''
            x_k = self.x_0
            # x_k = np.random.rand(1, n)
            # x_k = x_k[0]
            # print('xk:')
            # print(x_k)
            self.list_of_sk.append(x_k)
        else:
            x_k = self.list_of_sk[-1]
        # append the function value to list f_vals to track trend.
        self.f_vals.append(self.defined_func(x_k))
        # ---------
        # 2. generate a direction vector s_k.
        if self.direction_vector_type == 0:
            # original.
            '''
            random_direction = random.randint(0, n - 1)
            s_k = np.zeros(n, int)
            s_k[random_direction] = 1
            '''
            # print('sk:')
            # print(s_k)
            # *********
            # call the UTILS function.
            output = random_sampling_directions(1, n, 'original')
            s_k = output
            # *********
        elif self.direction_vector_type == 1:
            # gaussian.
            '''
            s_k = np.random.randn(n)/np.sqrt(n)
            '''
            # *********
            # call the UTILS function.
            output = random_sampling_directions(1, n, 'gaussian')
            s_k = output / np.sqrt(n)
            # *********
        elif self.direction_vector_type == 2:
            # uniform from sphere.
            '''
            s_k = np.random.randn(n)
            '''
            # formula: ||x_n|| = sqrt(x_n_1^2 + x_n_2^2 + ... + x_n_n^2).
            # let's calculate ||s_k||.

            ### DM: Easier:
            '''
            s_k_norm = np.linalg.norm(s_k)
            '''

            ### This is a nice implementation of finding the norm though!
            # sum = 0
            # for elem in s_k:
            #    elem_squared = elem * elem
            #    sum += elem_squared
            # sum_sqrt = sum ** 0.5
            # s_k_norm = sum_sqrt
            # print('s_k norm: ', s_k_norm)
            '''
            s_k = s_k / s_k_norm
            '''
            # *********
            # call the UTILS function.
            output = random_sampling_directions(1, n, 'uniform from sphere')
            s_k = output
            # *********
        elif self.direction_vector_type == 3:
            # rademacher.
            '''
            s_k = []
            count_positive1 = 0
            count_negative1 = 0
            ### DM: Easier:
            s_k = 2*np.round(np.random.rand(n))-1
            s_k = s_k/np.sqrt(n)
            '''

            ### It's interesting to think about why the above line of
            ### code indeed produces a Rademacher vector.

            #            for i in range(n):
            #                rand_choice = random.choice([-1, 1])
            #
            #                if rand_choice == 1:
            #                    count_positive1 += 1
            #                else:
            #                    count_negative1 += 1
            #                # print(str(i) + ': ', rand_choice)
            #                s_k.append(rand_choice)
            #            # print('type sk: ', type(s_k))
            # *********
            # call the UTILS function.
            output = random_sampling_directions(1, n, 'rademacher')
            # print('RADEMACHER output: ', output)
            s_k = output
            # *********
        else:
            print('invalid direction vector type. please input an integer, from 0 to 3.')
            ### Something I've been experimenting with lately is using the Python
            ### built in Error class
            # raise ValueError('Vector type must be an integer from 0 to 3')
            ### But this will terminate the function, so it may not be what we want.
            return 0
        # ---------
        # 3. generate x+, x-.
        # generate x+.
        x_plus = x_k + np.dot(a_k, s_k)
        # x_plus = x_k + a_k * s_k
        # generate x-.
        x_minus = x_k - np.dot(a_k, s_k)
        # x_minus = x_k - a_k * s_k
        # ---------
        # 4. compute function evaluations.

        # I have replaced the following code with UTIL #2 invocation.
        '''
        # call the Oracle class, inputting our function.
        new_instance_1 = Oracle(self.defined_func)
        # complete the first evaluation.
        first_comparison = new_instance_1(x_k, x_plus)
        self.function_evals += 1
        if first_comparison == -1:
            # x_plus is smaller.
            # complete the second evaluation.
            second_comparison = new_instance_1(x_plus, x_minus)
            self.function_evals += 1
            if second_comparison == -1:
                # x_minus is smaller.
                argmin = x_minus
                x_k = argmin
            elif second_comparison == +1:
                # x_plus is smaller.
                argmin = x_plus
                x_k = argmin
        elif first_comparison == +1:
            # x_k is smaller.
            # complete the second evaluation.
            second_comparison = new_instance_1(x_k, x_minus)
            self.function_evals += 1
            if second_comparison == -1:
                # x_minus is smaller.
                argmin = x_minus
                x_k = argmin
            elif second_comparison == +1:
                # x_k is smaller.
                argmin = x_k
                x_k = argmin
        '''

        v_list = [x_k, x_plus, x_minus]
        # *********
        # call the UTILS function.
        if len(v_list) > 1:
            # argmin, function_evaluations = multiple_comparisons_oracle(v_list, self.defined_func)
            argmin, function_evaluations = multiple_comparisons_oracle_2(v_list, self.oracle)
            self.function_evals += function_evaluations
            # v_list = argmin
            x_k = argmin[0]
        # *********

        self.list_of_sk.append(x_k)
        # we incremented function evaluations (self.function_evals).
        # now we will see if we have hit the eval limit.
        # ---------
        # 5. check if function budget is depleted.
        if self.reachedFunctionBudget(self.function_budget, self.function_evals):
            # if budget is reached return parent.
            # solution, list of all function values, termination.
            while len(self.f_vals) > (self.function_budget / 2):
                self.f_vals.pop()
            return x_k, self.f_vals, 'B'
        # return solution, list of all function values, termination (which will be False here).
        return x_k, self.f_vals, False


"""
# ---------
print('sample invoke.')
# sample invocation.
n = 20000  # problem dimension.
s_exact = 200  # true sparsity.
noiseamp = 0.001  # noise amplitude.
# initialize objective function.
#obj_func = MaxK(n, s_exact, noiseamp)
obj_func = SparseQuadratic(n, s_exact, noiseamp)
# create an instance of STPOptimizer.
# direction_vector_type = 0  # original.
# direction_vector_type = 1  # gaussian.
# direction_vector_type = 2  # uniform from sphere.
direction_vector_type = 3  # rademacher.
a_k = 0.1  # step-size.
function_budget = 10000
# stp instance.
stp1 = STPOptimizer(direction_vector_type, n, a_k, obj_func, function_budget)
# step.
termination = False
prev_evals = 0
while termination is False:
    # optimization step.
    solution, func_value, termination = stp1.step()
    # print('step')
    print('current value: ', func_value[-1])
# plot the decreasing function.
plt.plot(func_value)
plt.show()
# log x-axis.
plt.semilogy(func_value)
plt.show()
# ---------
print('\n')
print('number of function vals: ', len(func_value))
"""
# ---------
# OBSERVATIONS:
# Works very well with direction_vector_types = 0 (original) & 2 (uniform from sphere).
# For some reason, does not work with direction_vector_types = 1 (gaussian) & 3 (rademacher).
# This is strange, as it works in oracle.py. So I must've coded something wrong, maybe.
# * Talk about this more with Daniel on Wednesday. *
# ---------

# Reference.
########################################################################################################################
# '''
# This file contains the ZO-BCD algorithm, as an instance of the BaseOptimizer
# class (see base.py). This class is based on code originally available at
# https://github.com/NiMlr/High-Dim-ES-RL
# and used under license.
#
# Reference: A Zeroth Order Block Coordinate Descent Algorithm for
#     Black-Box Optimization" by Cai, Lou, McKenzie and Yin.
#
# '''
#
#
# import numpy as np
# from multiprocessing.dummy import Pool
# from base import BaseOptimizer
# from scipy.linalg import circulant
# from Cosamp import cosamp
#
#
# class ZOBCD(BaseOptimizer):
#     ''' ZOBCD for black box optimization. A sparsity-aware, block coordinate
#     descent method.
#
#     INPUTS:
#         y0 ................. initial iterate
#         step_size .......... step size
#         f .................. the objective function
#         params ............. A dict containing additional parameters, e.g. the
#         number of blocks (see Example.py)
#         function_budget .... total number of function evaluations allowed.
#         shuffle ............ If true, we choose a new random assignment of
#         variables to blocks every (number_of_blocks) iterations.
#         function_target .... If not none, this specifies the desired optimality
#         gap
#
#     March 23rd 2021
#
#     '''
#
#     def __init__(self, x0, step_size, f, params, function_budget=10000 ,shuffle=True,
#                  function_target=None):
#
#         super().__init__()
#
#         self.function_evals = 0
#         self.function_budget = function_budget
#         self.function_target = function_target
#         self.f = f
#         self.x = x0
#         self.n = len(x0)
#         self.t = 0
#         self.Type = params["Type"]
#         self.sparsity = params["sparsity"]
#         self.delta = params["delta"]
#         self.step_size = step_size
#         self.shuffle = shuffle
#         self.permutation = np.random.permutation(self.n)
#
#         # block stuff
#         oversampling_param = 1.1
#         self.J = params["J"]
#         self.block_size = int(np.ceil(self. n /self.J))
#         self.sparsity = int(np.ceil(oversampling_para m *self.sparsit y /self.J))
#         print(self.sparsity)
#         self.samples_per_block = int(np.ceil(oversampling_para m *self.sparsit y *np.log(self.block_size)))
#
#         # Define cosamp_params
#         if self.Type == "ZOBCD-R":
#             Z = 2* (np.random.rand(self.samples_per_block, self.block_size) > 0.5) - 1
#         elif self.Type == "ZOBCD-RC":
#             z1 = 2 * (np.random.rand(1, self.block_size) > 0.5) - 1
#             Z1 = circulant(z1)
#             SSet = np.random.choice(self.block_size, self.samples_per_block, replace=False)
#             Z = Z1[SSet, :]
#         else:
#             raise Exception("Need to choose a type, either ZOBCD-R or ZOBCD-RC")
#
#         cosamp_params = {"Z": Z, "delta": self.delta, "maxiterations": 10,
#                          "tol": 0.5, "sparsity": self.sparsity, "block": []}
#         self.cosamp_params = cosamp_params
#
#     def CosampGradEstimate(self):
#         # Gradient estimation
#
#         maxiterations = self.cosamp_params["maxiterations"]
#         Z = self.cosamp_params["Z"]
#         delta = self.cosamp_params["delta"]
#         sparsity = self.cosamp_params["sparsity"]
#         tol = self.cosamp_params["tol"]
#         block = self.cosamp_params["block"]
#         num_samples = np.size(Z, 0)
#         x = self.x
#         f = self.f
#         dim = len(x)
#
#         Z_padded = np.zeros((num_samples, dim))
#         Z_padded[:, block] = Z
#
#         y = np.zeros(num_samples)
#         print(num_samples)
#         function_estimate = 0
#
#         for i in range(num_samples):
#             y_temp = f(x + delta * np.transpose(Z_padded[i, :]))
#             y_temp2 = f(x)
#             function_estimate += y_temp2
#             y[i] = (y_temp - y_temp2) / (np.sqrt(num_samples) * delta)
#             self.function_evals += 2
#
#         Z = Z / np.sqrt(num_samples)
#         block_grad_estimate = cosamp(Z, y, sparsity, tol, maxiterations)
#         grad_estimate = np.zeros(dim)
#         grad_estimate[block] = block_grad_estimate
#         function_estimate = function_estimate / num_samples
#
#         return grad_estimate, function_estimate
#
#     def step(self):
#         # Take step of optimizer
#
#         if self.t % self.J == 0 and self.shuffle:
#             self.permutation = np.random.permutation(self.n)
#             print('Reshuffled!')
#
#         coord_index = np.random.randint(self.J)
#         block = np.arange((coord_index - 1) * self.block_size, min(coord_index * self.block_size, self.n))
#         block = self.permutation[block]
#         self.cosamp_params["block"] = block
#         grad_est, f_est = self.CosampGradEstimate()
#         self.f_est = f_est
#         self.x += -self.step_size * grad_est
#
#         if self.reachedFunctionBudget(self.function_budget, self.function_evals):
#             # if budget is reached return parent
#             return self.function_evals, self.x, 'B'
#
#         if self.function_target != None:
#             if self.reachedFunctionTarget(self.function_target, f_est):
#                 # if function target is reach return population expected value
#                 return self.function_evals, self.x, 'T'
#
#         self.t += 1
#
#         return self.function_evals, False, False
