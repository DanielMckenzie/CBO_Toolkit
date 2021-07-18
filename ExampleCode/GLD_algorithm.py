# isha slavin.
# week 2 - task 4.
# this code is currently not running;
# it is my work so far on implementing a comparison - based version of gradient-less descent algorithm.
# talk to Daniel about this on Wednesday.

from base import BaseOptimizer
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

# OLD VERSION (without fixes - saving to make sure I don't lose my work)....
"""
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
print('x0: ')
print(x_0_)
print('shape of x_0_: ', len(x_0_))

# Gaussian.
D_ = np.random.randn(n_def)
print('shape of sampling distribution D: ', len(D_))
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
"""


# NEW VERSION (with fixes in sampling distribution; still not implemented as a class)....

def GLD_algorithm(defined_func, max_iter, x_0, R, r):
    # n = len(D)
    n = len(x_0)
    # D is sampling distribution (of dimension n, I believe).
    # we will define it at each iteration.
    K = math.log(R / r, 10)
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
        # D = np.random.randn(n)/np.sqrt(n)
        D = np.random.randn(n) / n
        # print(D)
        # in the range of K (max. / min. search radius).
        for k in range(int(K)):
            # calculate r_k.
            r_k = 2 ** -k
            r_k = r_k * R
            # print('r_k: ', r_k)
            r_k_D = np.dot(r_k, D)
            # sample v_k from r_k_D.
            random_dir = random.randint(0, n - 1)
            v_k = r_k_D[random_dir]
            # print('vk: ', v_k)
            next_el = x_t + v_k
            # add each x_t + v_k to a list for all k in K.
            v_list.append(next_el)
        # length will never be 0 (I think), this is just to make sure.
        if len(v_list) == 0:
            list_of_xt.append(x_t)
            f_vals.append(defined_func(x_t))
            continue
        # now that we have our list of vk's, let's use the comparison oracle to determine the argmin of the elements.
        # while there are at least two elements to input into the comparison Oracle.
        while len(v_list) >= 2:
            new_instance_1 = Oracle(defined_func)
            # print('0:', v_list[0])
            # print('1:', v_list[1])
            # input the first two elements of the list into the oracle.
            first_comparison = new_instance_1(v_list[0], v_list[1])
            # INCREMENT function_evals by 1.
            if first_comparison == +1:
                # 0th elem is smaller.
                # remove 1st element.
                v_list.pop(1)
            elif first_comparison == -1:
                # 1st elem is smaller.
                # remove 0th element.
                v_list.pop(0)
            else:
                # function values are equal with elements 0 and 1 of list.
                # choose one at random to drop.
                rand_choice = random.choice([0, 1])
                v_list.pop(rand_choice)
        list_length = len(v_list)
        # the list is length 1 after all comparisons have been made (or if input R = input r).
        if list_length == 1:
            # print(t)
            # remaining element is our ARGMIN.
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
print('x0: ')
print(x_0_)
print('shape of x_0_: ', len(x_0_))

# Gaussian.
# I want to define this at each iteration, I don't want to give it as a parameter of the function.
# D_ = np.random.randn(n_def)
# print('shape of sampling distribution D: ', len(D_))
R_ = 10
r_ = .01
gld_1 = GLD_algorithm(obj_func_1, max_iterations, x_0_, R_, r_)
# gld_1 = GLD_algorithm(obj_func_2, max_iterations, x_0_, D_, R_, r_)

# print(gld_1)
answer = gld_1[0]
print('solution: ', answer)
func_vals = gld_1[1]
# print(func_vals)

# print(func_vals[0])
# print(func_vals[1])
# print(func_vals[2])

plt.semilogy(func_vals[:10])
plt.show()
plt.semilogy(func_vals[:100])
plt.show()
plt.semilogy(func_vals)
plt.show()


# MOVING the CLASS implementation of this algorithm to another file called gld_optimizer.py.
# -----------------------------------

# Now I will try to complete TASK #1 of WEEK 3 tasks.
# an attempt of TASK #1: implement the Gradient-less descent algorithm (GLD) as a class, like I did for STP.
# NEWEST VERSION (implemented as a class)....

"""
class GLDOptimizer(BaseOptimizer):
    
    # INPUTS:
    #     1. defined_func (type = FUNC) objective function; inputted into Oracle class for function evaluations.
    #     2. x_0: (type = NUMPY ARRAY) starting point (of dimension = n).
    #     3. R: (type = INT) maximum search radius (ex.: 10).
    #     4. r: (type = INT) minimum search radius (ex.: 0.1).
    #     5. function_budget: (type = INT) total number of function evaluations allowed.
    

    def __init__(self, defined_func, x_0, R, r, function_budget):
        super().__init__()

        self.function_evals = 0
        self.defined_func = defined_func
        self.x_0 = x_0
        self.R = R
        self.r = r
        self.function_budget = function_budget
        self.f_vals = []
        self.list_of_xt = []

        K = math.log(self.R / self.r, 10)
        self.K = K

    def step(self):
        # x_t.
        if self.function_evals == 0:
            ### DM: Rewrote in a slightly more efficient way
            x_t = self.x_0
            # x_k = np.random.rand(1, n)
            # x_k = x_k[0]
            # print('xk:')
            # print(x_k)
            self.list_of_xt.append(x_t)
        else:
            x_t = self.list_of_xt[-1]
        # list of x_t's for this one step.
        v_list = [x_t]
        # n: dimension of x_t.
        n = len(x_t)
        # sampling distribution (randomly generated at each step).
        D = np.random.randn(n) / n
        # iterate through k's in K (which equals log(R/r)).
        for k in range(int(self.K)):
            # calculate r_k.
            r_k = 2 ** -k
            r_k = r_k * self.R
            # print('r_k: ', r_k)
            r_k_D = np.dot(r_k, D)
            # sample v_k from r_k_D.
            random_dir = random.randint(0, n - 1)
            v_k = r_k_D[random_dir]
            # print('vk: ', v_k)
            next_el = x_t + v_k
            # add each x_t + v_k to a list for all k in K.
            v_list.append(next_el)
        # length will never be 0 (I think), this is just to make sure.
        if len(v_list) == 0:
            list_of_xt.append(x_t)
            f_vals.append(defined_func(x_t))
            # continue
            return 0
        # now that we have our list of vk's, let's use the comparison oracle to determine the argmin of the elements.
        # while there are at least two elements to input into the comparison Oracle.
        while len(v_list) >= 2:
            new_instance_1 = Oracle(self.defined_func)
            # print('0:', v_list[0])
            # print('1:', v_list[1])
            # input the first two elements of the list into the oracle.
            first_comparison = new_instance_1(v_list[0], v_list[1])
            # INCREMENT function_evals by 1.
            self.function_evals += 1
            # possibilities of Oracle output:
            if first_comparison == +1:
                # 0th elem is smaller.
                # remove 1st element.
                v_list.pop(1)
            elif first_comparison == -1:
                # 1st elem is smaller.
                # remove 0th element.
                v_list.pop(0)
            else:
                # function values are equal with elements 0 and 1 of list.
                # choose one at random to drop.
                rand_choice = random.choice([0, 1])
                v_list.pop(rand_choice)
        list_length = len(v_list)
        # the list is length 1 after all comparisons have been made (or if input R = input r).
        if list_length == 1:
            # print(t)
            # remaining element is our ARGMIN.
            argmin = v_list[0]
            x_t = argmin
            self.list_of_xt.append(x_t)
            self.f_vals.append(self.defined_func(x_t))
        # now, let's check if the function budget is depleted.
        if self.reachedFunctionBudget(self.function_budget, self.function_evals):
            # if budget is reached return parent.
            # solution, list of all function values, termination.
            return x_t, self.f_vals, 'B'
        # return solution, list of all function values, termination (which will be False here).
        return x_t, self.f_vals, False


'''
    # in the STEP function, use what Daniel did.
    # when function evals is still 0, take xk from self.x_0.
    # if not, do what you usually do to generate the xt and then append it to the list of xt's.
'''

# ---------
print('sample invoke.')
# GLD - FUNCTION sample invocation.
n_def = 20000  # problem dimension.
s_exact = 200  # True sparsity.
noise_amp = 0.001  # noise amplitude.
# initialize objective function.
obj_func_1 = SparseQuadratic(n_def, s_exact, noise_amp)
# obj_func_2 = MaxK(n_def, s_exact, noise_amp)
max_function_evals = 10000
x_0_ = np.random.rand(n_def)
print('shape of x_0_: ', len(x_0_))
R_ = 10
r_ = .01
# GLDOptimizer instance.
# def __init__(self, defined_func, x_0, R, r, function_budget).
stp1 = GLDOptimizer(obj_func_1, x_0_, R_, r_, max_function_evals)
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
plt.show()
# log x-axis.
plt.semilogy(func_value)
plt.show()
"""

# **********************************************************************************************************************
# reference:


# class STPOptimizer(BaseOptimizer):
#     """
#     INPUTS:
#         1. direction_vector_type: (type = INT) method to calculate direction vectors at each step.
#             INPUT = 0: original (randomly choose element from 0 to n-1, have sparse n-dim vec. with 1 at random index).
#             INPUT = 1: gaussian (randomly chosen elements of vector size n).
#             INPUT = 2: uniform from sphere (randomly chosen elements of vector size n... normalized).
#             INPUT = 3: rademacher (vector of length n with elements -1 or 1, 50% chance of each).
#         2. n: (type = INT) dimension of vectors.
#         3. a_k: (type = FLOAT) step_size; used in generating x+ and x-. (ex.: a_k = .1.)
#         4. defined_func: (type = FUNC) objective function; inputted into Oracle class for function evaluations.
#         5. function_budget: (type = INT) total number of function evaluations allowed.
#     """
#
#     def __init__(self, direction_vector_type, n, a_k, defined_func, function_budget):
#         super().__init__()
#
#         self.function_evals = 0
#         self.direction_vector_type = direction_vector_type
#         self.n = n
#         self.a_k = a_k
#         self.defined_func = defined_func
#         self.function_budget = function_budget
#         self.f_vals = []
#         self.list_of_sk = []
#
#     def step(self):
#         # step of optimizer.
#         # ---------
#         # 1. generate an initial x_0.
#         if self.function_evals == 0:
#             ### DM: Rewrote in a slightly more efficient way
#             x_k = np.random.randn(n)
#             # x_k = np.random.rand(1, n)
#             # x_k = x_k[0]
#             # print('xk:')
#             # print(x_k)
#             self.list_of_sk.append(x_k)
#         else:
#             x_k = self.list_of_sk[-1]
#         # append the function value to list f_vals to track trend.
#         self.f_vals.append(self.defined_func(x_k))
#         # ---------
#         # 2. generate a direction vector s_k.
#         if self.direction_vector_type == 0:
#             # original.
#             random_direction = random.randint(0, n - 1)
#             s_k = np.zeros(n, int)
#             s_k[random_direction] = 1
#             # print('sk:')
#             # print(s_k)
#         elif self.direction_vector_type == 1:
#             # gaussian.
#             s_k = np.random.randn(n) / np.sqrt(n)
#         elif self.direction_vector_type == 2:
#             # uniform from sphere.
#             s_k = np.random.randn(n)
#             # formula: ||x_n|| = sqrt(x_n_1^2 + x_n_2^2 + ... + x_n_n^2).
#             # let's calculate ||s_k||.
#
#             ### DM: Easier:
#             s_k_norm = np.linalg.norm(s_k)
#
#             ### This is a nice implementation of finding the norm though!
#             # sum = 0
#             # for elem in s_k:
#             #    elem_squared = elem * elem
#             #    sum += elem_squared
#             # sum_sqrt = sum ** 0.5
#             # s_k_norm = sum_sqrt
#             # print('s_k norm: ', s_k_norm)
#             s_k = s_k / s_k_norm
#         elif self.direction_vector_type == 3:
#             # rademacher.
#             s_k = []
#             count_positive1 = 0
#             count_negative1 = 0
#             ### DM: Easier:
#             s_k = 2 * np.round(np.random.rand(n)) - 1
#             s_k = s_k / np.sqrt(n)
#
#             ### It's interesting to think about why the above line of
#             ### code indeed produces a Rademacher vector.
#
#         #            for i in range(n):
#         #                rand_choice = random.choice([-1, 1])
#         #
#         #                if rand_choice == 1:
#         #                    count_positive1 += 1
#         #                else:
#         #                    count_negative1 += 1
#         #                # print(str(i) + ': ', rand_choice)
#         #                s_k.append(rand_choice)
#         #            # print('type sk: ', type(s_k))
#         else:
#             print('invalid direction vector type. please input an integer, from 0 to 3.')
#             ### Something I've been experimenting with lately is using the Python
#             ### built in Error class
#             # raise ValueError('Vector type must be an integer from 0 to 3')
#             ### But this will terminate the function, so it may not be what we want.
#             return 0
#         # ---------
#         # 3. generate x+, x-.
#         # generate x+.
#         x_plus = x_k + np.dot(a_k, s_k)
#         # x_plus = x_k + a_k * s_k
#         # generate x-.
#         x_minus = x_k - np.dot(a_k, s_k)
#         # x_minus = x_k - a_k * s_k
#         # ---------
#         # 4. compute function evaluations.
#         # call the Oracle class, inputting our function.
#         new_instance_1 = Oracle(self.defined_func)
#         # complete the first evaluation.
#         first_comparison = new_instance_1(x_k, x_plus)
#         self.function_evals += 1
#         if first_comparison == -1:
#             # x_plus is smaller.
#             # complete the second evaluation.
#             second_comparison = new_instance_1(x_plus, x_minus)
#             self.function_evals += 1
#             if second_comparison == -1:
#                 # x_minus is smaller.
#                 argmin = x_minus
#                 x_k = argmin
#             elif second_comparison == +1:
#                 # x_plus is smaller.
#                 argmin = x_plus
#                 x_k = argmin
#         elif first_comparison == +1:
#             # x_k is smaller.
#             # complete the second evaluation.
#             second_comparison = new_instance_1(x_k, x_minus)
#             self.function_evals += 1
#             if second_comparison == -1:
#                 # x_minus is smaller.
#                 argmin = x_minus
#                 x_k = argmin
#             elif second_comparison == +1:
#                 # x_k is smaller.
#                 argmin = x_k
#                 x_k = argmin
#         self.list_of_sk.append(x_k)
#         # we incremented function evaluations (self.function_evals).
#         # now we will see if we have hit the eval limit.
#         # ---------
#         # 5. check if function budget is depleted.
#         if self.reachedFunctionBudget(self.function_budget, self.function_evals):
#             # if budget is reached return parent.
#             # solution, list of all function values, termination.
#             return x_k, self.f_vals, 'B'
#         # return solution, list of all function values, termination (which will be False here).
#         return x_k, self.f_vals, False
#
#
#
# # ---------
# print('sample invoke.')
# # sample invocation.
# n = 20000  # problem dimension.
# s_exact = 200  # true sparsity.
# noiseamp = 0.001  # noise amplitude.
# # initialize objective function.
# # obj_func = MaxK(n, s_exact, noiseamp)
# obj_func = SparseQuadratic(n, s_exact, noiseamp)
# # create an instance of STPOptimizer.
# # direction_vector_type = 0  # original.
# # direction_vector_type = 1  # gaussian.
# # direction_vector_type = 2  # uniform from sphere.
# direction_vector_type = 3  # rademacher.
# a_k = 0.5  # step-size.
# function_budget = 10000
# # stp instance.
# stp1 = STPOptimizer(direction_vector_type, n, a_k, obj_func, function_budget)
# # step.
# termination = False
# prev_evals = 0
# while termination is False:
#     # optimization step.
#     solution, func_value, termination = stp1.step()
#     # print('step')
#     print('current value: ', func_value[-1])
# # plot the decreasing function.
# plt.plot(func_value)
# plt.show()
# # log x-axis.
# plt.semilogy(func_value)
# plt.show()

