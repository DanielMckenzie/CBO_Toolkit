"""
Isha Slavin.
Week 3 - TASK #1.
"""

from ExampleCode.base import BaseOptimizer
from ExampleCode.oracle import Oracle
import numpy as np
import pandas as pd
import math
import random
from ExampleCode.benchmarkfunctions import SparseQuadratic, MaxK
from matplotlib import pyplot as plt

from ExampleCode.utils import random_sampling_directions, multiple_comparisons_oracle_2

# this class implements the Gradient-Less Descent with Binary Search Algorithm using a Comparison Oracle.
# source: https://arxiv.org/pdf/1911.06317.pdf.


class GLDOptimizer(BaseOptimizer):
    """
    INPUTS:
        1. defined_func (type = FUNC) objective function; inputted into Oracle class for function evaluations.
        2. x_0: (type = NUMPY ARRAY) starting point (of dimension = n).
        3. R: (type = INT) maximum search radius (ex.: 10).
        4. r: (type = INT) minimum search radius (ex.: 0.1).
        5. function_budget: (type = INT) total number of function evaluations allowed.
    """

    def __init__(self, oracle, defined_func, x_0, R, r, function_budget):
        super().__init__()

        self.function_evals = 0
        self.defined_func = defined_func
        self.x_0 = x_0
        self.R = R
        self.r = r
        self.function_budget = function_budget
        '''
        self.f_vals = []
        '''

        ## DM Code suggestion
        self.f_vals = [defined_func(x_0)]
        self.list_of_xt = []
        # must be a comparison oracle.
        self.oracle = oracle

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
        # sampling distribution (randomly generated each step). This follows GAUSSIAN, divided by n instead of sqrt(n).
        '''
        D = np.random.randn(n) / n
        '''
        # *********
        # call the UTILS function.
        # *********
        # iterate through k's in K (which equals log(R/r)).
        for k in range(int(self.K)):
            # calculate r_k.
            r_k = 2 ** -k
            r_k = r_k * self.R
            # print('r_k: ', r_k)
            output = random_sampling_directions(1, n, 'gaussian')
            D = output / n
            v_k = np.dot(r_k, D)
            # sample v_k from r_k_D.
            # random_dir = random.randint(0, n - 1)
            # v_k = r_k_D[random_dir]
            # print('vk: ', v_k)
            next_el = x_t + v_k
            # add each x_t + v_k to a list for all k in K.
            v_list.append(next_el)
        # length will never be 0 (I think), this is just to make sure.
        if len(v_list) == 0:
            # list_of_xt.append(x_t)
            # f_vals.append(defined_func(x_t))
            # continue
            """ fix this to return the proper output. """
            return 0

        # I have replaced the following code with UTIL #2 invocation.
        '''
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
        '''

        # *********
        # call the UTILS function.
        if len(v_list) > 1:
            # argmin, function_evaluations = multiple_comparisons_oracle(v_list, self.defined_func)
            argmin, function_evaluations = multiple_comparisons_oracle_2(v_list, self.oracle)
            self.function_evals += function_evaluations
            v_list = argmin
        # *********

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
            while len(self.f_vals) > (self.function_budget/2):
                self.f_vals.pop()
            return x_t, self.f_vals, 'B'
        # return solution, list of all function values, termination (which will be False here).
        return x_t, self.f_vals, False


'''
    # in the STEP function, use what Daniel did.
    # when function evals is still 0, take xk from self.x_0.
    # if not, do what you usually do to generate the xt and then append it to the list of xt's.
'''


"""
# ---------
print('sample invoke.')
# GLD - FUNCTION sample invocation.
n_def = 20000  # problem dimension.
s_exact = 200  # True sparsity.
noise_amp = 0.001  # noise amplitude.
# initialize objective function.
obj_func_1 = SparseQuadratic(n_def, s_exact, noise_amp)
obj_func_2 = MaxK(n_def, s_exact, noise_amp)
max_function_evals = 10000
x_0_ = np.random.rand(n_def)
print('shape of x_0_: ', len(x_0_))
R_ = 10
r_ = .01
# GLDOptimizer instance.
# def __init__(self, defined_func, x_0, R, r, function_budget).
stp1 = GLDOptimizer(obj_func_1, x_0_, R_, r_, max_function_evals)
# stp1 = GLDOptimizer(obj_func_2, x_0_, R_, r_, max_function_evals)
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
# ---------
print('\n')
print('number of function vals: ', len(func_value))
"""
