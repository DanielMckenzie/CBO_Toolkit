"""
Isha Slavin.
Week 3 - TASK #3.
"""

# current implementation of signOPT (in old_algorithms.py).

'''
CHEETAH is not working for me for some reason. :(
'''
# from cheetah.mujocomodel import MujocoModel
from base import BaseOptimizer
from oracle import Oracle_2
import numpy as np
import pandas as pd
import math
import random
from benchmarkfunctions import SparseQuadratic, MaxK
from matplotlib import pyplot as plt
import gurobipy as gp


class SignOPTOptimizer(BaseOptimizer):
    """
    INPUTS:
        1. model: (type = FUNC) hard-label model (defined_func).
        2. function_budget: (type = INT) total number of function evaluations allowed.
        3. x0: (type = NUMPY ARRAY) starting point.
        4. m: (type = INT) the number of randomly sampled distribution vectors (u1, ..., um).
        5. d: (type = INT) the dimension of each of these randomly sampled vectors (should be same dimension as x0).
            *** Get rid of this input, and set d:=len(x0).
        6. alpha: (type =INT) step-size.
        7. r: (type = INT) sampling radius.
        8. show_runs: (type = BOOL)
            *** Get rid of this parameter once you test out the difference between show_runs = True & show_runs = False.
            *** the goal here is to be able to run this with any function input, like MaxK or SparseQuadratic (i think).
    """

    def __init__(self, model, function_budget, x0, m, d, alpha, r, show_runs):
        super().__init__()

        self.function_evals = 0
        self.model = model
        self.function_budget = function_budget
        self.x0 = x0
        self.m = m
        self.d = d
        self.alpha = alpha
        self.r = r
        self.show_runs = show_runs

        self.f_vals = []
        self.list_of_x = []

        self.rewards = np.zeros((self.function_budget, 1))
        self.x = np.squeeze(self.x0)
        # matrix: # rows = # of randomly sampled vectors, # columns = dimension of each vector.
        self.Z = np.zeros((self.m, self.d))
        self.conv = None  # step number we exceed reward threshold at.

    def signOPT_grad_estimate(self, x_in, Z, r):
        """
        Function which estimates the gradient vector from m Comparison oracle queries using the SignOPT method.
        Outputs a value g_hat, which is an approximation to g/||g||.
        INPUTS:
            1. x_in: (type = NUMPY ARRAY) any given point in R^d.
            2. Z: (type = MATRIX) an m*d matrix with rows z_i uniformly sampled from unit sphere.
            3. r: (type = INT) sampling radius.
        """

        g_hat = np.zeros([self.d])
        print(type(g_hat))
        for i in range(self.m):
            new_instance_1 = Oracle_2(self.model)
            first_comparison = new_instance_1(x_in, x_in * r * Z[i, :])
            print(first_comparison)
            # increment the number of function evaluations by 1.
            self.function_evals += 1
            g_hat += first_comparison * Z[i, :]
        g_hat = g_hat / np.float(self.m)
        print(np.linalg.norm(g_hat))
        return g_hat

    def step(self):

        if self.function_evals == 0:
            ### DM: Rewrote in a slightly more efficient way
            x = self.x
            # x_k = np.random.rand(1, n)
            # x_k = x_k[0]
            # print('xk:')
            # print(x_k)
            self.list_of_x.append(x)
        else:
            x = self.list_of_x[-1]
        # randomly sample vectors u1, ..., um.
        Z = self.Z
        for j in range(0, self.m):
            temp = np.random.rand(1, self.d)
            Z[j, :] = temp / np.linalg.norm(temp) # normalize.
        g_hat = self.signOPT_grad_estimate(x, Z, self.r * 1.01 ** (-1 * self.function_evals))
        # x = x + self.alpha * 1.001 ** (-1 * (self.function_evals - 1)) * g_hat
        x = x - self.alpha*g_hat
        self.list_of_x.append(x)
        if show_runs:
            # self.f_vals.append(self.model.render(x))
            self.f_vals.append(self.model(x))
        else:
            self.f_vals.append(self.model(x))
        print('current rewards at step ' + str(self.function_evals) + ': ' + str(self.f_vals[-1]))
        print('step_size:', self.alpha * 1.001 ** (-1 * (self.function_evals - 1)))
        print('gradient norm:', np.linalg.norm(g_hat))
        #########
        '''
        conv = self.conv
        if bool(self.model.reward_threshold):
            if self.f_vals[-1] > self.model.reward_threshold and conv is None:  # if we converge early
                conv = int(self.function_evals-1) + 1
                print('************** Reward Threshold Exceeded **************')
        '''
        if self.reachedFunctionBudget(self.function_budget, self.function_evals):
            # if budget is reached return parent.
            # solution, list of all function values, termination.
            return x, self.f_vals, 'B'
        return x, self.f_vals, False


# sample invocation.
'''
CHEETAH isn't working for me for some reason, so I can't import MujocoModel.
Therefore I'm going to use a test function (but I'm not sure if I'm supposed to do that....)
'''
# m_1 = MujocoModel('Reacher-v2', 1000, -0.5, 0.5, reward_threshold=-8, msfilter='NoFilter')
# m_2 = MujocoModel('Reacher-v2', 1000, -0.5, 0.5, reward_threshold=-8, msfilter='MeanStdFilter')

n_def = 20000  # problem dimension.
s_exact = 200  # True sparsity.
noise_amp = 0.001  # noise amplitude.
# initialize objective function.
model_func_1 = SparseQuadratic(n_def, s_exact, noise_amp)
model_func_2 = MaxK(n_def, s_exact, noise_amp)
function_budget_ = 1000
m_ = 100
d_ = 20000
x_0_ = np.zeros((d_, 1))
step_size = .2
r_ = 0.1
show_runs = True

# trying something out.
# model = gp.Model("1BitRecovery")

stp1 = SignOPTOptimizer(model_func_1, function_budget_, x_0_, m_, d_, step_size, r_, show_runs)
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

# It seems to be doing the opposite of optimizing....
# I need to talk to Daniel about this. Since I don't iterate through t because of step size, I didn't add a variable
# for incrementing, so I depend on the value of self.function_evals().


'''
SCOBO.
'''
# ______________________________________________________________________________________________________________________
# Notes on the SCOBO algorithm (how to implement it in the class form we've been doing).
# IDEA: we have to create a 'step' function.
#   Everything under the main for-loop of the SCOBO_mujoco() function goes in the step function.
# We still have the other functions (cleaned up), and then variables for the function values and number of function
# calls so we know when to terminate.
# Talk to Daniel on Wednesday, and if this sounds right then complete it for the following week.

'''
https://github.com/numbbo/coco.
Comparing Continuous Optimizers.
'''
# ______________________________________________________________________________________________________________________
# Notes on TASK #6 of Week 3.
# Going through the ReadMe of the Github repo.
# Went through the code a bit... it's not in Python, and I don't know C so it's a bit hard for me to follow.
# Also it seems like the repo has not been updated in a while, so they may not have up-to-date algs.
# ______________________________________________________________________________________________________________________
#
#
#         def _signOPT(self, model, oracle, num_iterations, x0, m, d, alpha, r, show_runs=False):
#             '''This function implements the descent part of signOPT, as detailed in:
#             'signOPT: a query-efficient hard-label adversarial attack'
#             Cheng, Singh, Chen, Chen, Liu, Hsieh
#             (2020) '''
#             # initialize arrays
#
#             # *********
#             # so I need to do this in the __init__() func. of my class.
#             rewards = np.zeros((num_iterations, 1))
#             x = np.squeeze(x0)
#             Z = np.zeros((m, d))
#             step_size = alpha
#             conv = None  # step number we exceed reward threshold at
#
#             # *********
#             # so this will be in my STEP() func.
#             for t in range(0, num_iterations):
#                 # generating u1, ..., um randomly sampled from a Gaussian distribution.
#                 for j in range(0, m):
#                     # dimension: d.
#                     temp = np.random.randn(1, d)  # randn is N(0, 1)
#                     Z[j, :] = temp / np.linalg.norm(temp)  # normalize
#
#                 g_hat = self.signOPTGradEstimator(oracle, x, Z, m, d, r * 1.01 ** (-t))
#                 x = x + step_size * 1.001 ** (-t) * g_hat  # Decaying stepsize model...
#                 if show_runs:
#                     rewards[t] = model.render(x)
#                 else:
#                     rewards[t] = model(x)
#                 print('current rewards at step', t + 1, ':', rewards[t])
#                 print('step_size:', step_size * 1.001 ** (-t))
#                 print('gradient norm:', np.linalg.norm(g_hat))
#                 if bool(model.reward_threshold):
#                     if rewards[t] > model.reward_threshold and conv == None:  # if we converge early
#                         conv = t + 1
#                         print('************** Reward Threshold Exceeded **************')
#             if not conv:
#                 oracle_comparisons = num_iterations * m
#             else:
#                 oracle_comparisons = conv * m
#             return x, rewards, conv, oracle_comparisons
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # first we need a REFERENCE....
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
#     def __init__(self, x0, step_size, f, params, function_budget=10000, shuffle=True,
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
#         self.block_size = int(np.ceil(self.n / self.J))
#         self.sparsity = int(np.ceil(oversampling_param * self.sparsity / self.J))
#         print(self.sparsity)
#         self.samples_per_block = int(np.ceil(oversampling_param * self.sparsity * np.log(self.block_size)))
#
#         # Define cosamp_params
#         if self.Type == "ZOBCD-R":
#             Z = 2 * (np.random.rand(self.samples_per_block, self.block_size) > 0.5) - 1
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
#
#
#
# class signOPT:
#
#     def signOPTGradEstimator(self, oracle, x_in, Z, m, d, r):
#         '''This function estimates the gradient vector from m Comparison
#         oracle queries, using the SignOPT method as detailed in Cheng et al's
#         'Sign-OPT' paper.
#         ================ INPUTS ======================
#         Z ......... An m-by-d matrix with rows z_i uniformly sampled from unit sphere
#         x_in ................. Any point in R^d
#         r ................ Sampling radius.
#         ================ OUTPUTS ======================
#         g_hat ........ approximation to g/||g||
#
#         23rd May 2020
#     '''
#
#         # *********
#         # use my Oracle function in oracle.py instead of this.
#         g_hat = np.zeros([d])
#         for i in range(m):
#             comp, _ = oracle(x_in, x_in + r * Z[i, :])
#             g_hat += comp * Z[i, :]
#         g_hat = g_hat / np.float(m)
#         print(np.linalg.norm(g_hat))
#         return g_hat
#
#
#     def _signOPT(self, model, oracle, num_iterations, x0, m, d, alpha, r, show_runs=False):
#         '''This function implements the descent part of signOPT, as detailed in:
#         'signOPT: a query-efficient hard-label adversarial attack'
#         Cheng, Singh, Chen, Chen, Liu, Hsieh
#         (2020) '''
#         # initialize arrays
#
#         # *********
#         # so I need to do this in the __init__() func. of my class.
#         rewards = np.zeros((num_iterations, 1))
#         x = np.squeeze(x0)
#         Z = np.zeros((m, d))
#         step_size = alpha
#         conv = None  # step number we exceed reward threshold at
#
#         # *********
#         # so this will be in my STEP() func.
#         for t in range(0, num_iterations):
#             # generating u1, ..., um randomly sampled from a Gaussian distribution.
#             for j in range(0, m):
#                 # dimension: d.
#                 temp = np.random.randn(1, d)  # randn is N(0, 1)
#                 Z[j, :] = temp / np.linalg.norm(temp)  # normalize
#
#             g_hat = self.signOPTGradEstimator(oracle, x, Z, m, d, r * 1.01 ** (-t))
#             x = x + step_size * 1.001 ** (-t) * g_hat  # Decaying stepsize model...
#             if show_runs:
#                 rewards[t] = model.render(x)
#             else:
#                 rewards[t] = model(x)
#             print('current rewards at step', t + 1, ':', rewards[t])
#             print('step_size:', step_size * 1.001 ** (-t))
#             print('gradient norm:', np.linalg.norm(g_hat))
#             if bool(model.reward_threshold):
#                 if rewards[t] > model.reward_threshold and conv == None:  # if we converge early
#                     conv = t + 1
#                     print('************** Reward Threshold Exceeded **************')
#         if not conv:
#             oracle_comparisons = num_iterations * m
#         else:
#             oracle_comparisons = conv * m
#         return x, rewards, conv, oracle_comparisons
#
#     # *********
#     # will get rid of this method.
#     def __call__(self, model, oracle, num_iterations, x0, m, d, alpha, r, show_runs=False, algo_seed=0):
#         np.random.seed(algo_seed)
#         return self._signOPT(model, oracle, num_iterations, x0, m, d, alpha, r, show_runs)

    #
    # s = SCOBO()
    # # s = GLDBS()
    # m = MujocoModel('Reacher-v2', 1000, -0.5, 0.5, reward_threshold=-8, msfilter='NoFilter')
    # m1 = MujocoModel('Reacher-v2', 1000, -0.5, 0.5, reward_threshold=-8, msfilter='MeanStdFilter')
    # # x, re, qu, conv = s(100, .2, np.zeros((22, 1)) + .3, 0.2/np.sqrt(2), m, GUIOracle(m, max_steps=100, M=1, gif_path_1='./videos/firstVideo.gif', gif_path_2='./videos/secondVideo.gif'), 11, 22, 5, False, -10)
    # # print(x, re, qu, conv)
    #
    # params = {"kappa": 1, "delta": 0.5, "mu": 1, "M": 1}  # .5 + min(delta, mu*|f(y) - f(x)|**(kappa-1)
    # o = Oracle(m, params)
    #
    # results = {'NoFRewards': [], 'MFRewards': [], 'NoFConv': [], 'MFConv': []}
    # for i in range(5):
    #     x, re, qu, conv = s(100, 0.2, np.random.randn(22, 1), 0.02 / np.sqrt(2), m, o, 26, 22, 11, False)
    #     results['NoFRewards'].append(max(re))
    #     results['NoFConv'].append(conv)
    #     x, re, qu, conv = s(100, 0.2, np.random.randn(22, 1), 0.02 / np.sqrt(2), m1, o, 26, 22, 11, False)
    #     results['MFRewards'].append(max(re))
    #     results['MFConv'].append(conv)
    #
    # print(results)
    # print('NoFilter average max rewards', np.mean(results['NoFRewards']))
    # print('MS filter average resutls', np.mean(results['MFRewards']))


# calling the class....
    # s = signOPT()
    # s(m, o, 100, np.zeros((22,1)), 10, 22, .2, .1,show_runs=True)



