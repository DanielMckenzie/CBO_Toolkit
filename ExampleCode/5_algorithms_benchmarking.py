"""
ALGORITHMS.
"""
# STP.
# GLD.
# SignOPT.
# SCOBO.
# CMA.

"""
Imports.
"""
import random
import numpy as np
from matplotlib import pyplot as plt
from ExampleCode.benchmarkfunctions import SparseQuadratic, MaxK, NonSparseQuadratic
from ExampleCode.oracle import Oracle

''' STP. '''
from Algorithms.stp_optimizer import STPOptimizer

''' GLD. '''
from Algorithms.gld_optimizer import GLDOptimizer

''' SignOPT. '''
from Algorithms.SignOPT2 import SignOPT

''' SCOBO. '''
from Algorithms.scobo_optimizer import SCOBOoptimizer

''' CMA. '''
from Algorithms.CMA_2 import CMA
# import copy.
import copy
# import sys.
import sys

sys.path.append('..')

# initialize.
n = 200  # problem dimension.
s_exact = 20  # true sparsity.
noiseamp = 0.001  # noise amplitude.
function_budget = 500
# objective functions.
obj_func_1 = SparseQuadratic(n, s_exact, noiseamp)
obj_func_2 = MaxK(n, s_exact, noiseamp)
obj_func_3 = NonSparseQuadratic(n, noiseamp)
# decide which function we will benchmark on.
objective_function_choice = obj_func_3
# oracle.
main_oracle = Oracle(objective_function_choice)
oracle_1 = Oracle(obj_func_1)
oracle_2 = Oracle(obj_func_2)
oracle_3 = Oracle(obj_func_3)
# initialize lists.
stp_func_list = []
gld_func_list = []
signOPT_func_list = []
scobo_func_list = []
cma_func_list = []
# number of times we will run each alg and average over them.
number_runs = 3
# initial x0 to be used for each alg.
'''
X_0 = np.random.randn(n, number_runs)
'''
X_0 = np.random.randn(n)

# call STP.
def run_STP(number_of_runs):
    for number in range(number_of_runs):
        '''
        RUN STP.
        '''
        print('sample invoke.')
        # create an instance of STPOptimizer.
        # direction_vector_type = 0  # original.
        # direction_vector_type = 1  # gaussian.
        # direction_vector_type = 2  # uniform from sphere.
        direction_vector_type = 3  # rademacher.
        a_k = 0.5  # step-size.
        # function_budget = 10000
        # initial x_0.
        random.seed()
        '''
        x_0 = np.random.randn(n)
        '''
        ## DM Code suggestion
        '''
        x_0 = copy.copy(X_0[:, number])
        '''
        x_0 = copy.copy(X_0)
        # stp instance.
        stp1 = STPOptimizer(main_oracle, direction_vector_type, x_0, n, a_k, objective_function_choice, function_budget)
        # step.
        termination = False
        # prev_evals = 0
        while termination is False:
            # optimization step.
            solution, func_value, termination = stp1.step()
            # print('step')
            print('current value: ', func_value[-1])
        # plot the decreasing function.
        plt.plot(func_value)
        # plt.show()
        # log x-axis.
        plt.semilogy(func_value)
        # plt.show()
        # ---------
        print('\n')
        print('number of function vals: ', len(func_value))
        # cast func_value LIST to a NUMPY ARRAY.
        func_value_arr = np.array(func_value)
        # append array of function values to STP list.
        stp_func_list.append(func_value_arr)


# call GLD.
def run_GLD(number_of_runs):
    for j in range(number_of_runs):
        '''
        RUN GLD.
        '''
        # ---------
        print('sample invoke.')
        random.seed()
        '''
        x_0_ = np.random.rand(n)
        '''
        '''
        x_0_ = copy.copy(X_0[:, j])
        '''
        x_0_ = copy.copy(X_0)
        # print('shape of x_0_: ', len(x_0_))
        R_ = 10
        r_ = .01
        # GLDOptimizer instance.
        gld1 = GLDOptimizer(main_oracle, objective_function_choice, x_0_, R_, r_, function_budget)
        # gld1 = GLDOptimizer(obj_func_2, x_0_, R_, r_, max_function_evals)
        # gld1 = GLDOptimizer(obj_func_1, x_0_, R_, r_, max_function_evals)
        # step.
        termination_2 = False
        prev_evals = 0
        while termination_2 is False:
            # optimization step.
            solution_2, func_value_2, termination_2 = gld1.step()
            # print('step')
            print('current value: ', func_value_2[-1])
        # print the solution.
        print('\n')
        print('solution: ', solution_2)
        # plot the decreasing function.
        plt.plot(func_value_2)
        # plt.show()
        # log x-axis.
        plt.semilogy(func_value_2)
        # plt.show()
        # ---------
        print('\n')
        print('number of function vals: ', len(func_value_2))
        # cast func_value LIST to a NUMPY ARRAY.
        func_value_2_arr = np.array(func_value_2)
        # append array of function values to STP list.
        gld_func_list.append(func_value_2_arr)


# call SignOPT.
def run_SignOPT(number_of_runs):
    for k in range(number_of_runs):
        '''
        RUN SignOPT.
        '''
        print('sample invoke.')
        # function_budget_1 = int(1e5)
        m = 100
        random.seed()
        '''
        x0 = np.random.randn(n)
        '''
        '''
        x0 = copy.copy(X_0[:, k])
        '''
        x0 = copy.copy(X_0)
        step_size = 0.2
        r = 0.1
        max_iters = int(function_budget/2)
        # max_iters = int(500)
        # from Algorithms.SignOPT2 import SignOPT
        # SignOPT instance.
        Opt = SignOPT(main_oracle, function_budget, x0, m, step_size, r, debug=False, function=objective_function_choice)
        for i in range(max_iters - 1):
            print(i)
            Opt.step()
        plt.semilogy(Opt.f_vals)
        # plt.show()
        print('\n')
        print('number of function vals: ', len(Opt.f_vals))
        func_value_3_arr = np.array(Opt.f_vals)
        signOPT_func_list.append(func_value_3_arr)


# figure out how to get SCOBO working. email Daniel that the installation isn't working?

# call SCOBO.
def run_SCOBO(number_of_runs):
    for m in range(number_of_runs):
        '''
        RUN SCOBO.
        '''
        print('sample invoke')
        step_size = 0.5
        query_budget = int(1e5)
        random.seed()
        '''
        x0 = np.random.randn(n)
        '''
        '''
        x0 = copy.copy(X_0[:, m])
        '''
        x0 = copy.copy(X_0)
        r = 0.1
        m = 100
        s_ex = 20
        max_iters = int(function_budget/2)
        # SCOBO instance.
        Opt = SCOBOoptimizer(main_oracle, step_size, query_budget, x0, r, m, s_ex, objfunc=objective_function_choice)
        for i in range(max_iters):
            # print(i)
            err = Opt.step()
            print(err)
        print('\n')
        print('number of function vals: ', len(Opt.function_vals))
        func_value_4_arr = np.array(Opt.function_vals)
        scobo_func_list.append(func_value_4_arr)


# call CMA.
def run_CMA(number_of_runs):
    for l in range(number_of_runs):
        # write function evaluations into this list for each of the runs.
        all_CMA_func_vals = []
        '''
        RUN CMA_ES.
        '''
        print('sample invoke.')
        query_budget = int(1e5)
        m = 100
        '''
        x0 = 100 * np.random.randn(n)
        '''
        random.seed()
        '''
        x0 = np.random.randn(n)
        '''
        '''
        x0 = copy.copy(X_0[:, l])
        '''
        x0 = copy.copy(X_0)
        step_size = 0.2
        r = 0.1
        max_iters = int(function_budget/2)
        lam = 10
        mu = 5
        sigma = 0.5
        # CMA_ES instance.
        Opt = CMA(main_oracle, query_budget, x0, lam, mu, sigma, function=objective_function_choice)
        for i in range(max_iters):
            val = Opt.step()
            print(str(i) + ': ' + str(val))
            all_CMA_func_vals.append(val)
        print('\n')
        print('number of function vals: ', len(all_CMA_func_vals))
        func_value_5_arr = np.array(all_CMA_func_vals)
        cma_func_list.append(func_value_5_arr)


# invoke.
# STP:
print('\n')
print('STP....')
run_STP(3)
print(len(stp_func_list))
# GLD:
print('\n')
print('GLD....')
run_GLD(3)
print(len(gld_func_list))
print('\n')
print('*********')
print('*** GLD GLD GLD *** : ')
print(gld_func_list)
print('*********')
print('\n')
# SignOPT:
print('\n')
print('SIGNOPT....')
run_SignOPT(3)
print(len(signOPT_func_list))
# SCOBO:
print('\n')
print('SCOBO....')
run_SCOBO(3)
print(len(scobo_func_list))
# CMA:
print('\n')
print('CMA....')
run_CMA(3)
print(len(cma_func_list))

# mean & standard deviation lists.
# STP.
mean_STP = np.mean(stp_func_list, axis=0)
std_dev_STP = np.std(stp_func_list, axis=0)
mean_STP_list = mean_STP.tolist()
std_dev_STP_list = std_dev_STP.tolist()
# GLD.
mean_GLD = np.mean(gld_func_list, axis=0)
std_dev_GLD = np.std(gld_func_list, axis=0)
mean_GLD_list = mean_GLD.tolist()
std_dev_GLD_list = std_dev_GLD.tolist()
# SignOPT.
mean_SignOPT = np.mean(signOPT_func_list, axis=0)
std_dev_SignOPT = np.std(signOPT_func_list, axis=0)
mean_SignOPT_list = mean_SignOPT.tolist()
std_dev_SignOPT_list = std_dev_SignOPT.tolist()
# SCOBO.
mean_SCOBO = np.mean(scobo_func_list, axis=0)
std_dev_SCOBO = np.std(scobo_func_list, axis=0)
mean_SCOBO_list = mean_SCOBO.tolist()
std_dev_SCOBO_list = std_dev_SCOBO.tolist()
# CMA.
mean_CMA = np.mean(cma_func_list, axis=0)
std_dev_CMA = np.std(cma_func_list, axis=0)
mean_CMA_list = mean_CMA.tolist()
std_dev_CMA_list = std_dev_CMA.tolist()


# axes.
x = np.linspace(0, len(mean_STP_list) - 1, len(mean_STP_list))
x_2 = np.linspace(0, len(mean_GLD_list) - 1, len(mean_GLD_list))
x_3 = np.linspace(0, len(mean_SignOPT_list) - 1, len(mean_SignOPT_list))
x_4 = np.linspace(0, len(mean_SCOBO_list) - 1, len(mean_SCOBO_list))
x_5 = np.linspace(0, len(mean_CMA_list) - 1, len(mean_CMA_list))
y_1 = mean_STP_list
y_2 = mean_GLD_list
y_3 = mean_SignOPT_list
y_4 = mean_SCOBO_list
y_5 = mean_CMA_list

# STP standard deviation:
y_error_1 = std_dev_STP_list
y_error_np_1 = np.array(y_error_1)
y_error_bottom_1 = np.subtract(mean_STP, y_error_np_1)
y_error_top_1 = np.add(mean_STP, y_error_np_1)
y_error_bottom_list_1 = y_error_bottom_1.tolist()
y_error_top_list_1 = y_error_top_1.tolist()
# GLD standard deviation:
y_error_2 = std_dev_GLD_list
y_error_np_2 = np.array(y_error_2)
y_error_bottom_2 = np.subtract(mean_GLD, y_error_np_2)
y_error_top_2 = np.add(mean_GLD, y_error_np_2)
y_error_bottom_list_2 = y_error_bottom_2.tolist()
y_error_top_list_2 = y_error_top_2.tolist()
# SignOPT standard deviation:
y_error_3 = std_dev_SignOPT_list
y_error_np_3 = np.array(y_error_3)
y_error_bottom_3 = np.subtract(mean_SignOPT, y_error_np_3)
y_error_top_3 = np.add(mean_SignOPT, y_error_np_3)
y_error_bottom_list_3 = y_error_bottom_3.tolist()
y_error_top_list_3 = y_error_top_3.tolist()
# SCOBO standard deviation:
y_error_4 = std_dev_SCOBO_list
y_error_np_4 = np.array(y_error_4)
y_error_bottom_4 = np.subtract(mean_SCOBO, y_error_np_4)
y_error_top_4 = np.add(mean_SCOBO, y_error_np_4)
y_error_bottom_list_4 = y_error_bottom_4.tolist()
y_error_top_list_4 = y_error_top_4.tolist()
# CMA standard deviation:
y_error_5 = std_dev_CMA_list
y_error_np_5 = np.array(y_error_5)
y_error_bottom_5 = np.subtract(mean_CMA, y_error_np_5)
y_error_top_5 = np.add(mean_CMA, y_error_np_5)
y_error_bottom_list_5 = y_error_bottom_5.tolist()
y_error_top_list_5 = y_error_top_5.tolist()


# create plot.
plt.figure()

# graph the figure.
plt.plot(x, y_1, color='orange', label='STP')
plt.plot(x_2, y_2, color='blue', label='GLD')
plt.plot(x_3, y_3, color='black', label='SignOPT')
plt.plot(x_4, y_4, color='purple', label='SCOBO')
plt.plot(x_5, y_5, color='green', label='CMA')
# fill in error margins.
plt.fill_between(x, y_error_bottom_list_1, y_error_top_list_1, color='orange', alpha=.2)
plt.fill_between(x_2, y_error_bottom_list_2, y_error_top_list_2, color='blue', alpha=.2)
plt.fill_between(x_3, y_error_bottom_list_3, y_error_top_list_3, color='black', alpha=.2)
plt.fill_between(x_4, y_error_bottom_list_4, y_error_top_list_4, color='purple', alpha=.2)
plt.fill_between(x_5, y_error_bottom_list_5, y_error_top_list_5, color='green', alpha=.2)

# name axes & show graph.
plt.xlabel('number of oracle queries')
plt.ylabel('function values')
plt.legend()
plt.show()
plt.close()






