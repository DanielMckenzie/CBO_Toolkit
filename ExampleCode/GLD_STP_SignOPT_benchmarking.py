"""
Isha Slavin.
Week 3 - TASK #2.
"""

'''
Compare the performance of GLD vs. STP on our two benchmark
functions. As both are stochastic algorithms, we’ll need to average over
mutliple independent trials. When visualizing the results (i.e. function
values vs. number of oracle queries) we want to plot both the mean
and the standard deviation. This stackoverflow question: https://
stackoverflow.com/questions/51680298/plot-mean-and-standard-deviation-as-a-shaded-area-around-mean?
noredirect=1&lq=1 (and links therein) should explain how to do this.
'''

# So we want to compare the performance of GLD vs. STP on our 2 benchmark functions (MaxK, SparseQuadratic).
# Average over multiple independent trials.
# Graph: function values vs. # of oracle queries.
# we want to plot both the MEAN and the STANDARD DEVIATION.

# graph:
#   x-axis: # of oracle queries.
#   y_axis: function values.

import random
import numpy as np
from matplotlib import pyplot as plt
from ExampleCode.benchmarkfunctions import SparseQuadratic, MaxK, NonSparseQuadratic

from Algorithms.gld_optimizer import GLDOptimizer
from Algorithms.stp_optimizer import STPOptimizer

# oracle_queries = 20000


# --------------------------------
# STP.

n = 20000  # problem dimension.
s_exact = 200  # true sparsity.
noiseamp = 0.001  # noise amplitude.
function_budget = 20000
# function_budget = 1000

stp_func_list = []
gld_func_list = []
signOPT_func_list = []

# number_of_runs = 5


def run_STP_multiple_times(number_of_runs):
    for number in range(number_of_runs):
        '''
        RUN STP.
        '''
        print('sample invoke.')
        # sample invocation.
        # initialize objective function.
        obj_func_1_stp = SparseQuadratic(n, s_exact, noiseamp)
        obj_func_2_stp = MaxK(n, s_exact, noiseamp)
        obj_func_3_stp = NonSparseQuadratic(n, noiseamp)
        # create an instance of STPOptimizer.
        # direction_vector_type = 0  # original.
        # direction_vector_type = 1  # gaussian.
        # direction_vector_type = 2  # uniform from sphere.
        direction_vector_type = 3  # rademacher.
        a_k = 0.5  # step-size.
        # function_budget = 10000
        # initial x_0.
        x_0 = np.random.randn(n)
        # stp instance.
        stp1 = STPOptimizer(direction_vector_type, x_0, n, a_k, obj_func_1_stp, function_budget)
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

    # ***************************
    # Do this number_of_runs times.
    # We should be left with two lists (1 for stp, 1 for gld), each containing numpy arrays of the function values
    # from each run.
    # From there we can take the mean and standard deviation, and graph it.


# GLD.
def run_GLD_multiple_times(number_of_runs):
    for j in range(number_of_runs):
        # ***************************
        '''
        RUN GLD.
        '''
        # ---------
        print('sample invoke.')
        # GLD - FUNCTION sample invocation.
        # initialize objective function.
        obj_func_1 = SparseQuadratic(n, s_exact, noiseamp)
        obj_func_2 = MaxK(n, s_exact, noiseamp)
        obj_func_3 = NonSparseQuadratic(n, noiseamp)
        # max_function_evals = 10000
        random.seed()
        x_0_ = np.random.rand(n)
        print('shape of x_0_: ', len(x_0_))
        R_ = 10
        r_ = .01
        # GLDOptimizer instance.
        # def __init__(self, defined_func, x_0, R, r, function_budget).
        gld1 = GLDOptimizer(obj_func_1, x_0_, R_, r_, function_budget)
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

# Now, SignOPT.
def run_SignOPT_multiple_times(number_of_runs):
    for k in range(number_of_runs):
        '''
        run SignOPT.
        '''
        print('sample invoke.')
        n_def = 2000
        s_exact_1 = 200
        noise_amp = 0.001
        func_1 = SparseQuadratic(n_def, s_exact_1, noise_amp)
        func_2 = MaxK(n_def, s_exact_1, noise_amp)
        func_3 = NonSparseQuadratic(n, noiseamp)
        function_budget_1 = int(1e5)
        m = 100
        x0 = np.random.randn(n_def)
        step_size = 0.2
        r = 0.1
        # max_iters = int(2e4)
        max_iters = int(10000)
        # max_iters = int(500)

        # Define the comparison oracle
        from ExampleCode.oracle import Oracle
        oracle = Oracle(func_1)

        from Algorithms.SignOPT2 import SignOPT
        Opt = SignOPT(oracle, function_budget_1, x0, m, step_size, r, debug=False, function=func_1)

        for i in range(max_iters - 1):
            print(i)
            Opt.step()

        plt.semilogy(Opt.f_vals)
        # plt.show()
        print('\n')
        print('number of function vals: ', len(Opt.f_vals))
        func_value_3_arr = np.array(Opt.f_vals)
        signOPT_func_list.append(func_value_3_arr)


# GLD:
run_GLD_multiple_times(3)
print('*********')
print(len(gld_func_list))
print('*********')


# STP:
run_STP_multiple_times(3)
print('*********')
print(len(stp_func_list))
print('*********')

# SignOPT:
run_SignOPT_multiple_times(3)
print('*********')
print(len(signOPT_func_list))
print('*********')

print('\n')
print('\n')
print('shape of STP list: ', len(stp_func_list))
print('shape of GLD list: ', len(gld_func_list))
print('shape of SignOPT list: ', len(signOPT_func_list))

# STP.
mean_STP = np.mean(stp_func_list, axis=0)
std_dev_STP = np.std(stp_func_list, axis=0)

# GLD.
mean_GLD = np.mean(gld_func_list, axis=0)
std_dev_GLD = np.std(gld_func_list, axis=0)

# SignOPT.
mean_SignOPT = np.mean(signOPT_func_list, axis=0)
std_dev_SignOPT = np.std(signOPT_func_list, axis=0)

# STP.
mean_STP_list = mean_STP.tolist()
std_dev_STP_list = std_dev_STP.tolist()

# GLD.
mean_GLD_list = mean_GLD.tolist()
std_dev_GLD_list = std_dev_GLD.tolist()

# SignOPT.
mean_SignOPT_list = mean_SignOPT.tolist()
std_dev_SignOPT_list = std_dev_SignOPT.tolist()


# N = 21
# x = np.linspace(0, len(mean_STP_list)-1, len(mean_STP_list))
# x = np.linspace(0, len(mean_GLD_list)-1, len(mean_GLD_list))
# y = mean_STP_list
# y = mean_GLD_list

# print(y)
# print(len(y))

# fit a linear curve an estimate its y-values and their error.
# a, b = np.polyfit(x, y, deg=1)
# y_est = a * x + b


'''
fig, ax = plt.subplots()
ax.plot(x, y, '-')
ax.fill_between(x, y - y_err, y + y_err, alpha=0.2)
plt.show()
'''

# GLD.
"""
x = np.linspace(0, len(mean_GLD_list)-1, len(mean_GLD_list))
y = mean_GLD_list
# y_error = [std / 2 for std in std_dev_GLD_list]
y_error = std_dev_GLD_list
y_error_np = np.array(y_error)
fig, ax = plt.subplots()
ax.plot(x, y, '-')

y_error_bottom = np.subtract(mean_GLD, y_error_np)
y_error_top = np.add(mean_GLD, y_error_np)

y_error_bottom_list = y_error_bottom.tolist()
y_error_top_list = y_error_top.tolist()

ax.fill_between(x, y_error_bottom, y_error_top, alpha=0.2)
plt.show()
"""

# STP.
x = np.linspace(0, len(mean_STP_list)-1, len(mean_STP_list))
x_2 = np.linspace(0, len(mean_GLD_list)-1, len(mean_GLD_list))
x_3 = np.linspace(0, len(mean_SignOPT_list)-1, len(mean_SignOPT_list))
y_1 = mean_STP_list
y_2 = mean_GLD_list
y_3 = mean_SignOPT_list
print(y_1)
print(y_2)
print(y_3)
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
# y_error = [std / 2 for std in std_dev_GLD_list]

# fig, ax = plt.subplots()
# ax.plot(x, y_1, '-')
# ax.plot(x, y_2, '-')

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

# ax.fill_between(x, y_error_bottom_1, y_error_top_1, alpha=0.2)
# ax.fill_between(x, y_error_bottom_2, y_error_top_2, alpha=0.2)

plt.figure()

plt.plot(x, y_1, color='orange', label='STP')
plt.plot(x_2, y_2, color='blue', label='GLD')
plt.plot(x_3, y_3, color = 'black', label='SignOPT')
plt.fill_between(x, y_error_bottom_list_1, y_error_top_list_1, color='orange', alpha=.2)
plt.fill_between(x_2, y_error_bottom_list_2, y_error_top_list_2, color='blue', alpha=.2)
plt.fill_between(x_3, y_error_bottom_list_3, y_error_top_list_3, color='black', alpha=.2)

plt.xlabel('number of oracle queries')
plt.ylabel('function values')

plt.legend()
plt.show()
# plt.savefig('/Users/isha_slavin/Downloads/STP_GLD_figure_comparison.png')
plt.close()






