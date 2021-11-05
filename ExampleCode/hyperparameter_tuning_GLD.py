# we want to figure out optimal values for R & r in the GLD algorithm.

"""
experimenting with tuning GLD hyperparameters.
"""

'''
Add this to ENVIRONMENT VARIABLES (edit configurations):
PYCUTEST_CACHE=.;CUTEST=/usr/local/opt/cutest/libexec;MYARCH=mac64.osx.gfo;SIFDECODE=/usr/local/opt/sifdecode/libexec;MASTSIF=/usr/local/opt/mastsif/share/mastsif;ARCHDEFS=/usr/local/opt/archdefs/libexec
'''

# choose 3 problems.
# write code to run GLD and increment r and R (in two nested FOR loops).
# run GLD for this problem with all combinations (should be 16) of r={.001, .01, .1, 1} and R={10, 100, 1000, 10000}.
#   the same x0 should be used in each run; all other parameters & # of iterations the same.
#   each FINAL function evaluation should be saved to a list(?).
# create a heat map (do this after figuring out your code).

import pycutest
# invokes the GLD algorithm class.
from Algorithms.gld_optimizer import GLDOptimizer
from ExampleCode.benchmarkfunctions import SparseQuadratic, MaxK
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from ExampleCode.oracle import Oracle, Oracle_pycutest
import copy

# problem 1: ROSENBR.
# problem 2: HILBERTA.
# problem 3: WATSON.

p1_name = 'ROSENBR'
problem_1 = pycutest.import_problem(p1_name)
# x0_1 = problem_1.x0
p2_name = 'HILBERTA'
problem_2 = pycutest.import_problem(p2_name)
# x0_2 = problem_2.x0
p3_name = 'WATSON'
problem_3 = pycutest.import_problem(p3_name)
# x0_3 = problem_3.x0

"""
*********
"""
# select which problem you want. (EDIT choose_problem & choose_name.)
choose_problem = problem_3
choose_name = p3_name
"""
*********
"""

# obtain initial value (same for each run of the same problem).
choose_x0 = choose_problem.x0

# parameters.
query_budget = 100
R = 10
r = 0.001

# lists.
list_R = []
list_r = []
list_evaluation = []

# GLD - version of the code without averaging across multiple runs:
'''
for i in range(4):
    r = 0.001
    for j in range(4):
        p = choose_problem
        x0_gld = copy.copy(choose_x0)
        oracle_gld = Oracle_pycutest(p.obj)
        gld1 = GLDOptimizer(oracle_gld, p.obj, x0_gld, R, r, 2 * max_function_evals)
        # step.
        termination = False
        prev_evals = 0
        while termination is False:
            solution, func_value, termination = gld1.step()
        print('function evaluation at solution: ', func_value[-1])
        print('R: ', R)
        print('r: ', r)
        list_R.append(R)
        list_r.append(r)
        list_evaluation.append(func_value[-1])
        # increment r.
        r *= 10
    # increment R.
    R *= 10
'''

# ---------
# GLD - version of the code that averages across 5 runs:
for i in range(4):
    r = 0.001
    for j in range(4):
        p = choose_problem
        x0_gld = copy.copy(choose_x0)
        oracle_gld = Oracle_pycutest(p.obj)
        list_evals = []
        for k in range(5):
            gld1 = GLDOptimizer(oracle_gld, p.obj, x0_gld, R, r, 2 * query_budget)
            # step.
            termination = False
            prev_evals = 0
            while termination is False:
                solution, func_value, termination = gld1.step()
            print('function evaluation at solution: ', func_value[-1])
            list_evals.append(func_value[-1])
        # take average of 5 evaluations.
        avg_eval = sum(list_evals)/len(list_evals)
        print('R: ', R)
        print('r: ', r)
        list_R.append(R)
        list_r.append(r)
        list_evaluation.append(avg_eval)
        # increment r.
        r *= 10
    # increment R.
    R *= 10

# checks.
print('\n')
print('R: ', list_R)
print('r: ', list_r)
print('last function value: ', list_evaluation)
list_symbol = [choose_name for i in range(len(list_evaluation))]
print('symbol: ', list_symbol)

# reshape function evaluations and problem name.
evaluations = np.array(list_evaluation).reshape(4, 4)
symbols = np.array(list_symbol).reshape(4, 4)
print('\n')
print('evaluations: ')
print(evaluations)
print('symbols: ')
print(symbols)

# create dataframe of symbols, eval, r, R.
df1 = pd.DataFrame(list(zip(list_symbol, list_evaluation, list_r, list_R)),
                  columns=['Symbols', 'Last_Function_Eval', 'r_Values', 'R_Values'])

# rows and columns of heatmap.
rows = np.unique(np.array(list_r))
columns = np.unique(np.array(list_R))
print('rows: ', rows)
print('columns: ', columns)
print('\n')

# use function pivot.
result = df1.pivot(index='R_Values', columns='r_Values', values='Last_Function_Eval')
print('result: ')
print(result)

# generate labels.
labels = (np.asarray(["{0} \n {1:.2f}".format(symb,value) for symb, value in zip(symbols.flatten(), evaluations.flatten())])).reshape(4, 4)

# plot heatmap.
fig, ax = plt.subplots(figsize=(12, 7))
title = "GLD Algorithm - " + str(choose_name) + " Heat Map"
plt.title(title, fontsize=18)
tt1 = ax.title
tt1.set_position([0.5, 1.05])
#ax.set_xticks([])
#ax.set_yticks([])
#ax.axis('off')
"""
cmap: colormap.
"""
# BuPu.
# rocket.
# mako.
# cubehelix.
# twilight.
# 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r'
sns.heatmap(result, annot=labels, fmt="", cmap='BuPu', linewidth=.3, ax=ax)
ax.set_xlabel('r', fontsize=12)
ax.set_ylabel('R', fontsize=12)
plt.ylabel('R')
plt.show()
plt.close()

# run this code for Problem_1, Problem_2, Problem_3.







# ______________________________________________________________________________________________________________________
# result = df.pivot(index='rows', columns='columns', values=)

"""
def run_GLD_pycutest(problem, x0, function_budget):
    # GLD.
    print('RUNNING ALGORITHM GLD....')
    p = problem
    R_ = 1e-1
    r_ = 1e-4
    x0_gld = copy.copy(x0)
    n = len(x0_gld)  # problem dimension.
    '''
    p.obj(x, Gradient=False) -> method which evaluates the function at x.
    '''
    oracle_gld = Oracle_pycutest(p.obj)  # comparison oracle.
    # GLD instance.
    gld1 = GLDOptimizer(oracle_gld, p.obj, x0_gld, R_, r_, 2 * function_budget)
    # step.
    termination = False
    prev_evals = 0
    while termination is False:
        solution, func_value, termination = gld1.step()
        #print('current value: ', func_value[-1])
    print('solution: ', solution)
    # plot.
    plt.plot(func_value)
    plt.title('GLD - linear.')
    #plt.show()
    plt.close()
    plt.semilogy(func_value)
    plt.title('GLD - log.')
    #plt.show()
    plt.close()
    print('function evaluation at solution: ', func_value[-1])
    return func_value[-1]
"""

'''
probs = pycutest.find_problems(constraints='U', userN=True)
sorted_problems = sorted(probs)

probs_under_100 = []
for p in sorted_problems:
    prob = pycutest.import_problem(p)
    x0 = prob.x0
    print('dimension of input vector of FUNCTION ' + str(p) + ': ' + str(len(x0)))
    # only want <= 100.
    if len(x0) <= 100:
        probs_under_100.append(p)

print('\n')
print('problems under 100: ', probs_under_100)
'''

# LOG scale when trying diff values for parameters.
# pick one or two problems, try to tune for that.
# take some of the hard problems (GLD doesn't decrease much).

# option 2: set the parameters R, r. Run on all problems (dimension 100 or less).
# maybe for now choose a few problems, and check. GRID SEARCH.

# do this for STP as well.


# # ---------
# print('sample invoke.')
# # GLD - FUNCTION sample invocation.
# n_def = 200  # problem dimension.
# s_exact = 20  # True sparsity.
# noise_amp = 0.001  # noise amplitude.
# # initialize objective function.
# obj_func_1 = SparseQuadratic(n_def, s_exact, noise_amp)
# obj_func_2 = MaxK(n_def, s_exact, noise_amp)
# '''
# max_function_evals = 10000
# '''
# max_function_evals = 500
# x_0_ = np.random.rand(n_def)
# print('shape of x_0_: ', len(x_0_))
# R_ = 10
# r_ = 1e-3
#
# # Define the comparison oracle.
# oracle = Oracle(obj_func_1)
#
# # GLDOptimizer instance.
# # def __init__(self, defined_func, x_0, R, r, function_budget).
# stp1 = GLDOptimizer(oracle, obj_func_1, x_0_, R_, r_, max_function_evals)
# # stp1 = GLDOptimizer(obj_func_2, x_0_, R_, r_, max_function_evals)
# # step.
# termination = False
# prev_evals = 0
# while termination is False:
#     # optimization step.
#     solution, func_value, termination = stp1.step()
#     # print('step')
#     print('current value: ', func_value[-1])
# # print the solution.
# print('\n')
# print('solution: ', solution)
# # plot the decreasing function.
# plt.plot(func_value)
# plt.show()
# # log x-axis.
# plt.semilogy(func_value)
# plt.show()
# # ---------
# print('\n')
# print('number of function vals: ', len(func_value))
# print('function evaluations: ', func_value)
