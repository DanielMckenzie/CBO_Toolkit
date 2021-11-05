# we want to figure out optimal values for s_exact & m in the SCOBO algorithm.

"""
experimenting with tuning SCOBO hyperparameters.
"""

'''
Add this to ENVIRONMENT VARIABLES (edit configurations):
PYCUTEST_CACHE=.;CUTEST=/usr/local/opt/cutest/libexec;MYARCH=mac64.osx.gfo;SIFDECODE=/usr/local/opt/sifdecode/libexec;MASTSIF=/usr/local/opt/mastsif/share/mastsif;ARCHDEFS=/usr/local/opt/archdefs/libexec
'''

# choose 3 problems.
# write code to run SCOBO and increment s_exact and m (in two nested FOR loops).
# run SCOBO for this problem with all combinations (16) of s_exact={10, 20, 50, 100} & m={100, 200, 500, 1000}.
#   the same x0 should be used in each run; all other parameters & # of iterations the same.
#   each FINAL function evaluation should be saved to a list(?).
# create a heat map (do this after figuring out your code).

import pycutest
# invokes the GLD algorithm class.
from Algorithms.gld_optimizer import GLDOptimizer
from Algorithms.scobo_optimizer import SCOBOoptimizer
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

'''
SCOBO invocation:
# Defining the function
s_exact = 20
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
'''

# parameters.
query_budget = 1e4
s_exact_initialize_list = [10, 20, 50, 100]
m_initialize_list = [100, 200, 500, 1000]

# lists.
list_s_exact = []
list_m = []
list_evaluation = []

# SCOBO - version of the code without averaging across multiple runs:
'''
for i in range(4):
    s_exact = s_exact_initialize_list[i]
    for j in range(4):
        m = m_initialize_list[j]
        p = choose_problem
        x0_scobo = copy.copy(choose_x0)
        step_size = 0.01
        r_scobo = 0.1
        oracle_scobo = Oracle_pycutest(p.obj)
        scobo1 = SCOBOoptimizer(oracle_scobo, step_size, query_budget, x0_scobo, r_scobo, m, s_exact, objfunc=p.obj)
        # step.
        termination = False
        prev_evals = 0
        while termination is False:
            # optimization step.
            solution, func_value, termination, queries = scobo1.step()
            print('current value at ' + str(prev_evals) + ': ', func_value[-1])
            prev_evals += 1
        print('function evaluation at solution: ', func_value[-1])
        print('s_exact: ', s_exact)
        print('m: ', m)
        list_s_exact.append(s_exact)
        list_m.append(m)
        list_evaluation.append(func_value[-1])
        # m will be automatically incremented.
    # s_exact will be automatically incremented.
'''

# SCOBO - version of the code that averages across 5 runs:
for i in range(4):
    s_exact = s_exact_initialize_list[i]
    for j in range(4):
        m = m_initialize_list[j]
        p = choose_problem
        x0_scobo = copy.copy(choose_x0)
        step_size = 0.01
        r_scobo = 0.1
        oracle_scobo = Oracle_pycutest(p.obj)
        list_evals = []
        for k in range(5):
            scobo1 = SCOBOoptimizer(oracle_scobo, step_size, query_budget, x0_scobo, r_scobo, m, s_exact, objfunc=p.obj)
            # step.
            termination = False
            prev_evals = 0
            while termination is False:
                # optimization step.
                solution, func_value, termination, queries = scobo1.step()
                # print('current value at ' + str(prev_evals) + ': ', func_value[-1])
                prev_evals += 1
            print('function evaluation at solution: ', func_value[-1])
            # print('x0: ', x0_scobo)
            list_evals.append(func_value[-1])
        # take average of 5 evaluations.
        avg_eval = sum(list_evals) / len(list_evals)
        print('s_exact: ', s_exact)
        print('m: ', m)
        list_s_exact.append(s_exact)
        list_m.append(m)
        list_evaluation.append(avg_eval)
        # m will be automatically incremented.
    # s_exact will be automatically incremented.

print('\n')
print('s_exact: ', list_s_exact)
print('m: ', list_m)
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
df1 = pd.DataFrame(list(zip(list_symbol, list_evaluation, list_m, list_s_exact)),
                   columns=['Symbols', 'Last_Function_Eval', 'm_Values', 's_exact_Values'])

# rows and columns of heatmap.
rows = np.unique(np.array(list_m))
columns = np.unique(np.array(list_s_exact))
print('rows: ', rows)
print('columns: ', columns)
print('\n')

# use function pivot.
result = df1.pivot(index='s_exact_Values', columns='m_Values', values='Last_Function_Eval')
print('result: ')
print(result)

# generate labels.
labels = (np.asarray(
    ["{0} \n {1:.2f}".format(symb, value) for symb, value in zip(symbols.flatten(), evaluations.flatten())])).reshape(4,
                                                                                                                      4)

# plot heatmap.
fig, ax = plt.subplots(figsize=(12, 7))
title = "SCOBO Algorithm - " + str(choose_name) + " Heat Map"
plt.title(title, fontsize=18)
tt1 = ax.title
tt1.set_position([0.5, 1.05])
# ax.set_xticks([])
# ax.set_yticks([])
# ax.axis('off')
"""
cmap: colormap.
"""
# BuPu.
# rocket.
# mako.
# cubehelix.
# twilight.
# 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r'
sns.heatmap(result, annot=labels, fmt="", cmap='rocket', linewidth=.3, ax=ax)
ax.set_xlabel('m', fontsize=12)
ax.set_ylabel('s_exact', fontsize=12)
plt.show()
plt.close()

# run this code for Problem_1, Problem_2, Problem_3.







# ______________________________________________________________________________________________________________________
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
#
# # ---------
# # GLD - version of the code that averages across 5 runs:
# for i in range(4):
#     r = 0.001
#     for j in range(4):
#         p = choose_problem
#         x0_gld = copy.copy(choose_x0)
#         oracle_gld = Oracle_pycutest(p.obj)
#         list_evals = []
#         for k in range(5):
#             gld1 = GLDOptimizer(oracle_gld, p.obj, x0_gld, R, r, 2 * query_budget)
#             # step.
#             termination = False
#             prev_evals = 0
#             while termination is False:
#                 solution, func_value, termination = gld1.step()
#             print('function evaluation at solution: ', func_value[-1])
#             list_evals.append(func_value[-1])
#         # take average of 5 evaluations.
#         avg_eval = sum(list_evals)/len(list_evals)
#         print('R: ', R)
#         print('r: ', r)
#         list_R.append(R)
#         list_r.append(r)
#         list_evaluation.append(avg_eval)
#         # increment r.
#         r *= 10
#     # increment R.
#     R *= 10
#
# # checks.
# print('\n')
# print('R: ', list_R)
# print('r: ', list_r)
# print('last function value: ', list_evaluation)
# list_symbol = [choose_name for i in range(len(list_evaluation))]
# print('symbol: ', list_symbol)
#
# # reshape function evaluations and problem name.
# evaluations = np.array(list_evaluation).reshape(4, 4)
# symbols = np.array(list_symbol).reshape(4, 4)
# print('\n')
# print('evaluations: ')
# print(evaluations)
# print('symbols: ')
# print(symbols)
#
# # create dataframe of symbols, eval, r, R.
# df1 = pd.DataFrame(list(zip(list_symbol, list_evaluation, list_r, list_R)),
#                   columns=['Symbols', 'Last_Function_Eval', 'r_Values', 'R_Values'])
#
# # rows and columns of heatmap.
# rows = np.unique(np.array(list_r))
# columns = np.unique(np.array(list_R))
# print('rows: ', rows)
# print('columns: ', columns)
# print('\n')
#
# # use function pivot.
# result = df1.pivot(index='R_Values', columns='r_Values', values='Last_Function_Eval')
# print('result: ')
# print(result)
#
# # generate labels.
# labels = (np.asarray(["{0} \n {1:.2f}".format(symb,value) for symb, value in zip(symbols.flatten(), evaluations.flatten())])).reshape(4, 4)
#
# # plot heatmap.
# fig, ax = plt.subplots(figsize=(12, 7))
# title = "GLD Algorithm - " + str(choose_name) + " Heat Map"
# plt.title(title, fontsize=18)
# tt1 = ax.title
# tt1.set_position([0.5, 1.05])
# #ax.set_xticks([])
# #ax.set_yticks([])
# #ax.axis('off')
# """
# cmap: colormap.
# """
# # BuPu.
# # rocket.
# # mako.
# # cubehelix.
# # twilight.
# # 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r'
# sns.heatmap(result, annot=labels, fmt="", cmap='BuPu', linewidth=.3, ax=ax)
# ax.set_xlabel('r', fontsize=12)
# ax.set_ylabel('R', fontsize=12)
# plt.ylabel('R')
# plt.show()
# plt.close()
#
# # run this code for Problem_1, Problem_2, Problem_3.
#
