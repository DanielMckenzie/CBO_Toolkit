from __future__ import print_function
import numpy as np
import pandas as pd
import copy
import pycutest
from matplotlib import pyplot as plt
from benchmarkfunctions import SparseQuadratic, MaxK
from oracle import Oracle, Oracle_pycutest
from Algorithms.stp_optimizer import STPOptimizer
from Algorithms.gld_optimizer import GLDOptimizer
from Algorithms.SignOPT2 import SignOPT
from Algorithms.scobo_optimizer import SCOBOoptimizer
from Algorithms.CMA_2 import CMA

print('hello, world!')
print('\n')

# Find unconstrained, variable-dimension problems.
probs = pycutest.find_problems(constraints='U', userN=True)
# print(sorted(probs)).
print('number of problems: ', len(probs))
print(sorted(probs))

# Properties of problem ROSENBR.
print('\n')

for problem in probs:
    print(problem + ': ' + str(pycutest.problem_properties(problem)))

# functions to run.

'''
STP.
'''
def run_STP_pycutest(problem, x0, function_budget):
    # STP.
    print('RUNNING ALGORITHM STP....')
    p = problem
    # direction_vector_type = 0  # original.
    # direction_vector_type = 1  # gaussian.
    direction_vector_type = 2  # uniform from sphere.
    # direction_vector_type = 3  # rademacher.
    a_k = 0.001  # step-size.
    x0_stp = copy.copy(x0)
    n = len(x0_stp)  # problem dimension.
    '''
    p.obj(x, Gradient=False) -> method which evaluates the function at x.
    '''
    oracle_stp = Oracle_pycutest(p.obj)  # comparison oracle.
    # STP instance.
    stp1 = STPOptimizer(oracle_stp, direction_vector_type, x0_stp, n, a_k, p.obj, 2 * function_budget)
    # step.
    termination = False
    prev_evals = 0
    while termination is False:
        solution, func_value, termination = stp1.step()
        #print('current value: ', func_value[-1])
    print('solution: ', solution)
    # plot.
    plt.plot(func_value)
    plt.title('STP - linear.')
    #plt.show()
    plt.close()
    plt.semilogy(func_value)
    plt.title('STP - log.')
    #plt.show()
    plt.close()
    print('function evaluation at solution: ', func_value[-1])
    return func_value


'''
GLD.
'''
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
    return func_value


'''
SIGNOPT.
'''
def run_signOPT_pycutest(problem, x0, function_budget):
    # SignOPT.
    print('RUNNING ALGORITHM SIGNOPT....')
    p = problem
    m = 100
    x0_signopt = copy.copy(x0)
    n = len(x0_signopt)  # problem dimension.
    step_size = 0.2
    r = 0.1
    '''
    p.obj(x, Gradient=False) -> method which evaluates the function at x.
    '''
    oracle_signopt = Oracle_pycutest(p.obj)  # comparison oracle.
    # signOPT instance.
    signopt1 = SignOPT(oracle_signopt, function_budget, x0_signopt, m, step_size, r, debug=False, function=p.obj)
    # step.
    for i in range(function_budget - 1):
        #print(i)
        signopt1.step()
    # plot.
    plt.plot(signopt1.f_vals)
    plt.title('SignOPT - linear.')
    #plt.show()
    plt.close()
    plt.semilogy(signopt1.f_vals)
    plt.title('SignOPT - log.')
    #plt.show()
    plt.close()
    print('function evaluation at solution: ', signopt1.f_vals[-1])
    return signopt1.f_vals

def run_SCOBO_pycutest(problem, x0, function_budget):
    # SCOBO.
    print('RUNNING ALGORITHM SCOBO....')
    p = problem
    m_scobo = 100  # should always be larger than s_exact.
    x0_scobo = copy.copy(x0)
    n = len(x0_scobo)  # problem dimension.
    stepsize = 0.01
    s_exact = 20
    r = 0.1
    '''
    p.obj(x, Gradient=False) -> method which evaluates the function at x.
    '''
    oracle_scobo = Oracle_pycutest(p.obj)  # comparison oracle.
    # SCOBO instance.
    scobo1 = SCOBOoptimizer(oracle_scobo, stepsize, function_budget, x0_scobo, r, m_scobo, s_exact, objfunc=p.obj)
    # step.
    for i in range(function_budget):
        #print(i)
        err = scobo1.step()
        #print(err)
    # plot.
    plt.plot(scobo1.function_vals)
    plt.title('SCOBO - linear.')
    #plt.show()
    plt.close()
    plt.semilogy(scobo1.function_vals)
    plt.title('SCOBO - log.')
    #plt.show()
    plt.close()
    print('function evaluation at solution: ', scobo1.function_vals[-1])
    return scobo1.function_vals


# invoke functions.
probs = pycutest.find_problems(constraints='U', userN=True)
print('number of problems: ', len(probs))
sorted_problems = sorted(probs)
list_of_problems_testing = sorted_problems[:5]

probs_under_100 = []
for p in sorted_problems:
    prob = pycutest.import_problem(p)
    x0 = prob.x0
    print('dimension of input vector of FUNCTION ' + str(p) + ': ' + str(len(x0)))
    # only want <= 100.
    if len(x0) <= 100:
        probs_under_100.append(p)

print('\n')
print('*********')
print('problems: ', probs_under_100)

problem_now = ['LUKSAN22LS']

# function evaluations (to be plotted).
stp_func_list = []
gld_func_list = []
signOPT_func_list = []
scobo_func_list = []

# for problem in list_of_problems_testing:
for problem in probs_under_100:
    p_invoke_ = pycutest.import_problem(problem)
    '''
    x0_p_ = p_invoke_.x0
    dim_x0_ = len(x0_p_)
    print('dimension of problem: ', dim_x0_)
    x0_invoke_ = np.random.randn(dim_x0_)
    '''
    x0_invoke_ = p_invoke_.x0
    print('dimension of problem: ', len(x0_invoke_))
    function_budget_ = 100
    # STP.
    print('invoking STP in a loop....')
    stp_func_list = run_STP_pycutest(p_invoke_, copy.copy(x0_invoke_), function_budget_)
    print('\n')
    # GLD.
    print('invoking GLD in a loop....')
    gld_func_list = run_GLD_pycutest(p_invoke_, copy.copy(x0_invoke_), function_budget_)
    print('\n')
    # SignOPT.
    print('invoking SignOPT in a loop....')
    signOPT_func_list = run_signOPT_pycutest(p_invoke_, copy.copy(x0_invoke_), function_budget_)
    print('\n')
    # SCOBO.
    print('invoking SCOBO in a loop....')
    scobo_func_list = run_SCOBO_pycutest(p_invoke_, copy.copy(x0_invoke_), function_budget_)
    print('\n')

    # plot.
    plt.figure()
    # each algorithm's performance.
    plt.plot(stp_func_list, color='orange', label='STP')
    plt.plot(gld_func_list, color='blue', label='GLD')
    plt.plot(signOPT_func_list, color='black', label='SignOPT')
    plt.plot(scobo_func_list, color='purple', label='SCOBO')
    # name axes & show graph.
    plt.xlabel('number of oracle queries')
    plt.ylabel('function values')
    plt.legend()
    plt.title('problem: ' + str(problem))
    plt.show()
    plt.close()




