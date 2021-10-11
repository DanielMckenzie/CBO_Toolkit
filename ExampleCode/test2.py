# add this to environment variables:
"""
PYCUTEST_CACHE=.;CUTEST=/usr/local/opt/cutest/libexec;MYARCH=mac64.osx.gfo;SIFDECODE=/usr/local/opt/sifdecode/libexec;MASTSIF=/usr/local/opt/mastsif/share/mastsif
"""

"""
PyCUTEst example: minimize 2D Rosenbrock function using Newton's method.

Jaroslav Fowkes and Lindon Roberts, 2020.
"""

# Ensure compatibility with Python 2
from __future__ import print_function
import numpy as np
import pycutest
from matplotlib import pyplot as plt
from benchmarkfunctions import SparseQuadratic, MaxK
from oracle import Oracle, Oracle_pycutest
from Algorithms.gld_optimizer import GLDOptimizer

p = pycutest.import_problem('ROSENBR')

pycutest.print_available_sif_params('ROSENBR')

print("Rosenbrock function in %gD" % p.n)

iters = 0

# p.n = 3
x = p.x0
# x = [1, 2, 3]
f, g = p.obj(x, gradient=True)  # objective and gradient
H = p.hess(x)  # Hessian

while iters < 100 and np.linalg.norm(g) > 1e-10:
    print("Iteration %g: objective value is %g with norm of gradient %g at x = %s" % (
    iters, f, np.linalg.norm(g), str(x)))
    s = np.linalg.solve(H, -g)  # Newton step
    x = x + s  # used fixed step length
    f, g = p.obj(x, gradient=True)
    H = p.hess(x)
    iters += 1

print("Found minimum x = %s after %g iterations" % (str(x), iters))
print("Done")

# ---------
print('sample invoke.')
# GLD - FUNCTION sample invocation.
n_def = 200  # problem dimension.
s_exact = 20  # True sparsity.
noise_amp = 0.001  # noise amplitude.
# initialize objective function.
obj_func_1 = SparseQuadratic(n_def, s_exact, noise_amp)
obj_func_2 = MaxK(n_def, s_exact, noise_amp)
'''
max_function_evals = 10000
'''
max_function_evals = 500
x_0_ = np.random.rand(n_def)
print('shape of x_0_: ', len(x_0_))
R_ = 10
r_ = 1e-3

# Define the comparison oracle.

p.n = 200

print(pycutest.problem_properties('ROSENBR'))
oracle = Oracle_pycutest(p.objcons)

## TODO: oracle = Oracle(p)

# GLDOptimizer instance.
# def __init__(self, defined_func, x_0, R, r, function_budget).

stp1 = GLDOptimizer(oracle, obj_func_1, x_0_, R_, r_, max_function_evals)

## TODO: stp1 = GLDOptimizer(oracle, p, x_0_, R_, r_, max_function_evals)
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
# ---------
