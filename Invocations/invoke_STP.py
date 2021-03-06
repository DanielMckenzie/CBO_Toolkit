# invokes the STP algorithm class.
import sys
sys.path.append('..')
from Algorithms.stp_optimizer import STPOptimizer
from ExampleCode.benchmarkfunctions import SparseQuadratic, MaxK
from matplotlib import pyplot as plt
import numpy as np

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
function_budget = 100
# initial x_0.
x_0 = np.random.randn(n)
# stp instance.
stp1 = STPOptimizer(direction_vector_type, x_0, n, a_k, obj_func, function_budget)
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


