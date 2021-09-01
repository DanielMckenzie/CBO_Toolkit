import cma
import numpy as np
import pandas as pd
from ExampleCode.benchmarkfunctions import SparseQuadratic, MaxK
from matplotlib import pyplot as plt
from ExampleCode.base import BaseOptimizer




# ---------
"""
print('sample invoke.')
# sample invocation.
n = 20  # problem dimension.
s_exact = 200  # true sparsity.
noiseamp = 0.001  # noise amplitude.
# initialize objective function.
obj_func_1 = SparseQuadratic(n, s_exact, noiseamp)

# initialize.
x0 = np.random.randn(n)
options = {'CMA_diagonal': n, 'seed': n, 'verb_time': 0}
sigma = 0.5

# res = cma.fmin(obj_func_1, [0.1] * 10, 0.5, options)
res = cma.fmin(obj_func_1, x0, sigma, options)
print('\n')
print('\n')
print('res: ', res[-1])

solution = res[0]
print('SOLUTION: ', solution)
#print('dimension of solution: ', solution.shape)
'''
obj_func_1: objective function.
x0: initial point.
0.5: sigma (i.e. step - size).
options: dictionary with more options.
    CMA_diagonal: 
'''
# cma.plot()
# res.show()
# cma.savefig('/Users/isha_slavin/Downloads/myfirstrun.png')
# logger = cma.CMADataLogger()
# logger.plot()
# logger.savefig('/Users/isha_slavin/Downloads/fig325.png')
# logger.closefig()

logger = res[-1]  # the CMADataLogger
logger.load()
#plt.semilogy(logger.f[:, 0], logger.f[:, 5])
plt.plot(logger.f[:, 1], logger.f[:, 5])
#plt.plot(iabscissa=1)
plt.show()

print('\n')
print('function evaluations....')
print(logger.f[:, 1])
print('last func eval (i.e. total # of function evaluations....')
print(logger.f[:, 1][-1])

max_function_evals = 10000
"""

# in my class, I can't have a step function.
# actually, I can have a step function but it will literally only have ONE step....
# after I write this class, I need to write it myself....

class cmaOptimizer_pythonFunction(BaseOptimizer):

    def __init__(self, defined_func, x_0, sigma_, options_, function_budget):
        super().__init__()
        self.function_evals = 0
        self.defined_func = defined_func
        self.x_0 = x_0
        self.sigma = sigma_
        self.options = options_
        self.function_budget = function_budget
        # self.f_vals = []
        # self.list_of_xt = []

        # print('hi.')

    def step(self):

        # print('hey.')
        res = cma.fmin(self.defined_func, self.x_0, self.sigma, self.options)
        solution = res[0]
        logger = res[-1]  # the CMADataLogger.
        logger.load()
        # last iteration....
        number_func_evals = logger.f[:, 1][-1]

        """
        if self.reachedFunctionBudget(self.function_budget, number_func_evals):
            # number of function evaluations is higher than the function budget.
            # solution should remain the same I guess....
            '''
            # return solution, list of all function values, termination (which will be False here).
            return x_t, self.f_vals, False
            '''
            # solution: solution.
            x_t = solution
            # self.f_vals: logger.f[:, 5]
            self.f_vals = logger.f[:number_func_evals, 5]
            return x_t, self.f_vals, 'B'
        """

        x_t = solution
        f_vals = logger.f[:, 5]
        func_evals = logger.f[:, 1]
        return x_t, f_vals, func_evals, 'B'


# ---------
# invocation.
# def __init__(self, defined_func, x_0, sigma_, options_, function_budget).
# objective function.
print('sample invoke.')
# sample invocation.
n = 20  # problem dimension.
s_exact = 200  # true sparsity.
noiseamp = 0.001  # noise amplitude.
# initialize objective function.
obj_func_1 = SparseQuadratic(n, s_exact, noiseamp)
# initialize.
x0 = np.random.randn(n)
sigma = 0.5
max_function_evals = 10000
options = {'CMA_diagonal': n, 'seed': n, 'verb_time': 0, 'maxfevals': max_function_evals}

# call.
stp1 = cmaOptimizer_pythonFunction(obj_func_1, x0, sigma, options, max_function_evals)
# step.
termination = False
prev_evals = 0
while termination is False:
    # optimization step.
    solution, func_values, f_evals, termination = stp1.step()
# output solution.
print('\n')
print('SOLUTION: ')
print(solution)
# plot.
plt.plot(f_evals, func_values)
plt.show()





'''
so to build this as a CLASS, I somehow need to keep track of the # of oracle queries.
to do that, I need to know how CMA works, not just call it.
RES is the solution.... I think.
'''

# From the DOCUMENTATION McKenzie sent - CMA paper....
'''
Initialize distribution parameters θ(0) 
For generation g = 0,1,2,...
    Sample λ independent points from distribution P 􏰁x|θ(g) 􏰂 → x1 , . . . , xλ 
    Evaluate the sample x1,...,xλ on f
Update parameters θ(g+1) = Fθ(θ(g),(x1,f(x1)),...,(xλ,f(xλ))) 
Break, if termination criterion met.
'''





"""
Now, I am going to build CMA without using the Python function but build it myself, with function values, iterations,
calling the Oracle, etc.
"""
# let's start off by writing it as a FUNCTION that can take in another function, and optimize it using ORACLE.

