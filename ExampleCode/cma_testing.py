import cma
import numpy as np
import pandas as pd
from ExampleCode.benchmarkfunctions import SparseQuadratic, MaxK
from matplotlib import pyplot as plt
from ExampleCode.base import BaseOptimizer
import math




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
# actually, I can have a step function but it will literally only have 1 step.

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
n = 5  # problem dimension.
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
'''
plt.show()
'''





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


# writing a new CMA algorithm based on the MatLab code....
# writing a FUNCTION CMA....

def CMA_ES():

    """
    initialization.
    """

    '''
    1. 
    '''
    # user defines input parameters (so actually, take this in as parameters of the class you write).
    N = 10  # number of objective variables - problem dimension.
    strfitnessfct = SparseQuadratic(N, 200, .001)  # name of objective function (SparseQuadratic).
    '''
    xmean = np.random.randn(N, 1)  # objective variables initial point (like x_0).
    '''
    xmean = np.random.randn(N)  # objective variables initial point (like x_0).
    cma_sigma = 0.5
    stopfitness = 1e-10  # minimization.
    stopeval = 1e3*(N**2)  # max number of function evaluations.

    '''
    2.
    '''
    # strategy parameter setting: selection.
    cma_lambda = 4 + math.floor(3*math.log(N))
    mu = cma_lambda / 2
    mu = int(mu)
    '''
    weights = math.log(mu+1/2)-math.log(1:mu)'.
    '''
    # 'weights' is saying to make an array of length MU.
    # whatever the value of mu is.... make an array of that length.
    # each value of that array will have the value math.log(mu+1/2).
    # each value of the next array will be math.log(x from 1 to mu, inclusive).
    left_array = np.empty([mu, 1])
    # print(left_array.shape)
    right_array = np.empty([mu, 1])
    # print(right_array.shape)
    for i in range(mu):
        left_array[i, 0] = math.log(mu+1/2)
        right_array[i, 0] = math.log(i+1)
    """
    print('left_array: ')
    print(left_array)
    print('right_array: ')
    print(right_array)
    """
    weights = left_array - right_array
    print('weights: ')
    print(weights)
    mu = math.floor(mu)  # number of points for recombination.
    weights = weights/np.sum(weights)
    print('normalized weights: ')
    print(weights)
    # mueff calculation.
    top_expression = np.sum(weights) ** 2
    # print('top expression: ', top_expression)
    squared_weights = [x**2 for x in weights]
    bottom_expression = np.sum(squared_weights)
    # print('bottom expression: ', bottom_expression)
    mueff = top_expression / bottom_expression
    print('mueff: ', mueff)  # variance - effective size of mu.

    '''
    3. 
    '''
    # strategy parameter setting: adaptation.
    cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)  # time constant for cumulation for C.
    print('cc: ', cc)
    cs = (mueff+2) / (N+mueff+5)  # t-const for cumulation for sigma control.
    print('cs: ', cs)
    c1 = 2 / (((N + 1.3) ** 2) + mueff)  # learning rate for rank-1 update of C.
    print('c1: ', c1)
    cmu = 2 * (mueff-2+1/mueff) / ((N+2)**2+2*mueff/2)  # rank-mu update.
    print('cmu: ', cmu)
    damps = 1 + 2*max(0, np.sqrt((mueff-1)/(N+1))-1) + cs
    print('damps: ', damps)

    '''
    4.
    '''
    # initialize dynamic (internal) strategy parameters & constants.
    pc = np.zeros([N, 1])
    ps = np.zeros([N, 1])
    # print(pc, ps)
    B = np.eye(N)
    D = np.eye(N)
    left_mult = np.dot(B, D)
    right_mult = np.dot(B, D).transpose()
    C = np.dot(left_mult, right_mult)
    # print('B: ', B)
    # print('C: ', C)
    eigeneval = 0
    chiN = N**0.5 * (1 - 1 / (4 * N) + 1 / (21 * N**2))
    print('chiN: ', chiN)


    """
    generation loop.
    """

    '''
    1.
    '''
    counteval = 0
    arz = np.empty([N, cma_lambda])
    arx = np.empty([N, cma_lambda])
    arfitness = np.empty(cma_lambda)
    while counteval < stopeval:

        # generate & evaluate lambda offspring.
        for k in range(cma_lambda):
            # standard normally distributed vector.
            rand_col = np.random.rand(N)
            # print(rand_col)
            arz[:, k] = rand_col.transpose()
            # print(arz[:, k].shape)
            B_D = np.dot(B, D)
            new_col = arz[:, k].reshape((N, 1))
            arx_right_mult = np.dot(B_D, new_col)
            # print(arx_right_mult)
            arx_right = np.array([element[0] * cma_sigma for element in arx_right_mult])
            # print('RIGHT: ', arx_right)
            arx_right_trans = arx_right.reshape(N, 1)
            # print(arx_right_trans)
            # print('arx_right: ', arx_right)
            # print('x_o: ')
            # print(xmean)
            # print('this....')
            # print(xmean+arx_right_trans)
            # print('\n')
            # we need the sum of xmean and arx_right.
            arx_col = xmean + arx_right
            # print('ARX COL: ', arx_col)
            """
            # for element in arx_right_mult:
            #     print(element[0] * cma_sigma)
            # print('arx_right_mult: ', arx_right_mult[0])
            # arx_right_mult = arx_right_mult.ravel()
            '''
            arx[:, k] = xmean + cma_sigma * arx_right_mult
            '''
            # arx[:, k] = xmean+arx_right_trans
            """
            arx[:, k] = arx_col.transpose()
            arfitness[k] = strfitnessfct(arx[:, k])
            counteval += 1

        '''
        2.
        '''
        # sort by fitness & compute weighted mean into xmean.




        counteval += 100000000000
    print('\n')
    print('arz: ')
    print(arz)
    print('arx: ')
    print(arx)
    print()








# INVOCATION.
print('\n')
print('\n')
print('********************************************')
print('cma testing....')
CMA_ES()















