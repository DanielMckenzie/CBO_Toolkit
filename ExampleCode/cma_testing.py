import cma
import numpy as np
import pandas as pd
from ExampleCode.benchmarkfunctions import SparseQuadratic, MaxK
from matplotlib import pyplot as plt
from ExampleCode.base import BaseOptimizer
import math
from numpy import linalg as LA




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

'''
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
            
            # return solution, list of all function values, termination (which will be False here).
            return x_t, self.f_vals, False
            
            # solution: solution.
            x_t = solution
            # self.f_vals: logger.f[:, 5]
            self.f_vals = logger.f[:number_func_evals, 5]
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
# plt.plot(f_evals, func_values)

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
    # stopeval = 1e3*(N**2)  # max number of function evaluations.
    stopeval = 100000  # max number of function evaluations.

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

    # increment at each step.
    f_evaluations = []

    # main while loop.
    while counteval < stopeval:

        # generate & evaluate lambda offspring.
        for k in range(cma_lambda):
            print(k)
            # standard normally distributed vector.
            rand_col = np.random.rand(N)
            # print(rand_col)
            arz[:, k] = rand_col.transpose()
            # print(arz[:, k].shape)
            B_D = np.dot(B, D)
            new_col = arz[:, k].reshape((N, 1))
            arx_right_mult = np.dot(B_D, new_col)
            print('B: ', B)
            print('D: ', D)
            print('B_D: ', B_D)
            print('new_col: ', new_col)
            print('arx_right_mult:')
            print(arx_right_mult)
            arx_right = np.array([element[0] * cma_sigma for element in arx_right_mult])
            print('arx_right:')
            print(arx_right)
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
            print(xmean.shape)
            if xmean.shape == (N, 1):
                xmean = xmean.reshape(N)
                print('new shape: ', xmean.shape)
            arx_col = xmean + arx_right
            print('xmean: ')

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
        # rewrite so sort is using Oracle queries.
        # possibility: use bubbleSort.
        # set mu equal to 1.
        # arindex[:mu] is the index of the best value (since mu = 1).
        # use ORACLE to find best value (f(x) is lowest).
        # sort by fitness & compute weighted mean into xmean.
        arfitness_dict = dict()
        for i in range(len(arfitness)):
            arfitness_dict[i] = arfitness[i]
        print('before sorted: ', arfitness_dict)
        arfitness_dict = dict(sorted(arfitness_dict.items(), key=lambda x: x[1], reverse=False))
        print('after sorted: ', arfitness_dict)
        print('INDICES: ', arfitness_dict.keys())
        print('VALUES: ', arfitness_dict.values())
        arfitness = list(arfitness_dict.values())
        arindex = list(arfitness_dict.keys())
        print('arindex TESTING....')
        print(arindex[:mu])
        # print(arfitness)
        # print(type(arindex))
        xmean_arx = arx[:, arindex[:mu]]
        # print(xmean_arx)
        xmean = np.dot(xmean_arx, weights)
        print('xmean: ')
        print(xmean)
        zmean_arz = arz[:, arindex[:mu]]
        zmean = np.dot(zmean_arz, weights)
        print('zmean: ')
        print(zmean)

        # add the value to f_evaluations.
        # at this increment, arx[:, arindex[0]] is the current xmin.
        f_evaluations.append(strfitnessfct(arx[:, arindex[0]]))

        '''
        3.
        '''
        # cumulation: update evolution paths.
        ps = (1 - cs) * ps + (np.sqrt(cs * (2 - cs) * mueff)) * (np.dot(B, zmean))
        print('ps: ')
        print(ps)
        ## TODO: hsig = LA.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * counteval / cma_lambda)) / chiN < 1.4+2 / (N+1)
        hsig = LA.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * counteval / cma_lambda)) / chiN - 1.4 + 2 / (N + 1)
        '''
        *********
        ASK ABOUT HSIG FORMULA - what is the < symbol actually supposed to be.
        for now: I replaced "<" with "-". otherwise, the value becomes a BOOLEAN instead of a FLOAT.
        *********
        '''
        print('hsig: ')
        print(hsig)
        pc = (1-cc) * pc + hsig * np.sqrt(cc*(2-cc)*mueff) * (np.dot(np.dot(B, D), zmean))
        print('pc: ')
        print(pc)

        '''
        4. 
        '''
        # adapt covariance matrix C.
        """
    adapt covariance matrix C.
        C =  (1-c1-cmu) * C ...
            + c1 * (pc*pc’ ... % plus rank one update
                + (1-hsig) * cc*(2-cc) * C) ... % minor correction
                    + cmu ... % plus rank mu update * (B*D*arz(:,arindex(1:mu))) ...
                        * diag(weights) * (B*D*arz(:,arindex(1:mu)))’;
        """
        first_line_full = (1 - c1 - cmu) * C
        print('first_line_full:')
        print(first_line_full)
        second_line_one = c1
        second_line_two = (np.dot(pc, np.transpose(pc)))
        third_line = (1 - hsig) * cc * (2 - cc) * C
        second_line_full = second_line_one * (second_line_two + third_line)
        print('second_line_full:')
        print(second_line_full)
        fourth_line = cmu
        fifth_line_one = np.dot(B, D)
        print('fifth_line_one:')
        print(fifth_line_one)
        fifth_line_two = arz[:, arindex[:mu]]
        fifth_line = np.dot(fifth_line_one, fifth_line_two)
        print('fifth_line:')
        print(fifth_line)
        sixth_line_one = np.diag(np.transpose(weights)[0])
        print('weights transposed:')
        print(np.transpose(weights)[0])
        print('sixth_line_one:')
        print(sixth_line_one)
        # print(weights)
        sixth_line_two = np.transpose(fifth_line)
        sixth_line = np.dot(np.dot(fifth_line, sixth_line_one), sixth_line_two)
        fourth_line_full = fourth_line * sixth_line
        '''
        add first_line_full, second_line_full, fourth_line_full (where 4th_line_full = 4th_line * 5th_line * 6th_line).
        '''
        C = first_line_full + second_line_full + fourth_line_full
        print('\n')
        print('*********')
        print('C....')
        print(C)
        print('*********')
        print('shape of C: ', C.shape)
        print('\n')

        '''
        5. 
        '''
        # adapt step-size sigma.
        cma_sigma = cma_sigma * math.exp((cs / damps) * (LA.norm(ps) / chiN - 1))
        print('sigma: ', cma_sigma)

        '''
        6.
        '''
        # update B and D from C.
        # look into C more --> make sure it's right.
        # also LA.eig() --> make sure B and D are the correct matrices. (Same as MATLAB implementation.)
        ## TODO: if counteval - eigeneval > cma_lambda / (cone + cmu) / N / 10:
        if counteval - eigeneval > cma_lambda / cmu / N / 10:
            eigeneval = counteval
            C = np.triu(C) + np.transpose(np.triu(C, 1))
            D, B = LA.eig(C)
            """
            print('B: ')
            print(B)
            B = np.diag(B)
            print('D....')
            print(D)
            print(np.diag(D))
            D = np.diag(np.sqrt(np.absolute(np.diag(D))))
            print('C: ')
            print(C)
            print(C.shape)
            print('D: ')
            print(D)
            print(D.shape)
            """
            D = np.diag(np.sqrt(D))

        '''
        7.
        '''
        # break, if fitness is good enough.
        if arfitness[0] <= stopfitness:
            break

        '''
        8.
        '''
        # escape flat fitness, or better terminate.
        if arfitness[0] == arfitness[math.ceil(0.7 * cma_lambda)]:
            cma_sigma = cma_sigma * math.exp(0.2 + cs/damps)
            print('WARNING: flat fitness, consider reformulating the objective.')
        print(str(counteval) + ': ' + str(arfitness[0]))

        # counteval += 100000000000
    """
    break out of WHILE loop.
    """
    # end of WHILE loop; do print ending statements.
    print('\n')
    print('arz: ')
    print(arz)
    print('arx: ')
    print(arx)
    print('\n')

    '''
    1.
    '''
    # final message.
    print(str(counteval) + ': ' + str(arfitness[0]))
    xmin = arx[:, arindex[0]]
    print('xmin: ', xmin)
    return xmin, f_evaluations


# cma_testing.


# ----------------------------------------------------
# INVOCATION.
print('\n')
print('\n')
print('********************************************')
print('cma testing....')
minimizer, function_evaluations = CMA_ES()
print('\n')
print('xmin: ', minimizer)

# add feature to CMA - list of function values so we can graph and verify that it is decreasing.
# list of xmin values at each iteration of the while loop.
# future work -- Oracle generates wrong value 1 every 10 times.

'''
PLOT the function evaluations.
'''
print('function evaluations: ', function_evaluations)
#plt.plot(function_evaluations)
#plt.ylabel('func evals')
#plt.xlabel('iters')
#plt.show()

f_ev = [6.921865582968084, 4.606006914446414, 5.749723058734074, 5.146652787541188, 6.878170419649808, 6.675132477168817, 6.752000694148107, 6.705644025909386, 4.763693924969436, 5.3701349609275, 3.969398874804226, 4.32026851024463, 4.717311794599184, 5.386846948117601, 5.926757835773149, 4.967397324920249, 5.430952090110869, 6.432130048187147, 5.384342275916051, 5.541322774091719, 5.07675679700658, 5.281647830392805, 4.098167556700788, 4.338078657353882, 5.042412676162208, 4.334221762152559, 4.192930968518929, 3.9421854068347475, 3.0363511339792053, 3.780066753770837, 4.226358875386052, 4.011370529955889, 3.15555976337556, 3.5404866514044246, 3.647046755570129, 3.6759619419660132, 3.112221716316473, 3.272588997433446, 2.88751586799547, 2.6135849130956443, 2.3259922560660047, 2.438741208830235, 2.53443219556753, 1.9956211958340395, 2.0972258768009184, 1.759219618188357, 1.7795798395774423, 1.5779569878006088, 1.674372222753495, 1.5750542525228062, 1.305570603863326, 1.1593329125531144, 1.2803245622581094, 1.1402105770750575, 1.279594424971726, 1.2083246997318868, 1.3169314811413713, 1.278155737839891, 1.0093335390051812, 0.8347157422619376, 0.751602199962266, 0.8047493771705486, 0.9160828258907148, 0.8292549907542028, 0.8885368395585922, 0.9861607459434669, 0.9404222685575038, 0.863388844431041, 1.0074968365818555, 1.227039796960248, 1.5017288427242892, 1.4876713514951128, 1.263458299091122, 1.1040890200287383, 1.0531377180213037, 0.8680547232668385, 0.9833447484554586, 0.7985716004688523, 0.6442643520953752, 0.7555683769409218, 0.9117643271874111, 0.9203862491360525, 0.8151583731941624, 0.71364516363128, 0.6953880320349936, 0.6585525988141574, 0.5723624517275144, 0.47779147027444846, 0.4640864185705946, 0.4355863073932045, 0.49681256394452383, 0.3424981796171862, 0.34186314800935913, 0.27165640549593845, 0.21594787296616766, 0.22602986045473392, 0.3019531533694716, 0.3320053265829124, 0.2982742245810271, 0.344235297269408]
print(len(f_ev))
plt.plot(f_ev)
plt.show()



