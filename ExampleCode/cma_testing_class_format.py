import cma
import numpy as np
import pandas as pd
from ExampleCode.benchmarkfunctions import SparseQuadratic, MaxK
from matplotlib import pyplot as plt
from ExampleCode.base import BaseOptimizer
import math
from numpy import linalg as LA
#
# # ---------
# """
# print('sample invoke.')
# # sample invocation.
# n = 20  # problem dimension.
# s_exact = 200  # true sparsity.
# noiseamp = 0.001  # noise amplitude.
# # initialize objective function.
# obj_func_1 = SparseQuadratic(n, s_exact, noiseamp)
#
# # initialize.
# x0 = np.random.randn(n)
# options = {'CMA_diagonal': n, 'seed': n, 'verb_time': 0}
# sigma = 0.5
#
# # res = cma.fmin(obj_func_1, [0.1] * 10, 0.5, options)
# res = cma.fmin(obj_func_1, x0, sigma, options)
# print('\n')
# print('\n')
# print('res: ', res[-1])
#
# solution = res[0]
# print('SOLUTION: ', solution)
# #print('dimension of solution: ', solution.shape)
# '''
# obj_func_1: objective function.
# x0: initial point.
# 0.5: sigma (i.e. step - size).
# options: dictionary with more options.
#     CMA_diagonal:
# '''
# # cma.plot()
# # res.show()
# # cma.savefig('/Users/isha_slavin/Downloads/myfirstrun.png')
# # logger = cma.CMADataLogger()
# # logger.plot()
# # logger.savefig('/Users/isha_slavin/Downloads/fig325.png')
# # logger.closefig()
#
# logger = res[-1]  # the CMADataLogger
# logger.load()
# #plt.semilogy(logger.f[:, 0], logger.f[:, 5])
# plt.plot(logger.f[:, 1], logger.f[:, 5])
# #plt.plot(iabscissa=1)
# plt.show()
#
# print('\n')
# print('function evaluations....')
# print(logger.f[:, 1])
# print('last func eval (i.e. total # of function evaluations....')
# print(logger.f[:, 1][-1])
#
# max_function_evals = 10000
# """
#
# # in my class, I can't have a step function.
# # actually, I can have a step function but it will literally only have 1 step.
#
# '''
# class cmaOptimizer_pythonFunction(BaseOptimizer):
#
#     def __init__(self, defined_func, x_0, sigma_, options_, function_budget):
#         super().__init__()
#         self.function_evals = 0
#         self.defined_func = defined_func
#         self.x_0 = x_0
#         self.sigma = sigma_
#         self.options = options_
#         self.function_budget = function_budget
#         # self.f_vals = []
#         # self.list_of_xt = []
#
#         # print('hi.')
#
#     def step(self):
#
#         # print('hey.')
#         res = cma.fmin(self.defined_func, self.x_0, self.sigma, self.options)
#         solution = res[0]
#         logger = res[-1]  # the CMADataLogger.
#         logger.load()
#         # last iteration....
#         number_func_evals = logger.f[:, 1][-1]
#
#         """
#         if self.reachedFunctionBudget(self.function_budget, number_func_evals):
#             # number of function evaluations is higher than the function budget.
#             # solution should remain the same I guess....
#
#             # return solution, list of all function values, termination (which will be False here).
#             return x_t, self.f_vals, False
#
#             # solution: solution.
#             x_t = solution
#             # self.f_vals: logger.f[:, 5]
#             self.f_vals = logger.f[:number_func_evals, 5]
#         """
#
#         x_t = solution
#         f_vals = logger.f[:, 5]
#         func_evals = logger.f[:, 1]
#         return x_t, f_vals, func_evals, 'B'
#
#
# # ---------
# # invocation.
# # def __init__(self, defined_func, x_0, sigma_, options_, function_budget).
# # objective function.
# print('sample invoke.')
# # sample invocation.
# n = 5  # problem dimension.
# s_exact = 200  # true sparsity.
# noiseamp = 0.001  # noise amplitude.
# # initialize objective function.
# obj_func_1 = SparseQuadratic(n, s_exact, noiseamp)
# # initialize.
# x0 = np.random.randn(n)
# sigma = 0.5
# max_function_evals = 10000
# options = {'CMA_diagonal': n, 'seed': n, 'verb_time': 0, 'maxfevals': max_function_evals}
#
# # call.
# stp1 = cmaOptimizer_pythonFunction(obj_func_1, x0, sigma, options, max_function_evals)
# # step.
# termination = False
# prev_evals = 0
# while termination is False:
#     # optimization step.
#     solution, func_values, f_evals, termination = stp1.step()
# # output solution.
# print('\n')
# print('SOLUTION: ')
# print(solution)
# # plot.
# # plt.plot(f_evals, func_values)
#
# plt.show()
#
# '''
#
# '''
# so to build this as a CLASS, I somehow need to keep track of the # of oracle queries.
# to do that, I need to know how CMA works, not just call it.
# RES is the solution.... I think.
# '''
#
# # From the DOCUMENTATION McKenzie sent - CMA paper....
# '''
# Initialize distribution parameters θ(0)
# For generation g = 0,1,2,...
#     Sample λ independent points from distribution P 􏰁x|θ(g) 􏰂 → x1 , . . . , xλ
#     Evaluate the sample x1,...,xλ on f
# Update parameters θ(g+1) = Fθ(θ(g),(x1,f(x1)),...,(xλ,f(xλ)))
# Break, if termination criterion met.
# '''
#
# """
# Now, I am going to build CMA without using the Python function but build it myself, with function values, iterations,
# calling the Oracle, etc.
# """
#
#
# # let's start off by writing it as a FUNCTION that can take in another function, and optimize it using ORACLE.
#
#
# # writing a new CMA algorithm based on the MatLab code....
# # writing a FUNCTION CMA....
#
# def CMA_ES():
#     """
#     initialization.
#     """
#
#     '''
#     1.
#     '''
#     # user defines input parameters (so actually, take this in as parameters of the class you write).
#     N = 10  # number of objective variables - problem dimension.
#     strfitnessfct = SparseQuadratic(N, 200, .001)  # name of objective function (SparseQuadratic).
#     '''
#     xmean = np.random.randn(N, 1)  # objective variables initial point (like x_0).
#     '''
#     xmean = np.random.randn(N)  # objective variables initial point (like x_0).
#     cma_sigma = 0.5
#     stopfitness = 1e-10  # minimization.
#     # stopeval = 1e3*(N**2)  # max number of function evaluations.
#     stopeval = 1000  # max number of function evaluations.
#
#     '''
#     2.
#     '''
#     # strategy parameter setting: selection.
#     cma_lambda = 4 + math.floor(3 * math.log(N))
#     mu = cma_lambda / 2
#     mu = int(mu)
#     '''
#     weights = math.log(mu+1/2)-math.log(1:mu)'.
#     '''
#     # 'weights' is saying to make an array of length MU.
#     # whatever the value of mu is.... make an array of that length.
#     # each value of that array will have the value math.log(mu+1/2).
#     # each value of the next array will be math.log(x from 1 to mu, inclusive).
#     left_array = np.empty([mu, 1])
#     # print(left_array.shape)
#     right_array = np.empty([mu, 1])
#     # print(right_array.shape)
#     for i in range(mu):
#         left_array[i, 0] = math.log(mu + 1 / 2)
#         right_array[i, 0] = math.log(i + 1)
#     """
#     print('left_array: ')
#     print(left_array)
#     print('right_array: ')
#     print(right_array)
#     """
#     weights = left_array - right_array
#     print('weights: ')
#     print(weights)
#     mu = math.floor(mu)  # number of points for recombination.
#     weights = weights / np.sum(weights)
#     print('normalized weights: ')
#     print(weights)
#     # mueff calculation.
#     top_expression = np.sum(weights) ** 2
#     # print('top expression: ', top_expression)
#     squared_weights = [x ** 2 for x in weights]
#     bottom_expression = np.sum(squared_weights)
#     # print('bottom expression: ', bottom_expression)
#     mueff = top_expression / bottom_expression
#     print('mueff: ', mueff)  # variance - effective size of mu.
#
#     '''
#     3.
#     '''
#     # strategy parameter setting: adaptation.
#     cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)  # time constant for cumulation for C.
#     print('cc: ', cc)
#     cs = (mueff + 2) / (N + mueff + 5)  # t-const for cumulation for sigma control.
#     print('cs: ', cs)
#     c1 = 2 / (((N + 1.3) ** 2) + mueff)  # learning rate for rank-1 update of C.
#     print('c1: ', c1)
#     cmu = 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ** 2 + 2 * mueff / 2)  # rank-mu update.
#     print('cmu: ', cmu)
#     damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (N + 1)) - 1) + cs
#     print('damps: ', damps)
#
#     '''
#     4.
#     '''
#     # initialize dynamic (internal) strategy parameters & constants.
#     pc = np.zeros([N, 1])
#     ps = np.zeros([N, 1])
#     # print(pc, ps)
#     B = np.eye(N)
#     D = np.eye(N)
#     left_mult = np.dot(B, D)
#     right_mult = np.dot(B, D).transpose()
#     C = np.dot(left_mult, right_mult)
#     # print('B: ', B)
#     # print('C: ', C)
#     eigeneval = 0
#     chiN = N ** 0.5 * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))
#     print('chiN: ', chiN)
#
#     """
#     generation loop.
#     """
#
#     '''
#     1.
#     '''
#     counteval = 0
#     arz = np.empty([N, cma_lambda])
#     arx = np.empty([N, cma_lambda])
#     arfitness = np.empty(cma_lambda)
#
#     # increment at each step.
#     f_evaluations = []
#
#     # main while loop.
#     while counteval < stopeval:
#
#         # generate & evaluate lambda offspring.
#         for k in range(cma_lambda):
#             print(k)
#             # standard normally distributed vector.
#             rand_col = np.random.rand(N)
#             # print(rand_col)
#             arz[:, k] = rand_col.transpose()
#             # print(arz[:, k].shape)
#             B_D = np.dot(B, D)
#             new_col = arz[:, k].reshape((N, 1))
#             arx_right_mult = np.dot(B_D, new_col)
#             print('B: ', B)
#             print('D: ', D)
#             print('B_D: ', B_D)
#             print('new_col: ', new_col)
#             print('arx_right_mult:')
#             print(arx_right_mult)
#             arx_right = np.array([element[0] * cma_sigma for element in arx_right_mult])
#             print('arx_right:')
#             print(arx_right)
#             # print('RIGHT: ', arx_right)
#             arx_right_trans = arx_right.reshape(N, 1)
#             # print(arx_right_trans)
#             # print('arx_right: ', arx_right)
#             # print('x_o: ')
#             # print(xmean)
#             # print('this....')
#             # print(xmean+arx_right_trans)
#             # print('\n')
#             # we need the sum of xmean and arx_right.
#             print(xmean.shape)
#             if xmean.shape == (N, 1):
#                 xmean = xmean.reshape(N)
#                 print('new shape: ', xmean.shape)
#             arx_col = xmean + arx_right
#             print('xmean: ')
#
#             # print('ARX COL: ', arx_col)
#             """
#             # for element in arx_right_mult:
#             #     print(element[0] * cma_sigma)
#             # print('arx_right_mult: ', arx_right_mult[0])
#             # arx_right_mult = arx_right_mult.ravel()
#             '''
#             arx[:, k] = xmean + cma_sigma * arx_right_mult
#             '''
#             # arx[:, k] = xmean+arx_right_trans
#             """
#             arx[:, k] = arx_col.transpose()
#             arfitness[k] = strfitnessfct(arx[:, k])
#             counteval += 1
#
#         '''
#         2.
#         '''
#         # rewrite so sort is using Oracle queries.
#         # possibility: use bubbleSort.
#         # set mu equal to 1.
#         # arindex[:mu] is the index of the best value (since mu = 1).
#         # use ORACLE to find best value (f(x) is lowest).
#         # sort by fitness & compute weighted mean into xmean.
#         arfitness_dict = dict()
#         for i in range(len(arfitness)):
#             arfitness_dict[i] = arfitness[i]
#         print('before sorted: ', arfitness_dict)
#         arfitness_dict = dict(sorted(arfitness_dict.items(), key=lambda x: x[1], reverse=False))
#         print('after sorted: ', arfitness_dict)
#         print('INDICES: ', arfitness_dict.keys())
#         print('VALUES: ', arfitness_dict.values())
#         arfitness = list(arfitness_dict.values())
#         arindex = list(arfitness_dict.keys())
#         print('arindex TESTING....')
#         print(arindex[:mu])
#         # print(arfitness)
#         # print(type(arindex))
#         xmean_arx = arx[:, arindex[:mu]]
#         # print(xmean_arx)
#         xmean = np.dot(xmean_arx, weights)
#         print('xmean: ')
#         print(xmean)
#         zmean_arz = arz[:, arindex[:mu]]
#         zmean = np.dot(zmean_arz, weights)
#         print('zmean: ')
#         print(zmean)
#
#         # add the value to f_evaluations.
#         # at this increment, arx[:, arindex[0]] is the current xmin.
#         f_evaluations.append(strfitnessfct(arx[:, arindex[0]]))
#
#         '''
#         3.
#         '''
#         # cumulation: update evolution paths.
#         ps = (1 - cs) * ps + (np.sqrt(cs * (2 - cs) * mueff)) * (np.dot(B, zmean))
#         print('ps: ')
#         print(ps)
#         ## TODO: hsig = LA.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * counteval / cma_lambda)) / chiN < 1.4+2 / (N+1)
#         hsig = LA.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * counteval / cma_lambda)) / chiN - 1.4 + 2 / (N + 1)
#         '''
#         *********
#         ASK ABOUT HSIG FORMULA - what is the < symbol actually supposed to be.
#         for now: I replaced "<" with "-". otherwise, the value becomes a BOOLEAN instead of a FLOAT.
#         *********
#         '''
#         print('hsig: ')
#         print(hsig)
#         pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (np.dot(np.dot(B, D), zmean))
#         print('pc: ')
#         print(pc)
#
#         '''
#         4.
#         '''
#         # adapt covariance matrix C.
#         """
#     adapt covariance matrix C.
#         C =  (1-c1-cmu) * C ...
#             + c1 * (pc*pc’ ... % plus rank one update
#                 + (1-hsig) * cc*(2-cc) * C) ... % minor correction
#                     + cmu ... % plus rank mu update * (B*D*arz(:,arindex(1:mu))) ...
#                         * diag(weights) * (B*D*arz(:,arindex(1:mu)))’;
#         """
#         first_line_full = (1 - c1 - cmu) * C
#         print('first_line_full:')
#         print(first_line_full)
#         second_line_one = c1
#         second_line_two = (np.dot(pc, np.transpose(pc)))
#         third_line = (1 - hsig) * cc * (2 - cc) * C
#         second_line_full = second_line_one * (second_line_two + third_line)
#         print('second_line_full:')
#         print(second_line_full)
#         fourth_line = cmu
#         fifth_line_one = np.dot(B, D)
#         print('fifth_line_one:')
#         print(fifth_line_one)
#         fifth_line_two = arz[:, arindex[:mu]]
#         fifth_line = np.dot(fifth_line_one, fifth_line_two)
#         print('fifth_line:')
#         print(fifth_line)
#         sixth_line_one = np.diag(np.transpose(weights)[0])
#         print('weights transposed:')
#         print(np.transpose(weights)[0])
#         print('sixth_line_one:')
#         print(sixth_line_one)
#         # print(weights)
#         sixth_line_two = np.transpose(fifth_line)
#         sixth_line = np.dot(np.dot(fifth_line, sixth_line_one), sixth_line_two)
#         fourth_line_full = fourth_line * sixth_line
#         '''
#         add first_line_full, second_line_full, fourth_line_full (where 4th_line_full = 4th_line * 5th_line * 6th_line).
#         '''
#         C = first_line_full + second_line_full + fourth_line_full
#         print('\n')
#         print('*********')
#         print('C....')
#         print(C)
#         print('*********')
#         print('shape of C: ', C.shape)
#         print('\n')
#
#         '''
#         5.
#         '''
#         # adapt step-size sigma.
#         cma_sigma = cma_sigma * math.exp((cs / damps) * (LA.norm(ps) / chiN - 1))
#         print('sigma: ', cma_sigma)
#
#         '''
#         6.
#         '''
#         # update B and D from C.
#         # look into C more --> make sure it's right.
#         # also LA.eig() --> make sure B and D are the correct matrices. (Same as MATLAB implementation.)
#         ## TODO: if counteval - eigeneval > cma_lambda / (cone + cmu) / N / 10:
#         if counteval - eigeneval > cma_lambda / cmu / N / 10:
#             eigeneval = counteval
#             C = np.triu(C) + np.transpose(np.triu(C, 1))
#             D, B = LA.eig(C)
#             """
#             print('B: ')
#             print(B)
#             B = np.diag(B)
#             print('D....')
#             print(D)
#             print(np.diag(D))
#             D = np.diag(np.sqrt(np.absolute(np.diag(D))))
#             print('C: ')
#             print(C)
#             print(C.shape)
#             print('D: ')
#             print(D)
#             print(D.shape)
#             """
#             D = np.diag(np.sqrt(D))
#
#         '''
#         7.
#         '''
#         # break, if fitness is good enough.
#         if arfitness[0] <= stopfitness:
#             break
#
#         '''
#         8.
#         '''
#         # escape flat fitness, or better terminate.
#         if arfitness[0] == arfitness[math.ceil(0.7 * cma_lambda)]:
#             cma_sigma = cma_sigma * math.exp(0.2 + cs / damps)
#             print('WARNING: flat fitness, consider reformulating the objective.')
#         print(str(counteval) + ': ' + str(arfitness[0]))
#
#         # counteval += 100000000000
#     """
#     break out of WHILE loop.
#     """
#     # end of WHILE loop; do print ending statements.
#     print('\n')
#     print('arz: ')
#     print(arz)
#     print('arx: ')
#     print(arx)
#     print('\n')
#
#     '''
#     1.
#     '''
#     # final message.
#     print(str(counteval) + ': ' + str(arfitness[0]))
#     xmin = arx[:, arindex[0]]
#     print('xmin: ', xmin)
#     return xmin, f_evaluations
#
#
# # cma_testing.
#
#
# # ----------------------------------------------------
# # INVOCATION.
# print('\n')
# print('\n')
# print('********************************************')
# print('cma testing....')
# minimizer, function_evaluations = CMA_ES()
# print('\n')
# print('xmin: ', minimizer)
#
# # add feature to CMA - list of function values so we can graph and verify that it is decreasing.
# # list of xmin values at each iteration of the while loop.
# # future work -- Oracle generates wrong value 1 every 10 times.
#
# '''
# PLOT the function evaluations.
# '''
# print('function evaluations: ', function_evaluations)
# plt.plot(function_evaluations)
# # plt.ylabel('func evals')
# # plt.xlabel('iters')
# plt.show()
#
#
# # f_ev = [6.921865582968084, 4.606006914446414, 5.749723058734074, 5.146652787541188, 6.878170419649808, 6.675132477168817, 6.752000694148107, 6.705644025909386, 4.763693924969436, 5.3701349609275, 3.969398874804226, 4.32026851024463, 4.717311794599184, 5.386846948117601, 5.926757835773149, 4.967397324920249, 5.430952090110869, 6.432130048187147, 5.384342275916051, 5.541322774091719, 5.07675679700658, 5.281647830392805, 4.098167556700788, 4.338078657353882, 5.042412676162208, 4.334221762152559, 4.192930968518929, 3.9421854068347475, 3.0363511339792053, 3.780066753770837, 4.226358875386052, 4.011370529955889, 3.15555976337556, 3.5404866514044246, 3.647046755570129, 3.6759619419660132, 3.112221716316473, 3.272588997433446, 2.88751586799547, 2.6135849130956443, 2.3259922560660047, 2.438741208830235, 2.53443219556753, 1.9956211958340395, 2.0972258768009184, 1.759219618188357, 1.7795798395774423, 1.5779569878006088, 1.674372222753495, 1.5750542525228062, 1.305570603863326, 1.1593329125531144, 1.2803245622581094, 1.1402105770750575, 1.279594424971726, 1.2083246997318868, 1.3169314811413713, 1.278155737839891, 1.0093335390051812, 0.8347157422619376, 0.751602199962266, 0.8047493771705486, 0.9160828258907148, 0.8292549907542028, 0.8885368395585922, 0.9861607459434669, 0.9404222685575038, 0.863388844431041, 1.0074968365818555, 1.227039796960248, 1.5017288427242892, 1.4876713514951128, 1.263458299091122, 1.1040890200287383, 1.0531377180213037, 0.8680547232668385, 0.9833447484554586, 0.7985716004688523, 0.6442643520953752, 0.7555683769409218, 0.9117643271874111, 0.9203862491360525, 0.8151583731941624, 0.71364516363128, 0.6953880320349936, 0.6585525988141574, 0.5723624517275144, 0.47779147027444846, 0.4640864185705946, 0.4355863073932045, 0.49681256394452383, 0.3424981796171862, 0.34186314800935913, 0.27165640549593845, 0.21594787296616766, 0.22602986045473392, 0.3019531533694716, 0.3320053265829124, 0.2982742245810271, 0.344235297269408]
# # f_ev = [10.634440693680455, 10.054116243587439, 11.213083624749789, 10.753257669857017, 11.90885352226613, 13.456128557491114, 14.389607582421165, 14.947554524189595, 16.627304078585464, 22.225357056648882, 26.46245837323517, 33.58373161631505, 45.30253742959009, 57.925891145096884, 84.36087084458912, 114.16189464065427, 213.02670942360362, 283.502323874285, 561.2854564134111, 1233.2501731471114, 1772.096521019566, 3482.567952581177, 6582.4848333682085, 10289.453901096422, 22435.10620694921, 63965.29593091551, 171201.92795423116, 599555.9499254589, 2248787.862204783, 7056538.6796283545, 26341405.448614016, 124829650.5605721, 527228421.9859338, 3358261392.103911, 20050605124.618874, 133156314728.57445, 885915503953.1013, 5278418154647.059, 32631397574981.023, 271153011878331.1, 1692232825647946.0, 1.083251864492541e+16, 6.44116751652657e+16, 5.757537300551154e+17, 4.02059725209169e+18, 2.343590900988509e+19, 1.612396632884193e+20, 1.0539442580073564e+21, 8.071268659972886e+21, 4.7325176418765665e+22, 3.64052334067067e+23, 2.5247117113460993e+24, 1.7656190482722673e+25, 1.3131533919135606e+26, 9.194539107832893e+26, 8.665093150786199e+27, 5.975612948119654e+28, 3.6507506201282746e+29, 2.956810017106016e+30, 1.7406226204866338e+31, 1.5071768027498613e+32, 1.124260585367634e+33, 9.938906040851879e+33, 7.922244476475101e+34, 6.2111345524089795e+35, 2.9535597734812265e+36, 3.140723401011637e+37, 3.0965886279705855e+38, 2.1925593968295868e+39, 1.3050334278482557e+40, 1.3093759182749892e+41, 9.169387056007437e+41, 6.597200625465505e+42, 3.623693663340894e+43, 3.3925210442287648e+44, 2.5725027561181607e+45, 1.3430104923714543e+46, 6.443077787388553e+46, 4.882073639188245e+47, 4.402505054152611e+48, 2.587394800538272e+49, 2.2868114845269415e+50, 1.1962464477604263e+51, 7.206630869230548e+51, 4.1971339136882584e+52, 3.88744696761764e+53, 2.960758259985091e+54, 1.5707495990916627e+55, 7.601130303916525e+55, 8.134276601218627e+56, 5.385755091263239e+57, 3.780515473070021e+58, 2.820393333286114e+59, 1.8346949074706156e+60, 1.51670824263491e+61, 1.3084932316448102e+62, 9.025640170916268e+62, 8.72942197148936e+63, 7.005628958050709e+64, 6.546126875117884e+65]
# # f_ev = [6.7530197585018, 6.704701891277875, 5.357049803077245, 5.147870198025081, 4.349664472014985, 4.166097515726941, 3.740557246216642, 3.8164765219666332, 3.7968056678379716, 4.521524737180279, 3.8621683836451117, 3.1800148549591767, 3.70349981222148, 3.9578096270998087, 4.342260134699917, 3.9429585655563515, 3.79334973327553, 3.529424106519385, 4.14811440362742, 4.151448631902823, 3.5248253880778426, 3.638942247050223, 3.8705998197268014, 3.2510774595613854, 3.7426166202987834, 4.354139181060725, 3.7580542310452913, 3.810830615753196, 4.025493202986743, 3.6334679890596706, 3.454825564578995, 3.2844172332006205, 3.4595291988314, 4.074295454730607, 4.38076335452761, 4.0992375248527715, 4.21158079599497, 3.789760277829843, 3.700552691234354, 3.474586281935955, 2.956852133610726, 2.7947960105077247, 2.682631689983766, 2.422230516142305, 2.5208653106215273, 2.4167119653454185, 2.266696523100449, 2.1694250051341593, 2.045390346726828, 2.097109364959359, 1.9958033027386266, 2.0316848184009197, 2.10137451948386, 1.9305389773711479, 1.7949844334001979, 1.7380084645908453, 1.6772381697713008, 1.7180646955111192, 1.6486623090754686, 1.604434682090621, 1.457519111726408, 1.569015210717826, 1.4713007526259112, 1.4694184147796177, 1.4849306852317925, 1.4355961065406975, 1.5091595383493566, 1.4174050876248223, 1.4084325199279046, 1.3597622660211817, 1.4347756702699366, 1.4993226339078278, 1.5366227723242138, 1.522024031468896, 1.463111789967382, 1.4224959352620026, 1.3522328912527097, 1.418440929442633, 1.3987681157164706, 1.3810238089297096, 1.3555606734914643, 1.3672736656973699, 1.3868414976810253, 1.4195810683349224, 1.3870084584738436, 1.3460940096483383, 1.3605698364241505, 1.3431860892635588, 1.3296560693224728, 1.3309424197904522, 1.330343821600663, 1.3345384398444267, 1.3042826692317295, 1.294166830083621, 1.2734345326862078, 1.2415649047760497, 1.2082755575449309, 1.229966275723925, 1.159194306146981, 1.0792055437297614]
#
# # print(len(f_ev))
# # plt.plot(f_ev)
# # plt.show()

# now, we will re-write in CLASS format.
"""
************************************************************************************************************************
"""


# class format.
class CMAoptimizer(BaseOptimizer):

    def __init__(self, objective_function, x0, cma_lam, cma_mu, sigma, stop_fitness, stop_eval):
        """
        initialization.
        """
        '''
        1. 
        '''
        # user defines input parameters (so actually, take this in as parameters of the class you write).
        N = x0.shape[0]  # number of objective variables - problem dimension.
        strfitnessfct = objective_function  # name of objective function (SparseQuadratic).
        '''
        xmean = np.random.randn(N, 1)  # objective variables initial point (like x_0).
        '''
        xmean = x0  # objective variables initial point (like x_0).
        cma_sigma = sigma
        stopfitness = stop_fitness  # minimization.
        # stopeval = 1e3*(N**2)  # max number of function evaluations.
        stopeval = stop_eval  # max number of function evaluations.
        '''
        2.
        '''
        # strategy parameter setting: selection.
        cma_lambda = cma_lam
        mu = cma_mu
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
            left_array[i, 0] = math.log(mu + 1 / 2)
            right_array[i, 0] = math.log(i + 1)
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
        weights = weights / np.sum(weights)
        print('normalized weights: ')
        print(weights)
        # mueff calculation.
        top_expression = np.sum(weights) ** 2
        # print('top expression: ', top_expression)
        squared_weights = [x ** 2 for x in weights]
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
        cs = (mueff + 2) / (N + mueff + 5)  # t-const for cumulation for sigma control.
        print('cs: ', cs)
        c1 = 2 / (((N + 1.3) ** 2) + mueff)  # learning rate for rank-1 update of C.
        print('c1: ', c1)
        cmu = 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ** 2 + 2 * mueff / 2)  # rank-mu update.
        print('cmu: ', cmu)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (N + 1)) - 1) + cs
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
        chiN = N ** 0.5 * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))
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
        '''
        define variables used in step() method.
        '''
        self.N = N
        self.strfitnessfct = strfitnessfct
        self.xmean = xmean
        self.cma_sigma = cma_sigma
        self.stopfitness = stop_fitness
        self.stopeval = stopeval
        self.cma_lambda = cma_lambda
        self.mu = mu
        self.weights = weights
        self.mueff = mueff
        self.cc = cc
        self.cs = cs
        self.c1 = c1
        self.cmu = cmu
        self.damps = damps
        self.pc = pc
        self.ps = ps
        self.B = B
        self.D = D
        self.C = C
        self.eigeneval = eigeneval
        self.chiN = chiN
        self.counteval = counteval
        self.arz = arz
        self.arx = arx
        self.arfitness = arfitness
        self.f_evaluations = f_evaluations

    def step(self):

        # generate & evaluate lambda offspring.
        for k in range(self.cma_lambda):
            #print(k)
            # standard normally distributed vector.
            rand_col = np.random.rand(self.N)
            # print(rand_col)
            self.arz[:, k] = rand_col.transpose()
            # print(arz[:, k].shape)
            ## TODO: B_D = np.dot(self.B, self.D)
            B_D = np.identity(self.N)
            new_col = self.arz[:, k].reshape((self.N, 1))
            arx_right_mult = np.dot(B_D, new_col)
            #print('B: ', self.B)
            #print('D: ', self.D)
            #print('B_D: ', B_D)
            #print('new_col: ', new_col)
            #print('arx_right_mult:')
            #print(arx_right_mult)
            arx_right = np.array([element[0] * self.cma_sigma for element in arx_right_mult])
            #print('arx_right:')
            #print(arx_right)
            # print('RIGHT: ', arx_right)
            arx_right_trans = arx_right.reshape(self.N, 1)
            # print(arx_right_trans)
            # print('arx_right: ', arx_right)
            # print('x_o: ')
            # print(xmean)
            # print('this....')
            # print(xmean+arx_right_trans)
            # print('\n')
            # we need the sum of xmean and arx_right.
            #print(self.xmean.shape)
            if self.xmean.shape == (self.N, 1):
                self.xmean = self.xmean.reshape(self.N)
                #print('new shape: ', self.xmean.shape)
            arx_col = self.xmean + arx_right
            #print('xmean: ')

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
            self.arx[:, k] = arx_col.transpose()
            self.arfitness[k] = self.strfitnessfct(self.arx[:, k])
            self.counteval += 1

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
        for ij in range(len(self.arfitness)):
            arfitness_dict[ij] = self.arfitness[ij]
        #print('before sorted: ', arfitness_dict)
        arfitness_dict = dict(sorted(arfitness_dict.items(), key=lambda x: x[1], reverse=False))
        #print('after sorted: ', arfitness_dict)
        #print('INDICES: ', arfitness_dict.keys())
        #print('VALUES: ', arfitness_dict.values())
        self.arfitness = list(arfitness_dict.values())
        arindex = list(arfitness_dict.keys())
        #print('arindex TESTING....')
        #print(arindex[:self.mu])
        # print(arfitness)
        # print(type(arindex))
        xmean_arx = self.arx[:, arindex[:self.mu]]
        # print(xmean_arx)
        self.xmean = np.dot(xmean_arx, self.weights)
        #print('xmean: ')
        #print(self.xmean)
        zmean_arz = self.arz[:, arindex[:self.mu]]
        zmean = np.dot(zmean_arz, self.weights)
        #print('zmean: ')
        #print(zmean)

        # add the value to f_evaluations.
        # at this increment, arx[:, arindex[0]] is the current xmin.
        self.f_evaluations.append(self.strfitnessfct(self.arx[:, arindex[0]]))

        '''
        3.
        '''
        # cumulation: update evolution paths.
        self.ps = (1 - self.cs) * self.ps + (np.sqrt(self.cs * (2 - self.cs) * self.mueff)) * (np.dot(self.B, zmean))
        #print('ps: ')
        #print(self.ps)
        ## TODO: hsig = LA.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * counteval / cma_lambda)) / chiN < 1.4+2 / (N+1)
        hsig = LA.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.counteval / self.cma_lambda)) / self.chiN - 1.4 + 2 / (self.N + 1)
        '''
        *********
        ASK ABOUT HSIG FORMULA - what is the < symbol actually supposed to be.
        for now: I replaced "<" with "-". otherwise, the value becomes a BOOLEAN instead of a FLOAT.
        *********
        '''
        #print('hsig: ')
        #print(hsig)
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * (np.dot(np.dot(self.B, self.D), zmean))
        #print('pc: ')
        #print(self.pc)

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
        first_line_full = (1 - self.c1 - self.cmu) * self.C
        #print('first_line_full:')
        #print(first_line_full)
        second_line_one = self.c1
        second_line_two = (np.dot(self.pc, np.transpose(self.pc)))
        third_line = (1 - hsig) * self.cc * (2 - self.cc) * self.C
        second_line_full = second_line_one * (second_line_two + third_line)
        #print('second_line_full:')
        #print(second_line_full)
        fourth_line = self.cmu
        fifth_line_one = np.dot(self.B, self.D)
        #print('fifth_line_one:')
        #print(fifth_line_one)
        fifth_line_two = self.arz[:, arindex[:self.mu]]
        fifth_line = np.dot(fifth_line_one, fifth_line_two)
        #print('fifth_line:')
        #print(fifth_line)
        sixth_line_one = np.diag(np.transpose(self.weights)[0])
        #print('weights transposed:')
        #print(np.transpose(self.weights)[0])
        #print('sixth_line_one:')
        #print(sixth_line_one)
        # print(weights)
        sixth_line_two = np.transpose(fifth_line)
        sixth_line = np.dot(np.dot(fifth_line, sixth_line_one), sixth_line_two)
        fourth_line_full = fourth_line * sixth_line
        '''
        add first_line_full, second_line_full, fourth_line_full (where 4th_line_full = 4th_line * 5th_line * 6th_line).
        '''
        self.C = first_line_full + second_line_full + fourth_line_full
        #print('\n')
        #print('*********')
        #print('C....')
        #print(self.C)
        #print('*********')
        #print('shape of C: ', self.C.shape)
        #print('\n')

        '''
        5. 
        '''
        # adapt step-size sigma.
        self.cma_sigma = self.cma_sigma * math.exp((self.cs / self.damps) * (LA.norm(self.ps) / self.chiN - 1))
        #print('sigma: ', self.cma_sigma)

        '''
        6.
        '''
        # update B and D from C.
        # look into C more --> make sure it's right.
        # also LA.eig() --> make sure B and D are the correct matrices. (Same as MATLAB implementation.)
        ## TODO: if counteval - eigeneval > cma_lambda / (cone + cmu) / N / 10:
        if self.counteval - self.eigeneval > self.cma_lambda / self.cmu / self.N / 10:
            self.eigeneval = self.counteval
            self.C = np.triu(self.C) + np.transpose(np.triu(self.C, 1))
            #print('C: ', self.C)
            self.D, self.B = LA.eig(self.C)
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
            self.D = np.diag(np.sqrt(self.D))

        '''
        7.
        '''
        # break, if fitness is good enough.
        #if self.arfitness[0] <= self.stopfitness:
        #    continue

        '''
        8.
        '''
        # escape flat fitness, or better terminate.
        if self.arfitness[0] == self.arfitness[math.ceil(0.7 * self.cma_lambda)]:
            self.cma_sigma = self.cma_sigma * math.exp(0.2 + self.cs / self.damps)
            print('WARNING: flat fitness, consider reformulating the objective.')
        #print(str(self.counteval) + ': ' + str(self.arfitness[0]))

        return self.strfitnessfct(self.arx[:, arindex[0]])

        # counteval += 100000000000

    """
    break out of WHILE loop.
    """
    # end of WHILE loop; do print ending statements.



"""
************************************************************************************************************************
************************************************************************************************************************
************************************************************************************************************************
"""

# invocation.
'''
N_dim = 20
s_exact = 10
noise_amp = 0.0
class_objective_function = SparseQuadratic(N_dim, s_exact, noise_amp)
class_x0 = np.random.randn(N_dim)
class_cma_lam = 10
class_cma_mu = 2
class_sigma = 5
class_stop_fitness = 1e-10
class_stop_eval = 1000
query_budget = class_stop_eval
'''
N_dim = 20
class_objective_function = SparseQuadratic(N_dim, 200, .001)
class_x0 = np.random.randn(N_dim)
class_cma_lam = 4 + math.floor(3*math.log(N_dim))
class_cma_mu = int(class_cma_lam/2)
#class_cma_mu =
class_sigma = 0.01
class_stop_fitness = 1e-10
class_stop_eval = 500
query_budget = class_stop_eval

# def __init__(self, objective_function, x0, cma_lam, cma_mu, sigma, stop_fitness, stop_eval).

CMA_trial = CMAoptimizer(class_objective_function, class_x0, class_cma_lam, class_cma_mu, class_sigma, class_stop_fitness, class_stop_eval)

f_vals = []
for i in range(query_budget):
    print(i)
    val = CMA_trial.step()
    print('val: ', val)
    f_vals.append(val)
print('function evaluations: ', f_vals)

plt.plot(f_vals)
plt.show()






