"""
Week 6 tasks - Problem # 3.
"""
# class implementation of SCOBO algorithm.
# reference: Jupyter Notebook (more vanilla implementation - i.e. not tailored to MuJoCo).

'''
SCOBO class in old_algorithms.py....
'''
# class SCOBO:
#     """
#     This class is meant to act as a stand-alone version of the SCOBO algorithm that can be called on any model
#     with any oracle.
#     We will be choosing which parameters are relevant to the grand-scheme of SCOBO in the future.
#     """
#
#     def Solve1BitCS(self, y, Z, m, d, s):
#         '''This function creates a quadratic programming model, calls Gurobi
#         and solves the 1 bit CS subproblem. This function can be replaced with
#         any suitable function that calls a convex optimization package.
#         =========== INPUTS ==============
#         y ........... length d vector of one-bit measurements
#         Z ........... m-by-d sensing matrix
#         m ........... number of measurements
#         d ........... dimension of problem
#         s ........... sparsity level
#
#         =========== OUTPUTS =============
#         x_hat ....... Solution. Note that |x_hat|_2 = 1
#         '''
#
#         model = gp.Model("1BitRecovery")
#         x = model.addVars(2 * d, vtype=GRB.CONTINUOUS)
#         c1 = np.dot(y.T, Z)
#         c = list(np.concatenate((c1, -c1)))
#
#         model.setObjective(quicksum([c[i] * x[i] for i in range(0, 2 * d)]), GRB.MAXIMIZE)
#         model.addConstr(quicksum(x) <= np.sqrt(s), "ell_1")  # sum_i x_i <=1
#         model.addConstr(
#             quicksum(x[i] * x[i] for i in range(0, 2 * d)) - 2 * quicksum(x[i] * x[d + i] for i in range(0, d)) <= 1,
#             "ell_2")  # sum_i x_i^2 <= 1
#         model.addConstrs((x[i] >= 0 for i in range(0, 2 * d)))
#         model.Params.OUTPUTFLAG = 0
#
#         model.optimize()
#         TempSol = model.getAttr('x')
#         x_hat = np.array(TempSol[0:d] - np.array(TempSol[d:2 * d]))
#         return x_hat
#
#     def GradientEstimatorMujoco(self, x_in, Z, r, oracle, m, d, s):
#         '''This function estimates the gradient vector from m Comparison
#         oracle queries, using 1 bit compressed sensing and Gurobi
#         ================ INPUTS ======================
#         Z ......... An m-by-d matrix with rows z_i uniformly sampled from unit sphere
#         x_in ................. Any point in R^d
#         r ................ Sampling radius.
#         oracle..... Comparison oracle object.
#         m ..................... number of measurements.
#         d ..................... dimension of problem
#         s ..................... sparsity
#
#         ================ OUTPUTS ======================
#         g_hat ........ approximation to g/||g||
#         std .......... standard deviation of the results
#
#         23rd May 2020
#         '''
#         y = np.zeros(m)
#         fy = []
#
#         for i in range(0, m):
#             x_temp = Z[i, :]
#             y[i], results = oracle(x_in, x_in + r * Z[i, :])
#             # print('Oracle used to estimate gradient', y[i])
#             # needs to account for multiple oracle simulations
#             if results.shape[0] == 1:
#                 fy.append(results[:, 0])
#                 fy.append(results[:, 1])
#             else:
#                 for n in results:
#                     print(n)
#                     fy.append(n[0])
#                     fy.append(n[1])
#         std = np.std(fy)
#         g_hat = self.Solve1BitCS(y, Z, m, d, s)
#         return g_hat, std
#
#     def LineSearch(self, x, g_hat, last_step_size, default_step_size, oracle, d, s, w=.75, psi=1.01, M=1):
#         '''
#         This function is Charles Stoksik's implementation of the line search. Unfortunately when he
#         runs the code it does not work with a linesearch variable so he made this function
#         to try to get it to work himself.
#         ================ INPUTS ======================
#         x ........................ current point
#         g_hat .................... search direction
#         last_step_size ........... step size from last itertion
#         default_step_size......... a safe lower bound of step size
#         oracle.................... Comparison oracle parameters.
#         d ........................ dimension of problem
#         s ........................ sparsity
#         w ........................ confidence parameter
#         psi ...................... searching parameter
#
#         ================ OUTPUTS ======================
#         alpha .................... step size found
#         less_than_defalut ........ return True if found step size less than default step size
#         queries_count ............ number of oracle queries used in linesearch
#         28th July 2020
#         '''
#         if psi <= 1:
#             raise ValueError('psi increment for linesearch must be > 1')
#         if w > 1:
#             raise ValueError('omega increment for linesearch must be <= 1')
#         if M < 1:
#             M = 1
#             print('Set M to 1 for LineSearch Oracle')
#
#         previous_oracle_M = oracle.M
#         oracle.M = int(M)  # just in case we want to query the oracle more than once
#
#         alpha = default_step_size
#         increment_count = 0  # how many times we change the stepsize
#
#         while oracle(x + alpha * g_hat, x + psi * alpha * g_hat)[0] >= w:  # I changed it...
#             print('Oracle used for linesearch')
#             alpha *= psi
#             increment_count += 1
#
#         oracle.M = previous_oracle_M  # set the oracle count back to previous
#
#         return alpha, None, M * increment_count  # none is for less_the_defalut (not in implementation paper)
#
#     def GetStepSize(self, x, g_hat, last_step_size, default_step_size, oracle, d, s):
#         '''This function use line search to estimate the best step size on the given
#         direction via noisy comparison
#         ================ INPUTS ======================
#         x ........................ current point
#         g_hat .................... search direction
#         last_step_size ........... step size from last itertion
#         default_step_size......... a safe lower bound of step size
#         oracle.................... Comparison oracle parameters.
#         d ........................ dimension of problem
#         s ........................ sparsity
#
#         ================ OUTPUTS ======================
#         alpha .................... step size found
#         less_than_defalut ........ return True if found step size less than default step size
#         queries_count ............ number of oracle queries used in linesearch
#         25th May 2020
#         '''
#
#         # First make sure current step size descends
#         omega = 0.1
#         num_round = 20  # number of oracle queries per step
#         descend_count = 0
#         queries_count = 0
#         less_than_defalut = False
#         update_factor = np.sqrt(2)
#
#         alpha = last_step_size  # start with last step size
#         print(default_step_size)
#         point1 = x - alpha * g_hat
#
#         for round in range(0, num_round):  # compare n rounds for every pair of points,
#             # print('Oracle')
#             is_descend, _ = oracle(point1, x)  # ,kappa,mu,delta_0,d,s)
#             queries_count = queries_count + 1
#             if is_descend == 1:
#                 descend_count = descend_count + 1
#         p = descend_count / num_round
#
#         # we try increase step size if p is larger, try decrease step size is
#         # smaller, otherwise keep the current alpha
#         if p >= 0.5 + omega:  # compare with x
#             while True:
#                 point2 = x - update_factor * alpha * g_hat  # what is the point of this?
#                 descend_count = 0
#                 for round in range(0, num_round):  # compare n rounds for every pair of points,
#                     is_descend, _ = oracle(point1, x)  # comapre with point1
#                     # print('executed oracle p > 0.5')
#                     queries_count = queries_count + 1
#                     if is_descend == 1:
#                         descend_count = descend_count + 1
#                 p = descend_count / num_round
#                 # print('done with oracle')
#                 if p >= 0.5 + omega:
#                     alpha = update_factor * alpha
#                     point1 = x - alpha * g_hat
#                 else:
#                     # print('left function')
#                     return alpha, less_than_defalut, queries_count
#         elif p <= 0.5 - omega:  # else: we try decrease step size
#             while True:
#                 alpha = alpha / update_factor
#                 if alpha <= default_step_size:
#                     alpha = default_step_size
#                     less_than_defalut = True
#                     return alpha, less_than_defalut, queries_count
#                 point2 = x - alpha * g_hat
#                 descend_count = 0
#                 for round in range(0, num_round):
#                     is_descend, _ = oracle(point1, x)
#                     # print('executed oracle p < 0.5')
#                     queries_count = queries_count + 1
#                     if is_descend == 1:
#                         descend_count = descend_count + 1
#                 p = descend_count / num_round
#                 # print('done with oracle')
#                 if p >= 0.5 + omega:
#                     return alpha, less_than_defalut, queries_count
#         else:
#             # print('left function')
#             alpha = last_step_size
#
#         return alpha, less_than_defalut, queries_count
#
#     def SCOBO_mujoco(self, num_iterations, default_step_size, x0, r, model, oracle, m, d, s, linesearch,
#                      save_specs=False, show_runs=False):
#         ''' This function implements the SCOBO algorithm, as described
#         in our paper.
#
#         =============== INPUTS ================
#         num_iterations ................ number of iterations
#         default_step_size ............. default step size
#         x0 ............................ initial iterate
#         r ............................. sampling radius
#         model ......................... our reward function
#         oracle ........................ comparison oracle object
#         m ............................. number of samples per iteration
#         d ............................. dimension of problem
#         s ............................. sparsity level
#         linesearch ................... wheather linesearch for step size. if not, use default step size
#         show_runs ..................... Will render the simulation when recording it for rewards
#
#         =============== OUTPUTS ================
#         regret ....................... vector of errors f(x_k) - min f
#         tau_vec ...................... tau_vec(k) = fraction of flipped measurements at k-th iteration
#         c_num_queries ................ cumulative number of queries.
#
#         '''
#
#         # initialize arrays
#         rewards = np.zeros((num_iterations, 1))
#         tau_vec = np.zeros((num_iterations, 1))
#         x = np.squeeze(x0)
#         Z = np.zeros((m, d))
#         # default_step_size = self.alpha
#         conv = None  # step number we exceed reward threshold at
#
#         # start with default step size when using line search
#         linesearch_queries = 0
#         if linesearch:
#             step_size = default_step_size
#             less_than_default_vec = np.zeros((num_iterations, 1))  # not outputing this in current version
#
#         for i in range(0, num_iterations):
#             for j in range(0, m):
#                 temp = np.random.randn(1, d)  # randn is N(0, 1)
#                 Z[j, :] = temp / np.linalg.norm(temp)  # normalize
#
#             g_hat, std = self.GradientEstimatorMujoco(x, Z, r * 1.01 ** (-i), oracle, m, d,
#                                                       s)  # self.m oracle comparisons occur here
#
#             # search for optimal step-size time
#             if linesearch:
#                 print(step_size)
#                 # step_size, less_than_default, queries_count = self.GetStepSize(x, g_hat, step_size,
#                 default_step_size, oracle, d, s)
#                 step_size, less_than_default, queries_count = self.LineSearch(x, g_hat, step_size, default_step_size,
#                                                                               oracle, d, s)
#                 less_than_default_vec[i] = less_than_default
#                 linesearch_queries += queries_count
#             else:
#                 step_size = default_step_size
#             x = x + step_size * 1.001 ** (-i) * g_hat  # Decaying stepsize model...
#             print(x.shape)
#             # print('Somehow got here')
#
#             if show_runs:
#                 rewards[i] = model.render(x)
#             else:
#                 rewards[i] = model(x)
#             # print('Ran sim')
#
#             if bool(model.reward_threshold):  # if we have a reward threshold, note 0 will return False...
#                 if rewards[i] > model.reward_threshold and conv == None:  # replace -10 with reward_threshold
#                     conv = i * m + linesearch_queries
#                     print('************** Reward Threshold Exceeded **************')
#
#             if rewards[i] < rewards[i - 1]:
#                 default_step_size /= 1.01
#                 default_step_size = max(default_step_size, .02)  # interesting, look at later
#             print('current rewards at step', i + 1, ':', rewards[i])
#             print('step_size:', step_size * 1.001 ** (-i))
#             print('gradient norm:', np.linalg.norm(g_hat))
#
#         c_num_queries = m * np.arange(start=0, stop=num_iterations, step=1) + linesearch_queries
#         x_hat = x
#         return x_hat, rewards, c_num_queries, conv
#
#     def __call__(self, num_iterations, default_step_size, x0, r, model, oracle, m, d, s, linesearch, save_specs=False,
#                  algo_seed=0, show_runs=False):
#         np.random.seed(algo_seed)
#         return self.SCOBO_mujoco(num_iterations, default_step_size, x0, r, model, oracle, m, d, s, linesearch,
#                                  save_specs, show_runs)
#

'''
Example of Class implementation....
'''
# class ZOBCD(BaseOptimizer):
#     ''' ZOBCD for black box optimization. A sparsity-aware, block coordinate
#     descent method.
#
#     INPUTS:
#         y0 ................. initial iterate
#         step_size .......... step size
#         f .................. the objective function
#         params ............. A dict containing additional parameters, e.g. the
#         number of blocks (see Example.py)
#         function_budget .... total number of function evaluations allowed.
#         shuffle ............ If true, we choose a new random assignment of
#         variables to blocks every (number_of_blocks) iterations.
#         function_target .... If not none, this specifies the desired optimality
#         gap
#
#     March 23rd 2021
#
#     '''
#
#     def __init__(self, x0, step_size, f, params, function_budget=10000, shuffle=True,
#                  function_target=None):
#
#         super().__init__()
#
#         self.function_evals = 0
#         self.function_budget = function_budget
#         self.function_target = function_target
#         self.f = f
#         self.x = x0
#         self.n = len(x0)
#         self.t = 0
#         self.Type = params["Type"]
#         self.sparsity = params["sparsity"]
#         self.delta = params["delta"]
#         self.step_size = step_size
#         self.shuffle = shuffle
#         self.permutation = np.random.permutation(self.n)
#
#         # block stuff
#         oversampling_param = 1.1
#         self.J = params["J"]
#         self.block_size = int(np.ceil(self.n / self.J))
#         self.sparsity = int(np.ceil(oversampling_param * self.sparsity / self.J))
#         print(self.sparsity)
#         self.samples_per_block = int(np.ceil(oversampling_param * self.sparsity * np.log(self.block_size)))
#
#         # Define cosamp_params
#         if self.Type == "ZOBCD-R":
#             Z = 2 * (np.random.rand(self.samples_per_block, self.block_size) > 0.5) - 1
#         elif self.Type == "ZOBCD-RC":
#             z1 = 2 * (np.random.rand(1, self.block_size) > 0.5) - 1
#             Z1 = circulant(z1)
#             SSet = np.random.choice(self.block_size, self.samples_per_block, replace=False)
#             Z = Z1[SSet, :]
#         else:
#             raise Exception("Need to choose a type, either ZOBCD-R or ZOBCD-RC")
#
#         cosamp_params = {"Z": Z, "delta": self.delta, "maxiterations": 10,
#                          "tol": 0.5, "sparsity": self.sparsity, "block": []}
#         self.cosamp_params = cosamp_params
#
#     def CosampGradEstimate(self):
#         # Gradient estimation
#
#         maxiterations = self.cosamp_params["maxiterations"]
#         Z = self.cosamp_params["Z"]
#         delta = self.cosamp_params["delta"]
#         sparsity = self.cosamp_params["sparsity"]
#         tol = self.cosamp_params["tol"]
#         block = self.cosamp_params["block"]
#         num_samples = np.size(Z, 0)
#         x = self.x
#         f = self.f
#         dim = len(x)
#
#         Z_padded = np.zeros((num_samples, dim))
#         Z_padded[:, block] = Z
#
#         y = np.zeros(num_samples)
#         print(num_samples)
#         function_estimate = 0
#
#         for i in range(num_samples):
#             y_temp = f(x + delta * np.transpose(Z_padded[i, :]))
#             y_temp2 = f(x)
#             function_estimate += y_temp2
#             y[i] = (y_temp - y_temp2) / (np.sqrt(num_samples) * delta)
#             self.function_evals += 2
#
#         Z = Z / np.sqrt(num_samples)
#         block_grad_estimate = cosamp(Z, y, sparsity, tol, maxiterations)
#         grad_estimate = np.zeros(dim)
#         grad_estimate[block] = block_grad_estimate
#         function_estimate = function_estimate / num_samples
#
#         return grad_estimate, function_estimate
#
#     def step(self):
#         # Take step of optimizer
#
#         if self.t % self.J == 0 and self.shuffle:
#             self.permutation = np.random.permutation(self.n)
#             print('Reshuffled!')
#
#         coord_index = np.random.randint(self.J)
#         block = np.arange((coord_index - 1) * self.block_size, min(coord_index * self.block_size, self.n))
#         block = self.permutation[block]
#         self.cosamp_params["block"] = block
#         grad_est, f_est = self.CosampGradEstimate()
#         self.f_est = f_est
#         self.x += -self.step_size * grad_est
#
#         if self.reachedFunctionBudget(self.function_budget, self.function_evals):
#             # if budget is reached return parent
#             return self.x, self.function_evals, 'B'
#
#         return self.x, self.function_evals, False




# _______________________________________________________
'''
SCOBO implemented as a class.
'''
from ExampleCode.base import BaseOptimizer
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
import random
from gurobipy import GRB, quicksum

class SCOBOoptimizer(BaseOptimizer):
    def __init__(self, num_iterations,default_step_size,x0,r,kappa,delta_0,mu,m,d,s,line_search,warm_started):
        super().__init__()

        self.function_evals = 0
        self.function_vals = []
        self.function_budget = num_iterations
        self.default_step_size = default_step_size
        self.x0 = x0
        self.r = r
        self.kappa = kappa
        self.delta_0 = delta_0
        self.mu = mu
        self.m = m
        self.d = d
        self.s = s
        self.line_search = line_search
        self.warm_started = warm_started

        # whatever's in the original SCOBO class that isn't in the loop....
        regret = np.zeros((self.function_budget, 1))
        self.regret = regret
        tau_vec = np.zeros((self.function_budget, 1))
        self.tau_vec = tau_vec
        linesearch_queries = np.zeros(self.function_budget)
        self.linesearch_queries = linesearch_queries
        x = np.squeeze(self.x0)
        self.x = x

        # start with default step size when using line search
        step_size = default_step_size
        self.step_size = step_size

        if self.line_search:
            less_than_default_vec = np.zeros((self.function_budget, 1))
            self.less_than_default_vec = less_than_default_vec

    def SetQ(self, d_, s):
        """
        Setup the global Q for quadratic oracle, so we don't have to re-create Q every query
        Run this function before we run any function call that makes QuadraticOracle queries
        May 25th 2020
        """
        global Q
        Diagonal = np.zeros(d_)
        for i in range(0, s):
            p = 0.5
            # Diagonal[i]= (i+1)**(-1.0/p)
            Diagonal[i] = 1
        Q = np.diag(Diagonal)

    def object_fcn(self, x_):
        return np.dot(np.dot(x_.T, Q), x_)

    def QuadraticOracle(self, x, y, kappa, mu, delta_0):
        '''Implements comparison oracle for sparse quadratic
        In noiseless case, return 1 if f(x)<f(y); otherwise return -1
        function f(x) = x^TQx
        May 25th 2020'''
        self.function_evals += 1
        # Diagonal = np.concatenate((np.ones(s),np.zeros(d-s)))
        # Q = np.diag(Diagonal)
        # fx = np.dot(np.dot(x.T,Q),x)
        # fy = np.dot(np.dot(y.T,Q),y)
        fx = self.object_fcn(x)
        fy = self.object_fcn(y)
        f_diff = np.squeeze(fy - fx)
        if f_diff == 0:
            f_diff = (random.randrange(2) - 0.5) / 50
        prob = 0.5 + np.minimum(mu * np.absolute(f_diff) ** (kappa - 1.0), delta_0)  # Probability of bit-flip
        mask = 2 * np.random.binomial(1, prob) - 1
        res = np.squeeze(mask * np.sign(f_diff))
        if mask == 1:
            bit_flipped = 0
        else:
            bit_flipped = 1
        return res, bit_flipped

    def Solve1BitCS(self, y, Z, m, d, s):
        """
        This function creates a quadratic programming model, calls Gurobi
        and solves the 1 bit CS subproblem. This function can be replaced with
        any suitable function that calls a convex optimization package.
        =========== INPUTS ==============
        y ........... length d vector of one-bit measurements
        Z ........... m-by-d sensing matrix
        m ........... number of measurements
        d ........... dimension of problem
        s ........... sparsity level

        =========== OUTPUTS =============
        x_hat ....... Solution. Note that \|x_hat\|_2 = 1
        """

        model = gp.Model("1BitRecovery")
        x = model.addVars(2 * d, vtype=GRB.CONTINUOUS)
        c1 = np.dot(np.transpose(y), Z)
        c = list(np.concatenate((c1, -c1)))

        model.setObjective(quicksum(c[i] * x[i] for i in range(0, 2 * d)), GRB.MAXIMIZE)
        model.addConstr(quicksum(x) <= np.sqrt(s), "ell_1")  # sum_i x_i <=1
        model.addConstr(
            quicksum(x[i] * x[i] for i in range(0, 2 * d)) - 2 * quicksum(x[i] * x[d + i] for i in range(0, d)) <= 1,
            "ell_2")  # sum_i x_i^2 <= 1
        model.addConstrs(x[i] >= 0 for i in range(0, 2 * d))
        model.Params.OUTPUTFLAG = 0

        model.optimize()
        TempSol = model.getAttr('x')
        x_hat = np.array(TempSol[0:d] - np.array(TempSol[d:2 * d]))
        return x_hat

    def GradientEstimator(self, x_in, Z, r, kappa, delta_0, mu, m, d, s):
        '''This function estimates the gradient vector from m Comparison
        oracle queries, using 1 bit compressed sensing and Gurobi
        ================ INPUTS ======================
        Z ......... An m-by-d matrix with rows z_i uniformly sampled from unit sphere
        x_in ................. Any point in R^d
        r ................ Sampling radius.
        kappa,delta_0, mu..... Comparison oracle parameters.
        m ..................... number of measurements.
        d ..................... dimension of problem
        s ..................... sparsity

        ================ OUTPUTS ======================
        g_hat ........ approximation to g/||g||
        tau .......... fraction of bit-flips/ incorrect one-bit measurements.
        y ............ vector of measurements

        23rd May 2020
        '''
        y = np.zeros(m)
        tau = 0
        for i in range(0, m):
            x_temp = Z[i, :]
            y[i], bit_flipped = self.QuadraticOracle(x_in, x_in + r * Z[i, :], kappa, mu, delta_0, d, s)
            tau += bit_flipped
        g_hat = self.Solve1BitCS(y, Z, m, d, s)
        tau = tau / m
        return g_hat, tau, y

    def TestSparsity(self, y, g_hat, Z, m):
        '''
        This function tests agreement between the measurements, y, and the estimated
        gradient, g_hat. If the agreement is not good enough, it increases s and m.

        '''
        y_hat = np.sign(np.dot(Z, g_hat))
        HammingFraction = len([i for i in range(m) if y[i] != y_hat[i]]) / m
        if HammingFraction >= 0.45:
            print("Fail!")
        return HammingFraction

    def GetStepSize(self, x, g_hat, last_step_size, default_step_size, kappa, mu, delta_0, d, s, warm_started):
        '''This function use line search to estimate the best step size on the given
        direction via noisy comparison
        ================ INPUTS ======================
        x ........................ current point
        g_hat .................... search direction
        last_step_size ........... step size from last itertion
        default_step_size......... a safe lower bound of step size
        kappa,delta_0, mu......... Comparison oracle parameters.
        d ........................ dimension of problem
        s ........................ sparsity

        ================ OUTPUTS ======================
        alpha .................... step size found
        less_than_defalut ........ return True if found step size less than default step size
        queries_count ............ number of oracle queries used in linesearch
        25th May 2020
        '''

        # First make sure current step size descends
        omega = 0.05
        num_round = 40
        descend_count = 0
        queries_count = 0
        less_than_defalut = False
        # update_factor = np.sqrt(2)
        update_factor = 2

        if warm_started:
            alpha = last_step_size  # start with last step size
        else:
            alpha = default_step_size
        point1 = x - alpha * g_hat

        for round in range(0, num_round):  # compare n rounds for every pair of points,
            is_descend, bit_flipped = self.QuadraticOracle(point1, x, kappa, mu, delta_0, d, s)
            queries_count = queries_count + 1
            if is_descend == 1:
                descend_count = descend_count + 1
        p = descend_count / num_round
        # print(p)

        # we try increase step size if p is larger, try decrease step size is
        # smaller, otherwise keep the current alpha
        if p >= 0.5 + omega:  # compare with x
            while True:
                point2 = x - update_factor * alpha * g_hat
                descend_count = 0
                for round in range(0, num_round):  # compare n rounds for every pair of points,
                    is_descend, bit_flipped = self.QuadraticOracle(point2, point1, kappa, mu, delta_0, d,
                                                              s)  # comapre with point1
                    queries_count = queries_count + 1
                    if is_descend == 1:
                        descend_count = descend_count + 1
                p = descend_count / num_round
                if p >= 0.5 + omega:
                    alpha = update_factor * alpha
                    point1 = x - alpha * g_hat
                else:
                    return alpha, less_than_defalut, queries_count
        elif warm_started == False:
            less_than_defalut = True
            return alpha, less_than_defalut, queries_count
        elif p <= 0.5 - omega:  # else: we try decrease step size
            while True:
                alpha = alpha / update_factor
                if alpha < default_step_size:
                    alpha = default_step_size
                    less_than_defalut = True
                    return alpha, less_than_defalut, queries_count
                point2 = x - alpha * g_hat
                descend_count = 0
                for round in range(0, num_round):
                    is_descend, bit_flipped = self.QuadraticOracle(point2, x, kappa, mu, delta_0, d, s)  # compare with x
                    queries_count = queries_count + 1
                    if is_descend == 1:
                        descend_count = descend_count + 1
                p = descend_count / num_round
                if p >= 0.5 + omega:
                    return alpha, less_than_defalut, queries_count
        # else:
        #    alpha = last_step_size

        return alpha, less_than_defalut, queries_count

    # (i) will be the iteration.
    # will input it when I create an instance of this class and then call the step function for the # of iterations.
    def step(self, i):
        g_hat, tau, y = self.GradientEstimator(self.x, self.Z, self.r, self.kappa, self.delta_0, self.mu, self.m, self.d, self.s)
        HammingFraction = self.TestSparsity(y, g_hat, self.Z, self.s, self.m)
        print(["The Hamming Fraction is " + str(HammingFraction)])
        if self.line_search:
            step_size, less_than_defalut, queries_count = self.GetStepSize(self.x, g_hat, self.step_size, self.default_step_size, self.kappa, self.mu,
                                                                      self.delta_0, self.d, self.s, self.warm_started)
            self.less_than_default_vec[i] = less_than_defalut
            self.linesearch_queries[i] = queries_count
            # print(queries_count)
        # print(step_size)
        x = self.x - self.step_size * g_hat
        self.x = x
        self.regret[i] = self.object_fcn(x)  # f(x_min) = 0
        print(self.regret[i])
        self.tau_vec[i] = tau

        self.function_vals.append(self.object_fcn(self.x))

        if self.reachedFunctionBudget(self.function_budget, self.function_evals):
            # if budget is reached return parent
            return self.x, self.function_evals, 'B'

        return self.x, self.function_evals, False




# _____________________________________________
"""
WEEK 6 - task #5.
"""
# the following are references from the first paper linked to Week #3 TASKS....

# https://papers.nips.cc/paper/2015/file/934815ad542a4a7c5e8a2dfa04fea9f5-Paper.pdf
'''
Not sure if this example works since it involves calculating the gradient but we can't get any function values.
Check with Daniel - show Algorithm #2.
'''

# https://proceedings.neurips.cc/paper/2018/file/36d7534290610d9b7e9abed244dd2f28-Paper.pdf
'''
Show Daniel Algorithm #2. It sounds promising based off the title:
(Zeroth-order (Non)-Convex Stochastic Optimization via Conditional Gradient and Gradient Updates).
'''

# https://arxiv.org/pdf/2003.13001.pdf
'''
This is Daniel's paper! Is ZORO something we can use / have used?
'''





