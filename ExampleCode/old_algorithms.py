from agent import Agent
from oracle import Oracle
import oracle as o
from oracle import GUIOracle
from agent import MujModel

import numpy as np
from cheetah.mujocomodel import MujocoModel

import gurobipy as gp 
from gurobipy import GRB, quicksum

class SCOBO:
    """
    This class is meant to act as a stand-alone version of the SCOBO algorithm that can be called on any model with any oracle.
    We will be choosing which parameters are relevant to the grand-scheme of SCOBO in the future.
    """

    def Solve1BitCS(self, y, Z, m, d, s):
        '''This function creates a quadratic programming model, calls Gurobi
        and solves the 1 bit CS subproblem. This function can be replaced with
        any suitable function that calls a convex optimization package.
        =========== INPUTS ==============
        y ........... length d vector of one-bit measurements
        Z ........... m-by-d sensing matrix
        m ........... number of measurements
        d ........... dimension of problem
        s ........... sparsity level
        
        =========== OUTPUTS =============
        x_hat ....... Solution. Note that |x_hat|_2 = 1
        '''
        
        model = gp.Model("1BitRecovery")
        x = model.addVars(2*d, vtype = GRB.CONTINUOUS)
        c1 = np.dot(y.T,Z)
        c = list(np.concatenate((c1,-c1)))

        model.setObjective(quicksum([c[i]*x[i] for i in range(0, 2*d)]), GRB.MAXIMIZE)
        model.addConstr(quicksum(x) <= np.sqrt(s),"ell_1")  # sum_i x_i <=1
        model.addConstr(quicksum(x[i]*x[i] for i in range(0,2*d)) - 2*quicksum(x[i]*x[d+i] for i in range(0, d))<= 1, "ell_2") # sum_i x_i^2 <= 1
        model.addConstrs((x[i] >= 0 for i in range(0, 2*d)))
        model.Params.OUTPUTFLAG = 0

        model.optimize()
        TempSol = model.getAttr('x')
        x_hat = np.array(TempSol[0:d] - np.array(TempSol[d:2*d]))
        return x_hat


    def GradientEstimatorMujoco(self, x_in, Z, r, oracle, m, d, s):
        '''This function estimates the gradient vector from m Comparison
        oracle queries, using 1 bit compressed sensing and Gurobi
        ================ INPUTS ======================
        Z ......... An m-by-d matrix with rows z_i uniformly sampled from unit sphere
        x_in ................. Any point in R^d
        r ................ Sampling radius.
        oracle..... Comparison oracle object.
        m ..................... number of measurements.
        d ..................... dimension of problem
        s ..................... sparsity
        
        ================ OUTPUTS ======================
        g_hat ........ approximation to g/||g||
        std .......... standard deviation of the results 
        
        23rd May 2020
        '''
        y = np.zeros(m)
        fy = []

        for i in range(0,m):
            x_temp = Z[i,:]
            y[i], results = oracle(x_in, x_in + r*Z[i,:])
            #print('Oracle used to estimate gradient', y[i])
            # needs to account for multiple oracle simulations
            if results.shape[0] == 1:
                fy.append(results[:, 0])
                fy.append(results[:, 1])
            else:
                for n in results:
                    print(n)
                    fy.append(n[0])
                    fy.append(n[1])
        std = np.std(fy)
        g_hat = self.Solve1BitCS(y, Z, m, d, s)
        return g_hat, std

    def LineSearch(self, x, g_hat, last_step_size, default_step_size, oracle, d, s, w=.75, psi=1.01, M=1):
        '''
        This function is Charles Stoksik's implementation of the line search. Unfortunately when he
        runs the code it does not work with a linesearch variable so he made this function
        to try to get it to work himself.
        ================ INPUTS ======================
        x ........................ current point
        g_hat .................... search direction
        last_step_size ........... step size from last itertion
        default_step_size......... a safe lower bound of step size
        oracle.................... Comparison oracle parameters.
        d ........................ dimension of problem
        s ........................ sparsity
        w ........................ confidence parameter
        psi ...................... searching parameter

        ================ OUTPUTS ======================
        alpha .................... step size found
        less_than_defalut ........ return True if found step size less than default step size
        queries_count ............ number of oracle queries used in linesearch
        28th July 2020
        '''
        if psi <= 1:
            raise ValueError('psi increment for linesearch must be > 1')
        if w > 1:
            raise ValueError('omega increment for linesearch must be <= 1')
        if M < 1:
            M = 1
            print('Set M to 1 for LineSearch Oracle')

        previous_oracle_M = oracle.M
        oracle.M = int(M) # just in case we want to query the oracle more than once

        alpha = default_step_size
        increment_count = 0 # how many times we change the stepsize

        while oracle(x + alpha*g_hat, x + psi*alpha*g_hat)[0] >= w: # I changed it...
            print('Oracle used for linesearch')
            alpha *= psi
            increment_count += 1

        oracle.M = previous_oracle_M # set the oracle count back to previous

        return alpha, None, M*increment_count # none is for less_the_defalut (not in implementation paper)


    def GetStepSize(self, x, g_hat, last_step_size, default_step_size, oracle, d, s):
        '''This function use line search to estimate the best step size on the given 
        direction via noisy comparison 
        ================ INPUTS ======================
        x ........................ current point
        g_hat .................... search direction
        last_step_size ........... step size from last itertion
        default_step_size......... a safe lower bound of step size
        oracle.................... Comparison oracle parameters.
        d ........................ dimension of problem
        s ........................ sparsity
        
        ================ OUTPUTS ======================
        alpha .................... step size found
        less_than_defalut ........ return True if found step size less than default step size
        queries_count ............ number of oracle queries used in linesearch
        25th May 2020
        '''
        
        # First make sure current step size descends
        omega = 0.1
        num_round = 20 # number of oracle queries per step
        descend_count = 0
        queries_count = 0
        less_than_defalut = False
        update_factor = np.sqrt(2)
        
        alpha = last_step_size  # start with last step size
        print(default_step_size)
        point1 = x - alpha * g_hat
        
        for round in range(0, num_round): # compare n rounds for every pair of points, 
            #print('Oracle')
            is_descend, _  = oracle(point1, x) #,kappa,mu,delta_0,d,s)
            queries_count = queries_count + 1
            if is_descend == 1:
                descend_count = descend_count + 1
        p = descend_count/num_round
        
        
        # we try increase step size if p is larger, try decrease step size is
        # smaller, otherwise keep the current alpha
        if p >= 0.5 + omega:   # compare with x
            while True:        
                point2 = x - update_factor * alpha * g_hat # what is the point of this?
                descend_count = 0
                for round in range(0,num_round):   # compare n rounds for every pair of points,
                    is_descend, _ = oracle(point1, x)   # comapre with point1
                    #print('executed oracle p > 0.5')
                    queries_count = queries_count + 1
                    if is_descend == 1:
                        descend_count = descend_count + 1
                p = descend_count/num_round
                #print('done with oracle')
                if p >= 0.5 + omega:
                    alpha = update_factor * alpha
                    point1 = x - alpha * g_hat
                else:
                    #print('left function')
                    return alpha, less_than_defalut, queries_count
        elif p <= 0.5 - omega:   # else: we try decrease step size
            while True:
                alpha = alpha / update_factor
                if alpha <= default_step_size:
                    alpha = default_step_size
                    less_than_defalut = True
                    return alpha, less_than_defalut, queries_count
                point2 = x - alpha * g_hat
                descend_count = 0
                for round in range(0, num_round): 
                    is_descend, _ = oracle(point1, x)
                    #print('executed oracle p < 0.5')
                    queries_count = queries_count + 1
                    if is_descend == 1:
                        descend_count = descend_count + 1
                p = descend_count/num_round
                #print('done with oracle')
                if p >= 0.5 + omega:
                    return alpha, less_than_defalut, queries_count
        else:
            #print('left function')
            alpha = last_step_size
        
        return alpha, less_than_defalut, queries_count


    def SCOBO_mujoco(self, num_iterations, default_step_size, x0, r, model, oracle, m, d, s, linesearch, save_specs=False, show_runs=False):
        ''' This function implements the SCOBO algorithm, as described 
        in our paper. 
        
        =============== INPUTS ================
        num_iterations ................ number of iterations
        default_step_size ............. default step size
        x0 ............................ initial iterate
        r ............................. sampling radius
        model ......................... our reward function
        oracle ........................ comparison oracle object
        m ............................. number of samples per iteration
        d ............................. dimension of problem
        s ............................. sparsity level
        linesearch ................... wheather linesearch for step size. if not, use default step size
        show_runs ..................... Will render the simulation when recording it for rewards
         
        =============== OUTPUTS ================
        regret ....................... vector of errors f(x_k) - min f
        tau_vec ...................... tau_vec(k) = fraction of flipped measurements at k-th iteration
        c_num_queries ................ cumulative number of queries.
        
        '''

        # initialize arrays
        rewards = np.zeros((num_iterations,1))
        tau_vec = np.zeros((num_iterations,1))
        x = np.squeeze(x0)
        Z = np.zeros((m,d))
        # default_step_size = self.alpha
        conv = None # step number we exceed reward threshold at
        
        # start with default step size when using line search
        linesearch_queries = 0
        if linesearch:
            step_size = default_step_size
            less_than_default_vec = np.zeros((num_iterations,1))  # not outputing this in current version
        
        for i in range(0,num_iterations):
            for j in range(0, m):
                temp = np.random.randn(1, d) # randn is N(0, 1)
                Z[j,:] = temp/np.linalg.norm(temp) # normalize

            g_hat, std = self.GradientEstimatorMujoco(x, Z, r*1.01**(-i), oracle, m, d, s) # self.m oracle comparisons occur here

            # search for optimal step-size time
            if linesearch:
                print(step_size)
                #step_size, less_than_default, queries_count = self.GetStepSize(x, g_hat, step_size, default_step_size, oracle, d, s)
                step_size, less_than_default, queries_count = self.LineSearch(x, g_hat, step_size, default_step_size, oracle, d, s)
                less_than_default_vec[i] = less_than_default
                linesearch_queries += queries_count
            else:
                step_size = default_step_size
            x = x + step_size*1.001**(-i) * g_hat # Decaying stepsize model...
            print(x.shape)  
            #print('Somehow got here')

            if show_runs:
                rewards[i] = model.render(x)
            else:
                rewards[i] = model(x)
            #print('Ran sim')
            
            
            if bool(model.reward_threshold): # if we have a reward threshold, note 0 will return False...
                if rewards[i] > model.reward_threshold and conv == None: # replace -10 with reward_threshold
                    conv = i*m + linesearch_queries
                    print('************** Reward Threshold Exceeded **************')

            if rewards[i] < rewards[i-1]:
                default_step_size /= 1.01
                default_step_size = max(default_step_size, .02) # interesting, look at later
            print('current rewards at step', i+1, ':', rewards[i])
            print('step_size:', step_size*1.001**(-i))
            print('gradient norm:', np.linalg.norm(g_hat))
            
        c_num_queries = m*np.arange(start=0, stop=num_iterations, step=1) + linesearch_queries
        x_hat = x
        return x_hat, rewards, c_num_queries, conv

    def __call__(self, num_iterations, default_step_size, x0, r, model, oracle, m, d, s, linesearch, save_specs=False, algo_seed=0, show_runs=False):
        np.random.seed(algo_seed)
        return self.SCOBO_mujoco(num_iterations, default_step_size, x0, r, model, oracle, m, d, s, linesearch, save_specs, show_runs)


# s = SCOBO()
# m = MujocoModel('Reacher-v2', 2000, -0.5, 0.5)
# x, re, qu, conv = s.SCOBO_mujoco(100, .2, np.zeros((22, 1)) + .3, 0.2/np.sqrt(2), m, Oracle(m, {'kappa':2, 'delta':.5, 'mu':1, 'M':1}), 26, 22, 4, True, -10)
# print(conv)


class GLDBS:

    def __init__(self, fast=False):
        self.fast = fast # FBS or BS


    def flat_comparisons(self, x0, list_of_permutations, oracle):
        """
        Simple function for finding the optimal policy in a list of policies

        =============== INPUTS ===============
        x0 ......................... Initial x0, currently working from
        list_of_permutations ....... A list of noise to add to the best_x
        oracle ..................... Comparison Oracle

        =============== OUTPUTS ==============
        best_x ..................... The best permutation of the oracle
        oracle_comparisons ......... Number of oracle comparisons
        """
        best_x = x0.copy()
        oracle_comparisons = 0
        for v in list_of_permutations:
            comp, _ = oracle(best_x, x0 + v)
            oracle_comparisons += 1
            if comp == 1:
                best_x = x0 + v
        return best_x, oracle_comparisons

    def tournament_comparisons(self, x0, list_of_permutations, oracle, skip_x0=False):
        """
        Tournament style comparison oracle to find best permutation.

        =============== INPUTS ===============
        x0 ......................... Initial x0, currently working from
        list_of_permutations ....... A list of noise to add to the best_x
        oracle ..................... Comparison Oracle

        =============== OUTPUTS ==============
        best_x ..................... The best permutation of the oracle
        oracle_comparisons ......... Number of oracle comparisons 
        """
        x0vs = [x0 + v for v in list_of_permutations] + [x0] # add all possible permutations to a list
        # if odd
        return tourney(x0vs, oracle, 0)

    def tourney(self, entries, oracle, count):
        """
        Recursive tournament function for comparisons

        ================ INPUTS ================
        entries ..................... Each tournament participant, should be policy vectors
        oracle ...................... Comparison Oracle

        ================ OUTPUTS ===============
        entries[0] .................. When only one policy left
        count ....................... Number of oracle queries
        """
        if len(entries) == 1:
            return entries[0], count
        winners = []
        if len(entries) % 2:
            for i in range(0, len(entries) - 1, step=2):
                comp, _ = oracle(entries[i], entries[i+1])
                count += 1
                if comp:
                    winners.append(entries[i+1])
                else:
                    winners.append(entries[i])
            winners.append(entries[-1])
        else:
            for i in range(0, len(entries)):
                comp, _ = oracle(entries[i], entries[i+1])
                count += 1
                if comp:
                    winners.append(entries[i+1])
                else:
                    winners.append(entries[i])
        return tourney(winners, oracle, count)            


    def _GLDBS(self, model, oracle, num_iterations, x0, GL_r, GL_R,show_runs=False,D='Normal'):
        """
        GLDBS = Gradientless Descent with Binary Search

        =============== INPUTS ================
        model ......................... evaluator of our policy
        oracle ........................ Comparison Oracle
        num_iterations ................ number of iterations
        x0 ............................ initial iterate
        GL_r .......................... sampling radius lower bound
        GL_R .......................... sampling radius upper bound
        D ............................. distribution for sampling
        show_runs ..................... Boolean. Whether or not to show runs
        
        =============== OUTPUTS ================
        x_t .......................... final x for policy evaluation
        rewards ...................... list of rewards accumulated throughout process
        conv ......................... algorithm iteration number we converge at
        oracle_comparisons ........... number of oracle comparisons needed to converge
        
        Charles Stoksik July 29 2020
        """
        oracle_comparisons = 0
        rewards = np.zeros(num_iterations)
        K = int(np.log(GL_R/GL_r))
        #print(K)
        x_t = np.squeeze(x0)
        conv = None
        if K < 1:
            K = 1
        for t in range(num_iterations):
            comparison_vectors = []
            for k in range(K):
                r_k = 2**(-k)*GL_R
                v_k = r_k*np.random.randn(len(x0)) # guassian about 0 with var 1
                comparison_vectors.append(v_k)

            # need tournament setup here:
            x_t, round_of_oracles = self.flat_comparisons(x_t, comparison_vectors, oracle) # returns new best x_t
            
            if not conv:
                oracle_comparisons += round_of_oracles
                
            if show_runs:
                rewards[t] = model.render(x_t)
            else:
                rewards[t] = model(x_t)
            print(f'Current Rewards at step {t+1}: {rewards[t]}')

            if bool(model.reward_threshold):
                if rewards[t] > model.reward_threshold and conv == None: # if we converge early
                    conv = oracle_comparisons
                    print('************** Reward Threshold Exceeded **************')

        return x_t, rewards, conv, oracle_comparisons


    def _GLDFBS(self, model, oracle, num_iterations, x0, R, GLF_Q, D='Normal'):
        """
        Gradientless Descent with Fast Binary Search (in Mujoco it is ascent)

        ================= Inputs =================
        model ......................... evaluator of our policy
        oracle ........................ Comparison Oracle
        num_iterations ................ number of iterations
        x0 ............................ initial iterate
        R ............................. sampling radius upper bound
        GLF_Q ......................... condition number
        D ............................. distribution for sampling, currently no implementation of other options

        """
        oracle_comparisons = 0
        rewards = np.zeros(num_iterations)
        x_t = np.squeeze(x0)
        K = int(np.log(4*np.sqrt(GLF_Q)))
        H = int(len(x0)*GLF_Q)
        conv = None
        for t in range(1, num_iterations+1): # to avoid mod
            list_of_permutations = []
            if t % H == 0:
                R /= 2
            for k in range(-K, K+1):
                r_k = 2**(-k)*R
                v_k = r_k*np.random.randn(len(x0))
                list_of_permutations.append(v_k)

            # can replace with a more efficient tournament style eventually
            x_t, round_of_oracles = self.flat_comparisons(x_t, list_of_permutations, oracle) 
            
            if not conv: # if we have not converged
                oracle_comparisons += round_of_oracles
            
            rewards[t-1] = model(x_t)
            print(f'Rewards at step {t}: {rewards[t-1]}')

            if bool(model.reward_threshold):
                if rewards[t-1] > model.reward_threshold and conv == None: # if we converge early
                    conv = oracle_comparisons
                    print('************** Reward Threshold Exceeded **************')

        return x_t, rewards, conv, oracle_comparisons

    
    def __call__(self, model, oracle, num_iterations, x0, r, R, D='Normal',show_runs=False, algo_seed=0):
        
        # Facilitator
        np.random.seed(algo_seed)
        
        if self.fast:
            print('Using fast binary search for GLDBS')
            return self._GLDFBS(model, oracle, num_iterations, x0, r, R, D) 
        else:
            return self._GLDBS(model, oracle, num_iterations, x0, r, R,show_runs, D)

class signOPT:

    def signOPTGradEstimator(self, oracle, x_in, Z, m, d, r):
        '''This function estimates the gradient vector from m Comparison
        oracle queries, using the SignOPT method as detailed in Cheng et al's 
        'Sign-OPT' paper. 
        ================ INPUTS ======================
        Z ......... An m-by-d matrix with rows z_i uniformly sampled from unit sphere
        x_in ................. Any point in R^d
        r ................ Sampling radius.
        ================ OUTPUTS ======================
        g_hat ........ approximation to g/||g||

        23rd May 2020
    '''
        g_hat = np.zeros([d])
        for i in range(m):
            comp, _ = oracle(x_in, x_in + r*Z[i,:])
            g_hat += comp*Z[i,:]
        g_hat = g_hat/np.float(m)
        print(np.linalg.norm(g_hat))
        return g_hat
        
        
    def _signOPT(self, model, oracle, num_iterations, x0, m, d, alpha, r,show_runs=False):
        '''This function implements the descent part of signOPT, as detailed in:
        'signOPT: a query-efficient hard-label adversarial attack' 
        Cheng, Singh, Chen, Chen, Liu, Hsieh
        (2020) '''
         # initialize arrays
        rewards = np.zeros((num_iterations,1))
        x = np.squeeze(x0)
        Z = np.zeros((m, d))
        step_size = alpha
        conv = None # step number we exceed reward threshold at
            
        for t in range(0,num_iterations):
            for j in range(0, m):
                temp = np.random.randn(1, d) # randn is N(0, 1)
                Z[j,:] = temp/np.linalg.norm(temp) # normalize

            g_hat = self.signOPTGradEstimator(oracle, x, Z, m, d, r*1.01**(-t))
            x = x + step_size*1.001**(-t) * g_hat # Decaying stepsize model...
            if show_runs:
                rewards[t] = model.render(x)
            else:
                rewards[t] = model(x)
            print('current rewards at step', t+1, ':', rewards[t])
            print('step_size:', step_size*1.001**(-t))
            print('gradient norm:', np.linalg.norm(g_hat))
            if bool(model.reward_threshold):
                if rewards[t] > model.reward_threshold and conv == None: # if we converge early
                    conv = t+1
                    print('************** Reward Threshold Exceeded **************')
        if not conv:
            oracle_comparisons = num_iterations*m
        else:
            oracle_comparisons = conv*m
        return x, rewards, conv, oracle_comparisons

    def __call__(self, model, oracle, num_iterations, x0, m, d, alpha, r, show_runs = False,algo_seed=0):
        np.random.seed(algo_seed)
        return self._signOPT(model, oracle, num_iterations, x0, m, d, alpha, r,show_runs)


if __name__ == '__main__':
    
    s = SCOBO()
    #s = GLDBS()
    m = MujocoModel('Reacher-v2', 1000, -0.5, 0.5, reward_threshold=-8, msfilter='NoFilter')
    m1 = MujocoModel('Reacher-v2', 1000, -0.5, 0.5, reward_threshold=-8, msfilter='MeanStdFilter')
    #x, re, qu, conv = s(100, .2, np.zeros((22, 1)) + .3, 0.2/np.sqrt(2), m, GUIOracle(m, max_steps=100, M=1, gif_path_1='./videos/firstVideo.gif', gif_path_2='./videos/secondVideo.gif'), 11, 22, 5, False, -10)
    # print(x, re, qu, conv)

    params = {"kappa": 1, "delta": 0.5, "mu":1, "M":1} # .5 + min(delta, mu*|f(y) - f(x)|**(kappa-1)
    o = Oracle(m,params)

    results = {'NoFRewards':[], 'MFRewards':[], 'NoFConv':[], 'MFConv':[]}
    for i in range(5):
        x, re, qu, conv = s(100, 0.2, np.random.randn(22, 1), 0.02/np.sqrt(2), m, o, 26, 22, 11, False)
        results['NoFRewards'].append(max(re))
        results['NoFConv'].append(conv)
        x, re, qu, conv = s(100, 0.2, np.random.randn(22, 1), 0.02/np.sqrt(2), m1, o, 26, 22, 11, False)
        results['MFRewards'].append(max(re))
        results['MFConv'].append(conv)

    print(results)
    print('NoFilter average max rewards', np.mean(results['NoFRewards']))
    print('MS filter average resutls', np.mean(results['MFRewards']))
    #print(x, re, qu, conv)

    #s = signOPT()
    #s(m, o, 100, np.zeros((22,1)), 10, 22, .2, .1,show_runs=True)
    
    #s = GLDBS()
    #x, re, qu, conv = s(m, o, 500, np.random.randn(102,1), 0.5, 3, show_runs=True)

    # go = GUIOracle(m, )



    #print(c, xf)



## O.K. it works, but needs cleaning and the linesearch does not work for reacher...