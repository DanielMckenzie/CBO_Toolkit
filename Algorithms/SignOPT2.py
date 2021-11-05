#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 11:19:28 2021

@author: danielmckenzie

Testing implementing SignOPT as a class.
"""

from ExampleCode.base import BaseOptimizer
import numpy as np

from ExampleCode.utils import random_sampling_directions

class SignOPT(BaseOptimizer):
    def __init__(self, oracle, query_budget, x0, m, step_size, r, debug=False, function=None):
        self.oracle = oracle # This must be a comparison oracle.
        self.query_budget = query_budget
        self._queries = 0
        self.x = x0
        self.m = m  # number of directions sampled per iteration.
        self.d = len(x0)
        self.step_size = step_size
        self.r = r
        self.debug_status = debug
        self._function = function
        
        if self._function is not None:
        # In development, we'll have access to both the oracle and the function.
        # In practice this will not be the case.
            self.f_vals = [self._function(x0)]
            self._queries += 1
        
        self.x_vals = [x0]
        
    def signOPT_grad_estimate(self, Z, x_in):
        """
        Estimate the gradient from comparison oracle queries.
        See Sign-OPT: A Query Efficient Hard-Label Adversarial Attack"
        by Minhao Cheng et al
        """
        
        g_hat = np.zeros(self.d)
        for i in range(self.m):
            comparison = self.oracle(x_in, x_in + self.r*Z[i,:])
            if self.debug_status:
                print('comparison is' + str(comparison))
            self._queries += 1
            g_hat += comparison*Z[i,:]
        
        g_hat = g_hat/ self.m
        if self.debug_status:
            print(['Gradient is ', g_hat])
        
        return g_hat
    
    def step(self):
        # rows of Z are sample directions.
        """
        Z = np.random.randn(self.m,self.d)
        """
        # *********
        # call the UTILS function.
        output = random_sampling_directions(self.m, self.d, 'gaussian')
        Z = output
        # *********
        # TODO: Allow for different distributions.
        
        g_hat = self.signOPT_grad_estimate(Z, self.x)
        self.x -= self.step_size*g_hat
        self.x_vals.append(self.x)
        if self._function is not None:
            self.f_vals.append(self._function(self.x))
            
        if self.reachedFunctionBudget(self.query_budget, self._queries):
            # if budget is reached return current iterate.
            # solution, list of all function values, termination.
            if self._function is not None:
                return self.x, self.f_vals, 'B', self._queries
            else:
                return None, None, 'B', None
        return self.x, self.f_vals, False, self._queries
        
        
    
    
            
        
        

