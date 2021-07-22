#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 11:19:28 2021

@author: danielmckenzie

Testing implementing SignOPT as a class.
"""

from base import BaseOptimizer
import numpy as np

class SignOPT(BaseOptimizer):
    def __init__(self, function_budget, x0, m, step_size, r):
        self.function_budget = function_budget
        self.x0 = x0
        self.m = m  # number of directions sampled per iteration.
        self.d = len(x0)
        self.step_size = step_size
        self.r = r
        
        self.f_vals = []
        self.x_vals = []
        
    def signOPT_grad_estimate(self, x_in, Z, r):
        '''
        Estimate the gradient from comparison oracle queries.
        See Sign-OPT: A Query Efficient Hard-Label Adversarial Attack"
        by Minhao Cheng et al
        '''
        
        
