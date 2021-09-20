#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 08:30:57 2021

@author: danielmckenzie
"""

import numpy as np
from base import BaseOptimizer
from utils import BubbleSort
from scipy.linalg import sqrtm


class CMA(BaseOptimizer):
    '''
    Simple version of CMA
    '''

    def __init__(self, oracle, query_budget, x0, lam, mu, sigma,
                 function=None):
        '''
        lambda is a reserved word in Python, so lam = lambda.
        '''
        self.m = x0
        self.dim = len(x0)
        self.oracle = oracle
        self.C = np.eye(len(x0))  # Covariance matrix. Init to the identity.
        self.lam = lam
        self.mu = mu
        self.sigma = sigma
        self.p_sigma = 0
        self.p_c = 0
        self.queries = 0
        self.function = function
        
        # For the following parameters we use the defaults
        self.mu_eff = 1
        self.c_mu = 0.5 
        self.c_sigma = 0.5

    def step(self):
        # The following code samples self.lam vectors from a normal dist with
        # mean self.m and covariance self.sigma^2 * self.N

        Yg = np.random.multivariate_normal(self.m, self.C, self.lam)
        Xg = self.m + self.sigma*Yg
        # print(Xg.shape)
        # for i in range(self.mu):
        #     print(self.function(Xg[i,:]))
        # print('\n')
        # The next line sorts according to function values
        Sorted_Xg, num_queries = BubbleSort(Xg, self.oracle)
        Sorted_Yg = (Sorted_Xg - self.m)/self.sigma
        print(Sorted_Yg)

        # In the next line, we use weights w_i = 1/mu for all i
        y_w = np.sum(Sorted_Yg[0:self.mu, :], axis=0)/self.mu
        print(y_w)

        # Update mean
        self.m += self.sigma*y_w
        
        # Update step size
        # C_half_inverse = np.linalg.matrix_power(sqrtm(self.C + 0.01*np.eye(self.dim)), -1)
        # self.p_sigma = (1-self.c_sigma)*self.p_sigma + np.sqrt(self.c_sigma*
                       # (2-self.c_sigma)*self.mu_eff)*np.dot(C_half_inverse, y_w)

        # Update covariance matrix
        # temp1 = (self.c_mu/self.mu)*np.dot(Sorted_Yg[0:self.mu,:].T, Sorted_Yg[0:self.mu,:])
        # self.C = (1-self.c_mu/self.mu)*self.C + temp1
        
        return self.function(self.m)