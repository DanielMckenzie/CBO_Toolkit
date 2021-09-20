#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 18:22:32 2021

@author: danielmckenzie

Wrapper
"""
from oracle import Oracle

def WrapObFunc(Algorithm, ObjFunc):
    '''
    Simple Wrapper that takes an objective function and turns it into a 
    comparison-based oracle for use with a CBO algorithm.
    CBO algorithm must be an instance of the BaseOptimizer class.
    
    '''
    Function_Oracle = oracle(ObjFunc)
    flag = False
    
    while not flag:
        feval, _, Status = 
    

