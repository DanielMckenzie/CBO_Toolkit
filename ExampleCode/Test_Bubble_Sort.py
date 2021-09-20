#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 15:47:11 2021

@author: danielmckenzie
"""

import numpy as np
from oracle import Oracle
import matplotlib.pyplot as plt
from benchmarkfunctions import SparseQuadratic, MaxK
from utils import BubbleSort

func = SparseQuadratic(6, 4, 0)
oracle = Oracle(func)

v_arr = np.random.randn(5, 6)

sorted_v_arr, num_queries = BubbleSort(v_arr, oracle)

print(v_arr)
print('\n')
print(sorted_v_arr)
print('\n')

for i in range(5):
    print(func(sorted_v_arr[i,:]))
