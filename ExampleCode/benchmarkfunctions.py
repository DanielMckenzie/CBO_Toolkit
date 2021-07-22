"""
This module contains the following:
- Sparse Quadric
- Max-k-sum-squared
"""

import numpy as np
import sys


# TODO:
# Code up a non-sparse quadric test function.
# return np.dot(x,x)


class SparseQuadratic(object):
    """An implementation of the sparse quadric function."""

    def __init__(self, n, s, noiseamp):
        self.noiseamp = noiseamp / np.sqrt(n)
        self.s = s
        self.dim = n
        self.rng = np.random.RandomState()

    def __call__(self, x):
        if not len(x) == self.dim:
            raise ValueError('Error! Dimension of input must be'+str(self.dim)) 
        f_no_noise = np.dot(x[0:self.s], x[0:self.s])
        return f_no_noise + self.noiseamp * self.rng.randn()


class MaxK(object):
    """An implementation of the max-k-squared-sum function."""

    def __init__(self, n, s, noiseamp):
        self.noiseamp = noiseamp / np.sqrt(n)
        self.dim = n
        self.s = s
        self.rng = np.random.RandomState()

    def __call__(self, x):
        idx = np.argsort(np.abs(x))
        idx2 = idx[self.dim - self.s:self.dim]
        f_no_noise = np.dot(x[idx2], x[idx2]) / 2
        return f_no_noise + self.noiseamp * self.rng.randn()


# Week #2 - TASK #2.
# __________________
# a bit unsure about my code for the non-sparse Quadric function.
# i don't take in the parameter s as that defines my function's sparcity but I don't want it to be sparse.
# noiseamp is the amplitude of noise. I think I should still take this, and multiply it to self.rng.randn().
# then I still add it to f_no_noise as it accounts for noise.
# so I guess the only difference is I don't multiply x[0:self.s] with x[0:self.s], I multiply all of x with all of x.

### DM: Looks correct to me! Minor note: I recently realized I was using the 
### term "quadric" incorrectly. It refers to the graph of a quadratic function,
### but the function itself is a quadratic. I've made the change below.

class NonSparseQuadratic(object):
    """An implementation of the sparse quadric function."""

    def __init__(self, n, noiseamp):
        self.noiseamp = noiseamp / np.sqrt(n)
        self.dim = n
        self.rng = np.random.RandomState()

    def __call__(self, x):
        f_no_noise = np.dot(x, x)
        '''
        print('f_no_noise: ', f_no_noise)
        return f_no_noise + self.noiseamp * self.rng.randn()
        '''
        return f_no_noise + self.noiseamp * self.rng.randn()

