# isha slavin.
# i am using this file to test different ideas i have (it's basically my 'scratch work').

import random
import numpy as np
from numpy import linalg as LA

n = 10
'''
n = 10
list_elements = np.random.randn(n, 1)
print(list_elements)

x = 0
for elem in list_elements:
    # print(elem)
    # print(elem * elem + elem - elem)
    x += elem
print('\n')
print(x)

print('\n')
print('*********')
trial1 = np.zeros(n, int)
print(trial1)
new_list_elements = np.random.randn(n)
print(new_list_elements)

# euclidean norm.
sum = 0
for element in new_list_elements:
    element_squared = element * element
    sum += element_squared
print(sum)
sum_sqrt = sum ** 0.5
print(sum_sqrt)
print(new_list_elements / sum_sqrt)

# list_of_nums = [2, 4, 5, 9]
'''

'''
euclidean norm.
'''
# formula: ||x_n|| = sqrt(x_n_1^2 + x_n_2^2 + ... + x_n_n^2).
# in this case: sqrt(2^2 + 4^2 + 5^2 + 9^2) = sqrt(4 + 16 + 25 + 81) = sqrt(126) = approx. 11.2?
# for elemet

'''
Case 2: UNIFORM FROM SPHERE.
'''
s_k = np.random.randn(n)
# print('old s_k: ', s_k)
''' Case 2a: EUCLIDEAN - NORM. '''
# """
# formula: ||x_n|| = sqrt(x_n_1^2 + x_n_2^2 + ... + x_n_n^2).
# let's calculate ||s_k||.
sum = 0
print('***********')
print('***********')
print('***********')
print('***********')
print('***********')
# print('trying function -> new s_k: ', s_k / LA.norm(s_k))
for elem in s_k:
    elem_squared = elem * elem
    sum += elem_squared
sum_sqrt = sum ** 0.5
s_k_norm = sum_sqrt
# print('s_k norm: ', s_k_norm)
# print('trying function -> new s_k: ', )
s_k = s_k / s_k_norm
# print('new s_k: ', s_k)
# print('trying function -> new s_k: ', )

# """
''' Case 2b: P - NORM. '''
"""
# formula: ||x_n|| = (|x_n_1|^p + |x_n_2|^p + ... + |x_n_n|^p)^(1/p).
# let's calculate ||s_k|| when p = n, the dimension of our vector.
sum = 0
for elem in s_k:
    elem_p = elem ** n
    sum += elem_p
sum_one_over_n = sum ** (1/n)
s_k_norm = sum_one_over_n
print('s_k norm: ', s_k_norm)
s_k = s_k/s_k_norm
print('new s_k: ', s_k)
# NOTE: this way does NOT work. Since I am raising every element of s_k to the nth power, the sum approaches infinity
# and thus, s_k approaches 0. This method returns a vector (length n) of zeros. 
"""

'''
Case 3: RADEMACHER.
'''
# Rademacher: [s_k]_{i} = +1 or -1 with probability 50%.
s_k = []
count_positive1 = 0
count_negative1 = 0
for i in range(n):
    rand_choice = random.choice([-1, 1])

    if rand_choice == 1:
        count_positive1 += 1
    else:
        count_negative1 += 1
    # print(str(i) + ': ', rand_choice)
    s_k.append(rand_choice)
'''
print(s_k)
print(len(s_k))
print(count_positive1)
print(count_negative1)
'''


#############################
# Now I want to test out some things for Task #2 (Week 2).
class SparseQuadric(object):
    """An implementation of the sparse quadric function."""

    def __init__(self, n, s, noiseamp):
        self.noiseamp = noiseamp / np.sqrt(n)
        self.s = s
        self.dim = n
        self.rng = np.random.RandomState()

    def __call__(self, x):
        f_no_noise = np.dot(x[0:self.s], x[0:self.s])
        print('f_no_noise: ', f_no_noise)
        return f_no_noise + self.noiseamp * self.rng.randn()


n = 20000  # problem dimension
s_exact = 200  # True sparsity
noiseamp = 0.001  # noise amplitude
obj_func = SparseQuadric(n, s_exact, noiseamp)  # initialize objective function
# call the function you just initialized.
x0 = np.random.randn(n)
print(x0)
print(len(x0))
array = obj_func(x0)
print('sparse.')
print(array)

# _________
# testing....
list1 = [2, 3, 4]
list2 = [3, 4, 5]
# 2*3 + 3*4 + 4*5 = 6 + 12 + 20 = 38.
lists_mult = np.dot(list1, list2)
#print(lists_mult)


class NonSparseQuadric(object):
    """An implementation of the sparse quadric function."""

    def __init__(self, num):
        self.dim = num
        self.rng = np.random.RandomState()

    def __call__(self, x):
        f_no_noise = np.dot(x, x)
        '''
        print('f_no_noise: ', f_no_noise)
        return f_no_noise + self.noiseamp * self.rng.randn()
        '''
        return f_no_noise


print('non-sparse.')
obj_func_2 = NonSparseQuadric(n)
array1 = obj_func_2(x0)
print(array1)

