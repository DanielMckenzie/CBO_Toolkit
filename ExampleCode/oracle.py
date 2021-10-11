# isha slavin.
# WEEK 1 Tasks.


#########################################################################
'''
PROBLEM 1.
'''

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from numpy import linalg as LA

#from ExampleCode.benchmarkfunctions import SparseQuadratic, MaxK
#from ExampleCode.benchmarkfunctions import SparseQuadratic, MaxK
# suppose the function is something like f:R->R s.t. f(x) = 3x^2+2.
# then the function would be like....
'''
def function_f(x):
    output = 3*(x*x)+2
    return output
'''


# ---------

# class which represents Comparison Oracle.
class Oracle:

    def __init__(self, f_x):
        # f_x is a function.
        # we have to define what that function does... right?
        self.f_x = f_x

    # calls the function f_x the class is initialized with.
    # function takes in inputs x and y.
    #   returns +1 if f(y)-f(x) > 0.
    #   returns -1 if f(y)-f(x) < 0.
    def __call__(self, x, y):
        if self.f_x(y) - self.f_x(x) < 0:
            # print(-1)
            return -1
        elif self.f_x(y) - self.f_x(x) > 0:
            # print(1)
            return 1
        else:
            return 0


# class which represents Comparison Oracle.
class Oracle_pycutest:

    def __init__(self, f_x):
        # f_x is a function.
        # we have to define what that function does... right?
        self.f_x = f_x

    # calls the function f_x the class is initialized with.
    # function takes in inputs x and y.
    #   returns +1 if f(y)-f(x) > 0.
    #   returns -1 if f(y)-f(x) < 0.
    def __call__(self, x, y):
        if self.f_x(y) - self.f_x(x) < 0:
            # print(-1)
            return -1
        elif self.f_x(y) - self.f_x(x) > 0:
            # print(1)
            return 1
        else:
            return 0

# ---------
# TESTING EXAMPLE....
# define the function we will initialize the class with.
# this testing function f(x) = x+2.
def opt_function(input_x):
    # code whatever the function is.
    # TESTING (random function example):
    output = input_x + 2
    return output
    # return 0
# ---------


# create instance of the class; feed TESTING EXAMPLE function as parameter.
instance_1 = Oracle(opt_function)
# ignore the following lines:
'''
# instance_1 = Oracle(obj_func)
# comparison_oracle = instance_1.call(2,3)
# print(comparison_oracle)
'''

# __call__ method.
comparison_oracle = instance_1(4, 9)
# should output 1.
print(comparison_oracle)
# correct.
comparison_oracle_2 = instance_1(8, 5)
# should output -1.
print(comparison_oracle_2)
# correct.


# class which represents Comparison Oracle.
# takes into account noise - runs 3 times and uses the majority response.
# also takes into account a margin of error defined by the user.
#   if the difference in function values falls into the margin of error, they are considered equal.
class Oracle_2:

    def __init__(self, f_x):
        # f_x is a function.
        # we have to define what that function does... right?
        self.f_x = f_x
        self.count_plus = 0
        self.count_minus = 0
        self.count_equal = 0
        self.list_of_outputs = []

    # calls the function f_x the class is initialized with.
    # function takes in inputs x and y.
    # input margin_of_error: (type = FLOAT) accounts for noise of Oracle.
    #   user defines what they want the margin of error to be. (ex.: 0.005.)
    #   returns +1 if f(y)-f(x) > + margin_of_error.
    #   returns -1 if f(y)-f(x) < - margin_of_error.
    #   returns 0 if |f(y)-f(x)| < + margin_of_error.
    #   returns NONE if 1, -1, 0 are all found by the Oracle.
    def __call__(self, x, y, margin_of_error):
        for i in range(3):
            if self.f_x(y) - self.f_x(x) < -1 * margin_of_error:
                self.count_minus += 1
                self.list_of_outputs.append(-1)
                # print(-1)
                # return -1
            elif self.f_x(y) - self.f_x(x) > margin_of_error:
                self.count_plus += 1
                self.list_of_outputs.append(1)
                # print(1)
                # return 1
            else:
                self.count_equal += 1
                self.list_of_outputs.append(0)
        print('\n')
        print('+: ', self.count_plus)
        print('-: ', self.count_minus)
        print('*: ', self.count_equal)
        print('\n')
        # returns....
        if self.count_minus > 1:
            return -1
        elif self.count_plus > 1:
            return 1
        elif self.count_equal > 1:
            return 0
        else:
            return None


            # return 0

instance_2 = Oracle_2(opt_function)
# difference does not fall within margin of error (1).
# output should be -1.
comparison_oracle_3 = instance_2(9, 4, 1)
print('3: ', comparison_oracle_3)
instance_3 = Oracle_2(opt_function)
# difference falls within margin of error (10).
# output should be 0.
comparison_oracle_4 = instance_3(4, 9, 10)
print('4: ', comparison_oracle_4)



#########################################################################
'''
PROBLEM 2.
'''

# ---------------------------
'''
For this question, I think this is what I have to do.
Set k to be a really large # (like 100,000 or something).
A_n is the Set of positive stepsizes (maybe make them always increase by 0.01? But at first, keep the same step size?). 
- (Top of page 7 above algorithm for step size idea.)
D is the Probability distribution.
Enter a loop for k.
While k is less than 100,000:
    Generate a random vector (in Probability distribution?).
    Get x+ and x-. Call the ORACLE class on all 3.
    - Figure out how to get the smallest of the 3.
'''
'''
minimize f(x) where f: R^n -> R.
positive spanning set: {±ei : i = 1, 2, . . . , n}.
stepsize: let's take it to be .01.
x_0 has to be an n-dimensional vector.
a_k can be 0.01 or maybe 0.1.
s_k has to be a certain direction (i.e. one of the n entries of the column vector has to be a 1).
From what I understand, we can choose D to be the standard basis.
User inputs a value of n.
FUNCTION inputs for Comparison - Based version of the Stochastic Three Point (STP) method:
    - Dimension n.
    -   At the first iteration, randomly generate an array, called x_0, of this dimension.
    - stepsize a_k = .01.
    - probability distribution....
    -   I think I can use standard basis of dimension n for this set of arrays.
    -   So at each k, the randomly generated s_k will be e_k (ex. e_3 = [0, 0, 1, 0, 0, ....].
To generate an s_k at each kth iteration, I will first generate a random # between 0 and n-1.
Then, I will create an array of dimension n.
I will alter the randomly-generated element of the array to become a 1.
This will be my randomly generated direction sk.
'''
# Function parameters: n, a_k, x_0, defined_function.
'''
n = 10
a_k = .1
random_direction = random.randint(0, n-1)
print('\n')
print(random_direction)
s_k = np.zeros(n, int)
s_k[random_direction] = 1
print(s_k)
s_k_trans = np.array([s_k]).T
#print(s_k_trans)
'''
# now, to generate x_k.
# we have x_0.
# we can say when k = 0, then randomly generate an n-dimensional vector x_n.
# then, multiply step-size a_k by the directional vector s_k.
# then, input 3 things into f, and figure out how to work around argmin to in fact use the ORACLE class.
# ---------------------------


# FUNCTION that implements Comparison - Based version of Stochastic Three Point method (STP).
def stp(num, a_k, defined_func):
    list_of_xk = []
    ##################
    ## DM: A good way to check if an optimization algorithm is working is to
    ## ensure that the objective function is decreasing. (Of course in true 
    ## comparison-based optimization we wouldn't have access to objective function 
    ## values, but this is useful for debugging). Later we will talk 
    ## about the rate at which it decreases.
    ##################
    f_vals = []
    n = num
    count_same = 0
    # iterate through values of k. (Future idea: take in # of iterations as function argument?)
    x_k = np.random.rand(1, n)
    x_k = x_k[0]
    print(x_k)
    # this is the initial value x_0.
    for k in range(10000):
        '''
        #if k == 0:
            #x_k = np.random.rand(1, n)
            #x_k = x_k[0]
            #print(x_k)
            # this is the initial value x_0.
            ################
            ## DM: I would handle the k=0 case (initialization) outside of
            ## the loop.
            ###############
        '''
        list_of_xk.append(x_k)
        f_vals.append(defined_func(x_k))
         #########
         ## DM: Added logging of f_vals
         #########
        # ---------
        # TESTING....
        if k > 0:
            if list_of_xk[k].all() == list_of_xk[k - 1].all():
                # print('same')
                count_same += 1
                # counting how many times x_k's are equal.
        # ---------
        # generate random direction vector s_k.
        # ---------
        '''
        Case 0: ORIGINAL.
        '''
        # to test this case, comment out the following line:
        """
        random_direction = random.randint(0, n - 1)
        s_k = np.zeros(n, int)
        s_k[random_direction] = 1
        """
        # ---------
        '''
        Case 1: GAUSSIAN.
        '''
        # to test this case, comment out the following line:
        #"""
        s_k = np.random.randn(n)
        #"""
        # ---------
        '''
        Case 2: UNIFORM FROM SPHERE.
        '''
        # to test this case, comment out the following line:
        """
        s_k = np.random.randn(n)
        # print('old s_k: ', s_k)
        """
        ''' Case 2a: EUCLIDEAN - NORM. '''
        # to test this case, comment out the following line:
        """
        # formula: ||x_n|| = sqrt(x_n_1^2 + x_n_2^2 + ... + x_n_n^2).
        # let's calculate ||s_k||.
        sum = 0
        for elem in s_k:
            elem_squared = elem*elem
            sum += elem_squared
        sum_sqrt = sum ** 0.5
        s_k_norm = sum_sqrt
        # print('s_k norm: ', s_k_norm)
        s_k = s_k/s_k_norm
        # print('new s_k: ', s_k)
        """
        # OR, there is a Python function which calculates the Euclidean vector norm.
        # we can use this instead of calculating it with a For loop, which has high complexity and takes longer.
        """
        s_k_norm = LA.norm(s_k)
        s_k = s_k / s_k_norm
        """
        ''' Case 2b: P - NORM. '''
        # to test this case, comment out the following line:
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
        # ---------
        '''
        Case 3: RADEMACHER.
        '''
        # Rademacher: [s_k]_{i} = +1 or -1 with probability 50%.
        # to test this case, comment out the following line:
        """
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
        """
        # ---------
        #####################
        ## DM: Might be more robust to create a zero array of floats, instead
        ## of ints.
        #####################
        ## DM: Try the following sampling distributions:
        ## 1. Gaussian: s_k = np.random.randn(n,1)
        ## 2. Uniform from sphere: s_k = np.random.randn(n,1) then s_k/ ||s_k||
        ## 3. Rademacher: [s_k]_{i} = +1 or -1 with probability 50% 
        ####################
        # generate x+.
        x_plus = x_k + np.dot(a_k, s_k)
        # x_plus = x_k + a_k * s_k
        # generate x-.
        x_minus = x_k - np.dot(a_k, s_k)
        # x_minus = x_k - a_k * s_k
        ######################
        ## DM: a_k is a scalar right? Won't make much difference but we can 
        ## just use a_k*s_k
        ## NOTE: I tried switching to using a_k*s_k, it doesn't work in Case 3 for Rademacher. However it works
        ## fine for the other cases.
        ######################
        # compute comparisons using the Comparison Oracle.
        # compute 2 comparisons to determine the argmin.
        new_instance_1 = Oracle(defined_func)
        first_comparison = new_instance_1(x_k, x_plus)
        if first_comparison == -1:
            # x_plus is smaller.
            second_comparison = new_instance_1(x_plus, x_minus)
            if second_comparison == -1:
                # x_minus is smaller.
                argmin = x_minus
                x_k = argmin
            elif second_comparison == +1:
                # x_plus is smaller.
                argmin = x_plus
                x_k = argmin
        elif first_comparison == +1:
            # x_k is smaller.
            second_comparison = new_instance_1(x_k, x_minus)
            if second_comparison == -1:
                # x_minus is smaller.
                argmin = x_minus
                x_k = argmin
            elif second_comparison == +1:
                # x_k is smaller.
                argmin = x_k
                x_k = argmin
        # else:
        # print('neither')
    # once we reach the end of k's iterations, we want to return x_k.
    return x_k, f_vals


#########################################################################
'''
PROBLEM 3.
'''

# We will test our C-STP function on Sparse Quadric and Max K functions (from benchmarkfunctions.py).
# ---------
# taken from Example_isha.py:
n_def = 20000  # problem dimension.
s_exact = 200  # True sparsity.
noise_amp = 0.001  # noise amplitude.
# ---------
# initialize objective functions.
#obj_func_1 = SparseQuadratic(n_def, s_exact, noise_amp)
#obj_func_2 = MaxK(n_def, s_exact, noise_amp)

# testing with SPARSE QUADRIC FUNCTION.
# n:
param1 = n_def
# a_k:
param2 = 0.1
#trial1_STP, f_vals = stp(param1, param2, obj_func_1)
#print(trial1_STP)
'''
ex. output:
[0.92863736 0.67880925 0.55654282 ... 0.24465763 0.75835699 0.41617807]
'''
print('---------')

# plot the decreasing function.
#plt.plot(f_vals)
#plt.show()
###############
## DM: In optimization, use a log scale on the y-axis is usually more 
## informative than a linear scale. This is because most good algorithms 
## exhibit exponential convergence (weird convention: we call this linear 
## convergence) for simple functions 
###############
#plt.semilogy(f_vals)
#plt.show()

# testing with MAX K FUNCTION.
#trial2_STP, f_vals = stp(param1, param2, obj_func_2)
#print(trial2_STP)
'''
ex. output:
[0.63684335 0.01983142 0.82694289 ... 0.29871574 0.05529189 0.54062174]
'''


#########################################################################
"""
PROBLEM 4.
"""

""" Possible Comparison - Based Optimization Algorithm, found in literature. """
# ---------
# Reference EIGHT (8) of article 'Stochastic Three Points Method for Unconstrained Smooth Minimization':
#   An Accelerated Method for Derivative-Free Smooth Stochastic Convex Optimization.
#   Link: 'https://arxiv.org/pdf/1802.09022.pdf'.
# ---------
# NOTES:
'''
It seems like the proposed algorithm (Algorithm #2.2 on pg. 16) can be made into a comparison - based optimization alg.
The alg. incorporates calculating an argmin.
Week 1 tasks (specifically task 3) focused around translating an argmin calculation into a comparison oracle issue.
    Thus I think there is potential for this alg. to be a CBO alg. However, the contents of the argmin calculation seem
    very non-trivial so I'm worried about level of complexity / usability of this algorithm.
'''


#########################################################################
'''
SCRATCH WORK / NOTES.
'''
# now I need to input x_k, x_plus, and x_minus into the function.
# the name of the function, for now, is called defined_function(x).
'''
what I actually need to do is call class Oracle, inputting the function defined_function.
then I need to call the call function of oracle. 
what I should practice right now is inputting something like maxK into class Oracle(), but
    for now, let me work on how I would get the argmin using comparisons.

ORACLE:
    if f(y)-f(x) > 0 : +1.
    if f(y)-f(x) < 0 : -1.
    I have x_k, x+, x-.
    input (x, y) = (x_k, x+).
        if -1: we know f(x+) is smaller.
        input (x, y) = (x+, x-).
            if -1: x- is smaller. ARGMIN.
            if +1: x+ is smaller. ARGMIN.
        if +1: we know f(x_k) is smaller.
        input (x, y) = (x_k, x-).
            if -1: x- is smaller. ARGMIN.
            if +1: x_k is smaller. ARGMIN.
        
            
    input x+, x-.
    input x-, x_k.
    
    
'''
# print().


'''
question 3.
'''
# Deal with this problem at the end.

'''
n = 20000  # problem dimension
s_exact = 200  # True sparsity
noise_amp = 0.001  # noise amplitude
# initialize objective function
obj_func_1 = SparseQuadratic(n, s_exact, noise_amp)
obj_func_2 = MaxK(n, s_exact, noise_amp)
'''


#########################################################################
