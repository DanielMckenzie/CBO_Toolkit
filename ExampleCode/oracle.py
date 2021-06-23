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

#from ExampleCode.benchmarkfunctions import SparseQuadric
from benchmarkfunctions import SparseQuadric, MaxK
#from ExampleCode.benchmarkfunctions import MaxK

# suppose the function is something like f:R->R s.t. f(x) = 3x^2+2.
# then the function would be like....
'''
def function_f(x):
    output = 3*(x*x)+2
    return output
'''


class Oracle:

    def __init__(self, f_x):
        # f_x is a function.
        # we have to define what that function does... right?
        self.f_x = f_x

    def __call__(self, x, y):
        if self.f_x(y) - self.f_x(x) < 0:
            # print(-1)
            return -1
        elif self.f_x(y) - self.f_x(x) > 0:
            # print(1)
            return 1


# ---------
# define the function we will initialize the class with.
def opt_function(input_x):
    # code whatever the function is.
    # TESTING (random function example):
    output = input_x + 2
    return output
    # return 0


# create instance of the class. feed function as parameter.
instance_1 = Oracle(opt_function)

'''
n = 20000 # problem dimension
s_exact = 200   # True sparsity
noise_amp = 0.001  # noise amplitude
obj_func = MaxK(n, s_exact, noise_amp)
instance_1 = Oracle(obj_func)
'''

# instance_1 = Oracle(obj_func)
# comparison_oracle = instance_1.call(2,3)
# print(comparison_oracle)

# __call__ method.
comparison_oracle = instance_1(4, 9)
# should output 1.
print(comparison_oracle)
comparison_oracle_2 = instance_1(8, 5)
# should output -1.
print(comparison_oracle_2)


#########################################################################
'''
PROBLEM 2.
'''

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
a_k can be 0.01.
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
    for k in range(10000):
        if k == 0:
            x_k = np.random.rand(1, n)
            x_k = x_k[0]
            print(x_k)
            # initial_x_k = x_k
            # print(x_k)
            
            ################
            ## DM: I would handle the k=0 case (initialization) outside of
            ## the loop.
            ###############

        # print('x_k: ', x_k)
        list_of_xk.append(x_k)
        # print(len(list_of_xk))
        
        f_vals.append(defined_func(x_k))
        
         #########
         ## DM: Added logging of f_vals
         #########
        
        
        if k > 0:
            if list_of_xk[k].all() == list_of_xk[k - 1].all():
                # print('same')
                count_same += 1
        # n = 10
        # a_k = .1
        random_direction = random.randint(0, n - 1)
        # print('\n')
        # print(random_direction)
        s_k = np.zeros(n, int)
        #####################
        ## DM: Might be more robust to create a zero array of floats, instead
        ## of ints.
        #####################
        ## DM: Try the following sampling distributions:
        ## 1. Gaussian: s_k = np.random.randn(n,1)
        ## 2. Uniform from sphere: s_k = np.random.randn(n,1) then s_k/ ||s_k||
        ## 3. Rademacher: [s_k]_{i} = +1 or -1 with probability 50% 
        ####################
        
        s_k[random_direction] = 1
        # print(s_k)
        # s_k_trans = np.array([s_k]).T

        # print(x_k + np.dot(a_k, s_k))

        # break.
        x_plus = x_k + np.dot(a_k, s_k)
        # print('x+ : ', x_plus)
        x_minus = x_k - np.dot(a_k, s_k)
        # print('x- : ', x_minus)
        
        ######################
        ## DM: a_k is a scalar right? Won't make much difference but we can 
        ## just use a_k*s_k
        ######################

        # compute comparisons using the Comparison Oracle.
        # compute 2 comparisons to determine the argmin.
        new_instance_1 = Oracle(defined_func)
        first_comparison = new_instance_1(x_k, x_plus)
        if first_comparison == -1:
            #print('hey')
            second_comparison = new_instance_1(x_plus, x_minus)
            if second_comparison == -1:
                argmin = x_minus
                # print('MINUS')
                # print('argmin: ', argmin)
                x_k = argmin
            elif second_comparison == +1:
                argmin = x_plus
                # print('PLUS')
                # print('argmin: ', argmin)
                x_k = argmin
        elif first_comparison == +1:
            # print('hi')
            second_comparison = new_instance_1(x_k, x_minus)
            if second_comparison == -1:
                argmin = x_minus
                # print('MINUS')
                # print('argmin: ', argmin)
                x_k = argmin
            elif second_comparison == +1:
                argmin = x_k
                # print('X_K')
                # print('argmin: ', argmin)
                x_k = argmin
        # else:
        # print('neither')

        # the argmin is x_k+1.
        # we will now set the argmin to be the new x_k for the next iteration.
        # x_k = argmin

    # once we reach the end of k's iterations, we want to return x_k.
    # print(count_same)
    # print(x_k)

    return x_k, f_vals


#########################################################################
'''
PROBLEM 3.
'''

n_def = 20000  # problem dimension
s_exact = 200  # True sparsity
noise_amp = 0.001  # noise amplitude
# initialize objective function
obj_func_1 = SparseQuadric(n_def, s_exact, noise_amp)
obj_func_2 = MaxK(n_def, s_exact, noise_amp)

# testing with SPARSE QUADRIC FUNCTION.
param1 = n_def
#param2 = 0.1
param2 = 1.5
trial1_STP, f_vals = stp(param1, param2, obj_func_1)
print(trial1_STP)
'''
[0.92863736 0.67880925 0.55654282 ... 0.24465763 0.75835699 0.41617807]
'''

print('---------')

plt.plot(f_vals)
plt.show()

###############
## DM: In optimization, use a log scale on the y-axis is usually more 
## informative than a linear scale. This is because most good algorithms 
## exhibit exponential convergence (weird convention: we call this linear 
## convergence) for simple functions 
###############

plt.semilogy(f_vals)
plt.show()

# testing with MAXK FUNCTION.
trial2_STP, f_vals = stp(param1, param2, obj_func_2)
print(trial2_STP)
'''
[0.63684335 0.01983142 0.82694289 ... 0.29871574 0.05529189 0.54062174]
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
obj_func_1 = SparseQuadric(n, s_exact, noise_amp)
obj_func_2 = MaxK(n, s_exact, noise_amp)
'''
