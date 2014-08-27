
# coding: utf-8

## Order of convergence: A numerical comparison

# In the previous notebook, we wrote a numeric algorithm to approximate the third derivative of a function up to second order (which is another way to say that our approximation method is of order $O(h^2)$.
# 
# We are going to repeat the previous algorithm, for many values of *h*, so that we can compare graphically what it means to have a second order method.
# 
# First, let's import the numeric packages:

# In[1]:

import  numpy as np
from matplotlib import pyplot as plt
from time import time


# In[2]:

#get_ipython().magic(u'matplotlib qt')


# in case the last command could not be executed in your *terminal*, or *notebook server*, you will have to show the ploted graphics by hand, which means that anytime you type something like
# 
# ```python
# plt.plot(x, y)
# ```
# 
# next you should type
# 
# ```python
# plt.show()
# ```
# 

# Let's define an array of different values for $N$, the number of subintervals in which we are going to divide the interval *[-1, 1]*

# In[8]:

N = np.arange(10, 201, 5)
N


# Recall that we want to calculate the third derivative of the gaussian curve $e^{-x^2}$ in the interval *(-1, 1)*, and that we found a second order approximation, given by the formula:
# 
# $$
# f^{(3)} = \frac{-1/2 f_{i - 2} + f_{i - 1} - f_{i + 1} + 1/2 f_{i + 2}}{h^3}
# $$

# In[12]:

errors = []
lapses = []

for n in N:
    t0 = time()
    x = np.linspace(-1, 1, n + 1)
    y = np.exp(-x**2)
    
    idx = np.arange(2, n - 1)
    h = 2. / n
    dy = (-0.5 * y[idx - 2] + y[idx - 1] - y[idx + 1] + 0.5 * y[idx + 2]) / h**3
    
    xx = x[idx]
    df = 4 * xx * (-2 * xx**2 + 3) * np.exp(-xx**2)
    t1 = time()

    errors.append(np.std(df - dy))
    lapses.append(t1 - t0)


# See the previous section if you have trouble understanding what the previous algorithm does.
# 
# Lets see a nice graphical representation of the errors

# In[13]:

plt.plot(errors, '.')


# If you are running *ipython* you would need to type 
# 
# ```python
# plt.show()
# ```
# 
# so you can see the previous image. Apparently, the errors follow an exponential decrement. Let's see what happens when we plot the errors versus the subintervals length.

# In[15]:

h = 1. /N
plt.plot(h, errors, '.')


# That seems to be a parabola!. That's the meaning of our method being of *second order in h*.
# 
# Finnally, let me show you how to interpolate the previous data with a least squares parabola, so we can confirm numerically that indeed, our data is followoing a second order rule.

# In[31]:

from scipy.linalg import lstsq

errors = np.array(errors)

model = np.ones((errors.size, 3))
model[:, 1] = h
model[:, 2] = h**2

(a, b, c), residues, rank, sv = lstsq(model, errors)

print 'least squares model for the data:'
print '{0} + {1} h + {2} h^2'.format(a, b, c)

print 'residues:', residues


# Note that the residues are small, so we can be confident that our model fits well. 
# 
# Having calculated the interpolant parabola, let's plot everythin together:

# In[45]:

plt.plot(h, errors, '.b', h, a + b*h + c*h**2, '-r')
plt.title('Error in the third derivative formula')
plt.xlabel('Nodes separation')
plt.ylabel('Standard deviation')
plt.legend(['Error', 'Least squares approximation'], loc='upper left')
plt.grid()


# As you can see in the data, the parabola fits the errors in the approximation. This is what the order of approximation is about.
# 
# Note that the time of execution of our code should be inversly proportional to the size of the nodal separation. So that the order of approximation can be useful in order to determine for how long our code should run to make a good approximation.
# 
# **Exercise** 
# 
# 1. Can you make a plot of *h vs time*, for the *time* the algorithm runs and for different values of *h*?. 
# 2. Can you spot any rule?
# 3. With the data you have collected so far, if we require our method to have a standard deviation no greater than $10^{-9}$, at least how many nodes do we need? For how long do you predict that your algorithm would run?

# In[ ]:



