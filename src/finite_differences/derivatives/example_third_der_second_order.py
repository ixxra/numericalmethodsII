import numpy as np
from matplotlib import pyplot as plt

def third_der(y, h):
    '''
    third_der(y, h): Calculates a third derivative approximation for a given y-sample, comming from data with x-nodal separation h. It uses the formula

    f^{(3)} = (-0.5*f_{i-2} + f_{i-1} - f_{i+1} + 0.5*f_{i+2})/h**3 

    To make a second order approximation.
    '''
    h = float(h)
    idx = np.arange(2, len(y) - 2)

    return (-0.5 * y[idx - 2] + y[idx - 1] - y[idx + 1] + 0.5 * y[idx + 2]) / h**3


def f(x):
    """
    This is the target function in this example. f(x) is a gaussian curve non-normalized.
    """
    return np.exp(-x**2)


def df_3(x):
    """
    This is the real third derivative of the target gaussian
    """
    return 4 * x * (-2 * x**2 + 3) * np.exp(-x**2)


N = np.arange(20, 51, 5)
hs = 2. / N
devs = np.zeros(hs.shape)


for i, (n, h) in enumerate(zip(N, hs)):
    #Try modifying the next line: if you turn n + 1 into n 
    #the order 
    #turns out to be O(h), we know this is wrong, but it looks like
    #a complex system... how this small change can make such a big 
    #difference? (try it!)
    x = np.linspace(-1, 1, n + 1)
    y = f(x)
    dy = third_der(y, h)
    xx = x[2:-2]
    devs[i] = np.std(dy - df_3(xx))
   

#Let's fit a cuadratic polinomial to our deviation data:

from scipy import linalg

model = np.ones((len(hs), 3))
model[:,1] = hs
model[:, 2] = hs**2

coeffs, res, rank, sing_val = linalg.lstsq(model, devs)

a, b, c = coeffs
x_interp = np.linspace(min(hs), max(hs), 20)
interp = a + b * x_interp + c * x_interp**2


plt.plot(hs, devs, '*', x_interp, interp, 'r')
plt.grid()

plt.title('Error estimation in finite differences derivation')
plt.ylabel('Standard Deviation')
plt.xlabel('Nodes separation')

plt.legend([
        'Error estimate', 
        'Least squares second order polynomial'
    ], 
    loc='upper left'
)

plt.savefig('derivative_error.png')

