# coding: utf-8
get_ipython().magic(u'clear ')
import numpy as np
x = np.arange(0, 1, 50)
x
x = np.linspace(0, 1, 50)
y = np.exp(-x)
from matplotlib import pyplot as plt
plt.plot(x, y)
plt.show()
plt.plot(x, y, '*')
plt.show()
x
idx = np.arange(2, 48)
idx
dy = -0.5 * y[idx - 2] + y[idx - 1] - y[idx + 1] + 0.5 * y[idx + 2]
dy
x
dy = dy / 0.02040816**3
plt(x, y,'*',  x[idx], dy, 'o')
plt.plot(x, y,'*',  x[idx], dy, 'o')
plt.show()
plt.plot(x, y,'*',  x[idx], dy, 'o')
plt.legend('funcion', 'derivada')
plt.legend(['funcion', 'derivada'])
plt.show()
get_ipython().magic(u'ls ')
get_ipython().magic(u'save ejemplo_salon_clase.py 1-27')
