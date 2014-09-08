import numpy as np
from matplotlib import pyplot as plt


N = np.arange(10, 41, 5)
h = 2. / N

errors = []

exact_value = 1 - np.exp(-2)

for n in N:
    x = np.linspace(0, 2, n + 1)
    y = np.exp(-x)

    idx = np.arange(1, len(x) - 1)

    h = 2. / n
    y_int = sum(0.5 * h *(y[idx + 1] + y[idx - 1]))
    errors.append(abs(exact_value - y_int))


h = 2. / N

plt.plot(h, errors, '.r')
plt.grid()
plt.title('Trapezoidal rule errors')
plt.ylabel('Absolute error')
plt.xlabel('Nodes separation')
plt.show()

#print 'Trapezoidal rule: ', y_int
#print 'Exact value: ', 1 - np.exp(-2)
