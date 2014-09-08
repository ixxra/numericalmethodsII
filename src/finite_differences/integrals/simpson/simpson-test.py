import numpy as np
from matplotlib import pyplot as plt


N = np.arange(10, 401, 5)
h = 2. / N

errors = []
exact_value = 1 - np.exp(-2)

for n in N:
    x = np.linspace(0, 2, 2 * n + 1)
    y = np.exp(-x)

    idx = np.arange(1, len(x) - 1, 2)

    h = 2. / (2 * n)
    y_int = sum(h *(y[idx - 1] + 4 * y[idx] + y[idx - 2]) / 3.)
    errors.append(abs(exact_value - y_int))


h = 2. / (2 * N)

plt.plot(h, errors, '.r')
plt.grid()
plt.title('Simpson rule errors')
plt.ylabel('Absolute error')
plt.xlabel('Nodes separation')
plt.show()

