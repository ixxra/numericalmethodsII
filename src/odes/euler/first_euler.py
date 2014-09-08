import numpy as np
from matplotlib import pyplot as plt

x0 = .5
tmax = 2.0
N = 10
h = 2. / N

t = np.linspace(0, tmax, N)
x = np.zeros(t.shape)

x[0] = x0

for i, ti in enumerate(t[:-1]):
    x[i + 1] = x[i] + h * (x[i] - ti**2 + 1)


T, X = np.meshgrid(t, x)
U, V = np.ones(T.shape), X - T**2 + 1
