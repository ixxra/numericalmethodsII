import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 2,10)
y = np.zeros(t.shape)
w = np.zeros(t.shape)

y = t**2 + 2 * t + 1 - 0.5 * np.exp(t)

w[0] = 0.5
h = 2/10.0

for i, ti in enumerate(t[:-1]):
    w[i + 1] = w[i] + h * (w[i] - ti**2 + 1)


#Plano Fase

T, Y = np.meshgrid(t, y)
U, V = np.ones(T.shape), Y - T**2 + 1


