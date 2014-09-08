from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


def dx(x, t):
    return 1 + x**2 + t**3


def dx_2(x, dx, t):
    return 2 * x * dx + 3 * t**2


def dx_3(x, dx, dx_2, t):
    return 2 * x * dx_2 + 2 * dx**2 + 6 * t


def dx_4(x, dx, dx_2, dx_3, t):
    return 2 * x * dx_3 + 6 * dx * dx_2 + 6


N = 20
t = np.linspace(1, 2, N)
h = (2 - 1) / (N - 1)

w = np.zeros(t.shape)
dw = np.zeros(t.shape)
dw_2 = np.zeros(t.shape)
dw_3 = np.zeros(t.shape)
dw_4 = np.zeros(t.shape)


def taylor(t0, w0):
    dw0 = dx(w0, t0)
    dw0_2 = dx_2(w0, dw0, t0)
    dw0_3 = dx_3(w0, dw0, dw0_2, t0)
    dw0_4 = dx_4(w0, dw0, dw0_2, dw0_3, t0)
    return dw0, dw0_2, dw0_3, dw0_4


t0 = t[0]
w[0] = -4

dw[0], dw_2[0], dw_3[0], dw_4[0] = taylor(t0, w[0])
euler = np.zeros(t.shape)
euler[0] = -4


for i, ti in enumerate(t[:-1]):
    wi = w[i]
    dwi = dw[i]
    dwi_2 = dw_2[i]
    dwi_3 = dw_3[i]
    dwi_4 = dw_4[i]

    w[i + 1] = wi + dwi * h + dwi_2 * h**2 / 2 + dwi_3 * h**3 / 6 + dwi_4 * h**4 / 24
    dw[i + 1], dw_2[i + 1], dw_3[i + 1], dw_4[i + 1] = taylor(t[i + 1], w[i + 1])

    euler[i + 1] = euler[i] + dx(euler[i], ti) * h


x = np.linspace(w.min(), w.max(), 50)
T, X =  np.meshgrid(t, x)
U, V = np.ones(T.shape), 1 + X**2 + T**3

