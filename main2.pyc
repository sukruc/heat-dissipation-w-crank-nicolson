import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

from heatutils import initialize, calculate, propagate

MAT_FILENAME = 'T_step02s_2.pk'

X = 1
Y = .2
k = 1.3
ro = 2400
cp = 700
h = 1000
T_init = 20
step_size = .1
time_step = .001
t_end = 3600

A1, A2, G, T, nx, ny, tx, ty, dx= initialize(X, Y, T_init, k, h, ro, cp, step_size,
                       time_step, t_end)

A1, A2 = calculate(A1, A2,
                   int(X / step_size),
                   int(Y / step_size),
                   tx, ty, h, dx, k)

T = propagate(A1, A2, T, G, time_step)
