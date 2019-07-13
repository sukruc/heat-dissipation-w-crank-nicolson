import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import Ridge

MAT_FILENAME = 'test.pk'

X = 1
Y = .2
k = 1.3
ro = 2400
cp = 700
alfa = k / (ro * cp)

nx = 41
ny = 11

dx = X / (nx - 1)
dy = Y / (ny - 1)

dt = 1
tson = 3600
tilk = 0
time_step = tson / dt + 1

tx = alfa * dt / dx**2
ty = alfa * dt / dy**2
h = 1000
Ts = 1223

A1 = np.zeros((nx*ny, nx*ny))
A2 = np.zeros((nx*ny, nx*ny))
G = np.zeros((nx*ny, 1))
T = 20 * np.ones((nx*ny, 1, int(time_step)))

for i in range(0, nx-1):
    G[i,0] = tx * h * dx * Ts / k

for i in [1]:
    for j in [1]:
        A1 [(i-1)*nx+j-1,(i-1)*nx+j-1] = 1+tx+ty+tx*h*dx/2/k;
        A1 [(i-1)*nx+j-1,(i-1)*nx+j-1+1] = -tx;
        A1 [(i-1)*nx+j-1,i*nx+j-1] = -ty;

        A2 [(i-1)*nx+j-1,(i-1)*nx+j-1] = 1-tx-ty-tx*h*dx/2/k;
        A2 [(i-1)*nx+j-1,(i-1)*nx+j-1+1] = tx;
        A2 [(i-1)*nx+j-1,i*nx+j-1] = ty;
    for j in range(2,nx):
        A1 [(i-1)*nx+j-1,(i-1)*nx+j-1] = 1+tx+ty+tx*h*dx/2/k;
        A1 [(i-1)*nx+j-1,(i-1)*nx+j-1-1] = -tx/2;
        A1 [(i-1)*nx+j-1,(i-1)*nx+j-1+1] = -tx/2;
        A1 [(i-1)*nx+j-1,i*nx+j-1] = -ty;

        A2 [(i-1)*nx+j-1,(i-1)*nx+j-1] = 1-tx-ty-tx*h*dx/2/k;
        A2 [(i-1)*nx+j-1,(i-1)*nx+j-1-1] = tx/2;
        A2 [(i-1)*nx+j-1,(i-1)*nx+j-1+1] = tx/2;
        A2 [(i-1)*nx+j-1,i*nx+j-1] = ty;
    for j in [nx]:
        A1 [(i-1)*nx+j-1,(i-1)*nx+j-1] = tx+ty+tx*h*dx/2/k+1;
        A1 [(i-1)*nx+j-1,(i-1)*nx+j-1-1] = -tx;
        A1 [(i-1)*nx+j-1,i*nx+j-1] = -ty;

        A2 [(i-1)*nx+j-1,(i-1)*nx+j-1] = 1-tx-ty-tx*h*dx/2/k;
        A2 [(i-1)*nx+j-1,(i-1)*nx+j-1-1] = tx;
        A2 [(i-1)*nx+j-1,i*nx+j-1] = ty;

for i in range(2,ny):
    for j in [1]:
        A1 [(i-1)*nx+j-1,(i-1)*nx+j-1] = tx+ty+1;
        A1 [(i-1)*nx+j-1,(i-1)*nx+j-1+1] = -tx;
        A1 [(i-1)*nx+j-1,(i-2)*nx+j-1] = -ty/2;
        A1 [(i-1)*nx+j-1,i*nx+j-1] = -ty/2;

        A2 [(i-1)*nx+j-1,(i-1)*nx+j-1] = 1-tx-ty;
        A2 [(i-1)*nx+j-1,(i-1)*nx+j-1+1] = tx;
        A2 [(i-1)*nx+j-1,(i-2)*nx+j-1] = ty/2;
        A2 [(i-1)*nx+j-1,i*nx+j-1] = ty/2;
    for j in range(2,nx):
        A1 [(i-1)*nx+j-1,(i-1)*nx+j-1] = tx+ty+1;
        A1 [(i-1)*nx+j-1,(i-1)*nx+j-1-1] = -tx/2;
        A1 [(i-1)*nx+j-1,(i-1)*nx+j-1+1] = -tx/2;
        A1 [(i-1)*nx+j-1,(i-2)*nx+j-1] = -ty/2;
        A1 [(i-1)*nx+j-1,i*nx+j-1] = -ty/2;

        A2 [(i-1)*nx+j-1,(i-1)*nx+j-1] = 1-tx-ty;
        A2 [(i-1)*nx+j-1,(i-1)*nx+j-1-1] = tx/2;
        A2 [(i-1)*nx+j-1,(i-1)*nx+j-1+1] = tx/2;
        A2 [(i-1)*nx+j-1,(i-2)*nx+j-1] = ty/2;
        A2 [(i-1)*nx+j-1,i*nx+j-1] = ty/2;
    for j in [nx]:
        A1 [(i-1)*nx+j-1,(i-1)*nx+j-1] = tx+ty+1;
        A1 [(i-1)*nx+j-1,(i-1)*nx+j-1-1] = -tx;
        A1 [(i-1)*nx+j-1,(i-2)*nx+j-1] = -ty/2;
        A1 [(i-1)*nx+j-1,i*nx+j-1] = -ty/2;

        A2 [(i-1)*nx+j-1,(i-1)*nx+j-1] = 1-tx-ty;
        A2 [(i-1)*nx+j-1,(i-1)*nx+j-1-1] = tx;
        A2 [(i-1)*nx+j-1,(i-2)*nx+j-1] = ty/2;
        A2 [(i-1)*nx+j-1,i*nx+j-1] = ty/2;

for i in [ny]:
    for j in [1]:
        A1 [(i-1)*nx+j-1,(i-1)*nx+j-1] = tx+ty+1;
        A1 [(i-1)*nx+j-1,(i-1)*nx+j-1+1] = -tx;
        A1 [(i-1)*nx+j-1,(i-2)*nx+j-1] = -ty;

        A2 [(i-1)*nx+j-1,(i-1)*nx+j-1] = 1-tx-ty;
        A2 [(i-1)*nx+j-1,(i-1)*nx+j-1+1] = tx;
        A2 [(i-1)*nx+j-1,(i-2)*nx+j-1] = ty;

    for j in range(2,nx):
        A1 [(i-1)*nx+j-1,(i-1)*nx+j-1] = tx+ty+1;
        A1 [(i-1)*nx+j-1,(i-1)*nx+j-1-1] = -tx/2;
        A1 [(i-1)*nx+j-1,(i-1)*nx+j-1+1] = -tx/2;
        A1 [(i-1)*nx+j-1,(i-2)*nx+j-1] = -ty;

        A2 [(i-1)*nx+j-1,(i-1)*nx+j-1] = 1-tx-ty;
        A2 [(i-1)*nx+j-1,(i-1)*nx+j-1-1] = tx/2;
        A2 [(i-1)*nx+j-1,(i-1)*nx+j-1+1] = tx/2;
        A2 [(i-1)*nx+j-1,(i-2)*nx+j-1] = ty;

    for j in [nx]:
        A1 [(i-1)*nx+j-1,(i-1)*nx+j-1] = tx+ty+1;
        A1 [(i-1)*nx+j-1,(i-1)*nx+j-1-1] = -tx;
        A1 [(i-1)*nx+j-1,(i-2)*nx+j-1] = -ty;

        A2 [(i-1)*nx+j-1,(i-1)*nx+j-1] = 1-tx-ty;
        A2 [(i-1)*nx+j-1,(i-1)*nx+j-1-1] = tx;
        A2 [(i-1)*nx+j-1,(i-2)*nx+j-1] = ty;
for e in range(int(time_step)-1):
    print(e)
    # T[:,:,e+1] = np.linalg.solve(A1, (A2@T[:,:,e]+G))
    r = Ridge(alpha=0, fit_intercept=False, solver="sparse_cg", tol=1e-3)
    r.fit(A1, (A2@T[:,:,e]) + G)
    T[:,:,e+1] = r.coef_.T


metadata = {'data':T,
            'nx':nx,
            'ny':ny,
            'X':X,
            'Y':Y,
            'dt':dt}
pickle.dump(metadata, open(MAT_FILENAME,'wb'))
# #
# C = reshape (T[:,:,time_step],nx,ny);
# D = C';
