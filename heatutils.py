import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import Ridge
from datetime import datetime

# MAT_FILENAME = 'T_step02s_1.pk'

# X = 1
# 5
# Y = .2
# k = 1.3
# ro = 2400
# cp = 700
# T_init = 293
# h = 1000
# Ts = 1500

def initialize(X, Y, T_init, k, h, ro, cp, step_size, timestep, t_end, Ts):
    alfa = k / (ro * cp)
    nx = int(X / step_size)
    ny = int(Y / step_size)
    dx = X / (nx - 1)
    dy = Y / (ny - 1)
    dt = timestep
    tilk = 0
    time_step = int(t_end / dt + 1)
    tx = alfa * dt / dx**2
    ty = alfa * dt / dy**2
    tic = datetime.now()
    A1 = np.zeros((nx*ny, nx*ny))
    A2 = np.zeros((nx*ny, nx*ny))
    G = np.zeros((nx*ny, 1))
    T = T_init * np.ones((nx*ny, 1, int(time_step)))
    toc = datetime.now()
    print('Heat dissipation matrix compiled.\n')
    print('Compilatiion time: ', toc - tic)
    for i in range(0, nx-1):
        G[i,0] = tx * h * dx * Ts / k
    return A1, A2, G, T, nx, ny, tx, ty, dx


def calculate(A1, A2, nx, ny, tx, ty, h, dx, k):
    print('Creating Crank-Nicolson scheme...')
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
    print('Done.')
    return A1, A2

def propagate(A1, A2, T, G, time_step, method='cg'):
    time_elapsed = 0
    print('Gas flow commenced.')
    try:
        for e in range(1,T.shape[2]-1):
            time_elapsed += time_step
            #   Çözümü hızlandırmak için Lineer Çözüm yerine Konjuge Gradyan Metodu kullanıldı.
            #   Hata Oranı 1e-5
            if method.lower() == 'cg':
                r = Ridge(alpha=0, fit_intercept=False, tol=0.00001,
                solver='sparse_cg')
                r.fit(A1, (A2@T[:,:,e]+G))
                T[:,:,e+1] = r.coef_.reshape(-1,1)
            elif method == 'OLS':
                T[:,:,e+1] = np.linalg.solve(A1, (A2@T[:,:,e]+G))
            else:
                raise TypeError("Invalid solver. Pass method as either 'cg' or 'OLS'.")


            if time_elapsed % 600 <= 0.01:
                print('Time passed in sim: %.2f seconds' % time_elapsed)
    except KeyboardInterrupt:
        print('Calculation interrupted.\n')
        print('Last calculated time: %.2f' % time_elapsed)
        return T
    return T
# propagate(......time_step)

def heatmap(X, Y, T, dx, dt, time_elapsed, ax=None):
    '''Creates heatmap at given time.
    Parameters:
    -------------
    X : Length
    Y : Height
    T : Heat dissipation tensor
    dx : Delta x, step size for length/height
    dt : Delta t, timestep
    time_elapsed : Time to draw heat state at (seconds)
    ax : Axis to pass viz.
    '''
    cmap_intervals = np.arange(0, np.max(T), 10)
    mx = np.linspace(0, X, X/dx)
    my = np.linspace(0, Y, Y/dx)
    vx, vy = np.meshgrid(mx,my)
    if ax is None:
        fig = plt.figure(figsize=(12,8))
        plt.contourf(vx, vy, np.flipud(T[:,:,int(time_elapsed/dt)].reshape(len(my),len(mx))),
                    cmap_intervals,
                    antialiased=False,
                    cmap='RdYlGn')
        plt.colorbar()
        plt.title('Heat dissipation at the end of %.2f s' % time_elapsed)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        return fig
    else:
        ax.contourf(vx, vy,
                    np.flipud(T[:,:,int(time_elapsed/dt)]\
                        .reshape(len(my),len(mx))),
                    cmap_intervals,
                    antialiased=False,
                    cmap='jet')
        # plt.colorbar()
        ax.set_title('Heat dissipation at the end of %.2f s' % time_elapsed)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        return ax
