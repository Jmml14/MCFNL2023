import numpy as np
import matplotlib.pyplot as plt

import PML2D

fd = PML2D.FDTD_PML2D()

eps = 1.0
mu = 1.0
c = 1/np.sqrt(eps*mu)

x_ex, y_ex = np.meshgrid( fd.vx_d, fd.vy )
x_ey, y_ey = np.meshgrid( fd.vx, fd.vy_d )
x_h, y_h = np.meshgrid( fd.vx_d, fd.vy_d )

x_0, s_0 = fd.L/2, 0.2 # Parámetros de la gaussiana

fd.h_z[:, :] = np.exp(-(pow(x_h[:, :]-x_0, 2) + pow(y_h[:, :]-x_0, 2))/(2*s_0**2))

fd.b_z = mu*(fd.h_z)
fd.e_x = np.zeros(x_ex.shape)
fd.e_y = np.zeros(x_ey.shape)
fd.d_x = np.zeros(x_ex.shape)
fd.d_y = np.zeros(x_ey.shape)

for _ in np.arange(0, 120*fd.dt, fd.dt):

    fd.h_z, fd.b_z = fd.update_h()
    fd.e_x[1:-1, :], fd.e_y[:, 1:-1], fd.d_x[:, :], fd.d_y[:, :] = fd.update_e()

    plt.imshow(fd.h_z, cmap = 'viridis', interpolation='gaussian', vmin = -0.05, vmax = 0.4)
    plt.axis('off')
    cb = plt.colorbar()
    plt.pause(0.00001)
    plt.cla()
    cb.remove()