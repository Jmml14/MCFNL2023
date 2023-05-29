import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch

import PML2D
import OBSTACLE2D

fd = PML2D.FDTD_PML2D()

eps = 1.0                   # Permitividad eléctrica
mu = 1.0                    # Permeabilidad magnética
c = 1/np.sqrt(eps*mu)       # Velocidad de la onda


x_ex, y_ex = np.meshgrid( fd.vx_d, fd.vy )    # Espacio en el que vive el campo Ex
x_ey, y_ey = np.meshgrid( fd.vx, fd.vy_d )    # Espacio en el que vive el campo Ey
x_h, y_h   = np.meshgrid( fd.vx_d, fd.vy_d )  # Espacio en el que vive el campo Hz

x_0, y_0, s_0 = 1*fd.L/2, 1*fd.L/2, 0.2       # Parámetros de la gaussiana inicial


fd.h_z = np.exp(-(pow(x_h-x_0, 2) + pow(y_h-x_0, 2))/(2*s_0**2))

fd.b_z = mu*(fd.h_z)
fd.e_x = np.zeros(x_ex.shape)
fd.e_y = np.zeros(x_ey.shape)
fd.d_x = np.zeros(x_ex.shape)
fd.d_y = np.zeros(x_ey.shape)

counter = 0
for _ in np.arange(0, 160*fd.dt, fd.dt):

    fd.h_z, fd.b_z = fd.update_h()
    fd.e_x[1:-1, :], fd.e_y[:, 1:-1], fd.d_x[:, :], fd.d_y[:, :] = fd.update_e()

    # PULSOS GAUSSIANOS
    # counter += 1
    # if counter%20 == 0:
    #     fd.h_z[:, :] += np.exp(-(pow(x_h[:, :]-x_0, 2) + pow(y_h[:, :]-x_0, 2))/(2*s_0**2))
    #     fd.b_z += mu*(fd.h_z)
    
    ax = plt.figure(1, figsize=(6,6))
    plt.imshow(fd.h_z, cmap = 'viridis', interpolation = 'bilinear', vmin = -0.0, vmax = 0.4)

    # PINTAR PELOTA
    # ax = plt.gca()
    # circle = patch.Circle( (3*fd.N/4, 1*fd.N/2), radius = 20, fill = False, ls = '--', color = 'black')
    # ax.add_patch(circle)

    plt.hlines(fd.N*9/10, fd.N*1/10, fd.N*9/10, color = 'black', linestyles='dashed')
    plt.hlines(fd.N*1/10, fd.N*1/10, fd.N*9/10, color = 'black', linestyles='dashed')
    plt.vlines(fd.N*9/10, fd.N*1/10, fd.N*9/10, color = 'black', linestyles='dashed')
    plt.vlines(fd.N*1/10, fd.N*1/10, fd.N*9/10, color = 'black', linestyles='dashed')

    plt.axis('off')
    plt.pause(0.00001)
    plt.cla()