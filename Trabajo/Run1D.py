import numpy as np
import matplotlib.pyplot as plt

import PML1D
import OBSTACLE1D

fd = PML1D.FDTD_PML1D()

eps = 1.0                   # Permitividad eléctrica
mu = 1.0                    # Permeabilidad magnética
c = 1/np.sqrt(eps*mu)       # Velocidad de la onda

# CONDICIÓN INICAL gaussiana

x_0 = 1*fd.L/3  # Centro de la gaussiana
s_0 = 0.25      # anchura de la gaussiana

e_ini = np.exp(-pow(fd.x-x_0,2)/(2*s_0**2))

fd.e[:] = e_ini[:]

for t in np.arange(0, 100*fd.dt, fd.dt):

    fd.update()

    #plt.plot(fd.x[int(1*fd.N/3):int(3*fd.N/4)], fd.sigma[int(1*fd.N/3):int(3*fd.N/4)], linestyle = '--', color = 'black', label = 'Obstaculo')

    plt.plot(fd.x, fd.e, linestyle = '-', marker = 's', markersize = 2, color = 'dodgerblue', label = 'E(x,t)')
    plt.plot(fd.x_d, fd.h, linestyle = '-', marker = 's', markersize = 2, color = 'orangered', label = 'H(x,t)')
    plt.vlines(fd.w, -2, 2, color = 'black', linestyles='--')
    plt.vlines((fd.L-fd.w), -2, 2, color = 'black', linestyles='--')
    plt.xlabel('x (m)')
    plt.ylabel('Campos')
    plt.legend()
    plt.grid()
    plt.ylim(-2, 2)
    plt.xlim(fd.x[0], fd.x[-1])
    plt.pause(0.01)
    plt.cla()