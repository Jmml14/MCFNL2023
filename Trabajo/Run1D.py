import numpy as np
import matplotlib.pyplot as plt
import PML1D

fd = PML1D.FDTD_PML1D()

eps = 1.0
mu = 1.0
c = 1/np.sqrt(eps*mu)

# CONDICIÓN INICAL

x_0 = 2*fd.L/3
s_0 = 0.25

e_ini = np.exp(-pow(fd.x-x_0,2)/(2*s_0**2))

fd.e[:] = e_ini[:]

for _ in np.arange(0, 80*fd.dt, fd.dt):

    fd.update()

    plt.plot(fd.x, fd.e, linestyle = '-', marker = 's', markersize = 2, color = 'dodgerblue', label = 'E(x,t)')
    plt.plot(fd.x_d, fd.h, linestyle = '-', marker = 's', markersize = 2, color = 'orangered', label = 'H(x,t)')
    plt.vlines(fd.w, -2, 2, color = 'black')
    plt.vlines((fd.L-fd.w), -2, 2, color = 'black')
    plt.xlabel('x (m)')
    plt.ylabel('Campos')
    plt.legend()
    plt.grid()
    plt.ylim(-2, 2)
    plt.xlim(fd.x[0], fd.x[-1])
    plt.pause(0.01)
    plt.cla()