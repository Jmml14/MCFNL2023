import numpy as np

# PARÁMETROS DE LA SIMULACIÓN

N = 101
L = 10

x_m = np.linspace(0, L, N) # grid primario (main)
x_d = (x_m[1:] + x_m[:-1])/2 # En esta construcción es claro que el campo magnético tiene un nodo menos que el eléctrico
dx = x_m[1]-x_m[0]

# DEFINIMOS LOS PARÁMETROS DEL PROBLEMA

def sigmf(x,x0):
    return 9.0*(x-x0)**2

L0 = 1*L/10
L1 = 9*L/10

sigm = np.zeros(len(x_m))
sigm[0:int(N/2)] = np.where(x_m[0:int(N/2)] <= L0, sigmf(x_m[0:int(N/2)], L0), 0) # Al aumentar sigma* observamos que la señal se atenua
sigm[int(N/2):int(N)] = np.where(x_m[int(N/2):int(N)] <= L1, 0, sigmf(x_m[int(N/2):int(N)], L1))

sigmH = np.zeros(len(x_d))
sigmH[0:int((N-1)/2)] = np.where(x_d[0:int((N-1)/2)] <= L0, sigmf(x_d[0:int((N-1)/2)], L0), 0) # Al aumentar sigma* observamos que la señal se atenua
sigmH[int((N-1)/2):int(N-1)] = np.where(x_d[int((N-1)/2):int(N)] <= L1, 0, sigmf(x_d[int((N-1)/2):int(N)], L1))

mu = 1
eps = 1
c = 1/np.sqrt(mu*eps)

# CONDICIÓN INICAL

x_0 = 2*L/3
s_0 = 0.25

E = np.exp(-pow(x_m-x_0,2)/(2*s_0**2))
H = np.zeros(x_d.shape) 

E[0] = 0
E[-1] = 0

E_new = np.zeros(E.shape)
H_new = np.zeros(H.shape)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(x_m, E, 'b.-')
plt.plot(x_d, H, 'r.-')
plt.title('Condición inicial de $E(t,x)$ y $H(t,x)$')
plt.xlabel('x (m)')
plt.ylabel('$E(t,x)$ $H(t,x)$')
plt.grid()


# SIMULACIÓN

CFL = 1.0 # condición CFL de estabilidad

dt = CFL * dx / 1

t_range = np.arange(0, 100*dt, dt)

for t in t_range:

    alpha = ( eps/dt + sigm/2 )
    beta =  ( eps/dt - sigm/2 )
    alphaH = ( mu/dt + sigmH/2 )
    betaH =  ( mu/dt - sigmH/2 )

    E[1:-1] =  beta[1:-1]/alpha[1:-1]*E[1:-1] - (1 / dx / alpha[1:-1])*( H[1:] - H[:-1] )
    H[:]    =  betaH[:]/alphaH[:]*H[:]        - (1 / dx / alphaH[:])*( E[1:] - E[:-1] )

    E[0] =  0
    E[-1] =  0

    plt.plot(x_m, E, linestyle = '-', marker = 's', markersize = 2, color = 'dodgerblue', label = 'E(x,t)')
    plt.plot(x_d, H, linestyle = '-', marker = 's', markersize = 2, color = 'orangered', label = 'H(x,t)')
    plt.vlines(L0, -2, 2, color = 'black')
    plt.vlines(L1, -2, 2, color = 'black')
    plt.xlabel('x (m)')
    plt.ylabel('Campos')
    plt.legend()
    plt.grid()
    plt.ylim(-2, 2)
    plt.xlim(x_m[0], x_m[-1])
    plt.pause(0.01)
    plt.cla()