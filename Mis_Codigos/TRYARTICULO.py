import numpy as np

# PARÁMETROS DE LA SIMULACIÓN

N = 150
L = 10

x_m = np.linspace(0, L, N) # grid primario (main)
x_d = (x_m[1:] + x_m[:-1])/2 # En esta construcción es claro que el campo magnético tiene un nodo menos que el eléctrico
dx = x_m[1]-x_m[0]

# DEFINIMOS LOS PARÁMETROS DEL PROBLEMA

sigm = np.zeros(N)
sigm = np.where((x_m>=4*L/5) | (x_m<=1*L/5), 0.5, 0) # Al aumentar sigma observamos que la señal se atenua

sigmH = np.zeros(len(x_d))
sigmH = np.where((x_d>=4*L/5) | (x_d<=1*L/5), 0.5, 0) # Al aumentar sigma* observamos que la señal se atenua
mu = 1
eps = 1
c = 1/np.sqrt(mu*eps)

# CONDICIÓN INICAL

x_0 = L/2
s_0 = 0.25

e = np.exp(-pow(x_m-x_0,2)/(2*s_0**2))
h = np.zeros(x_d.shape)
b = np.zeros(x_d.shape)

e[0] = 0
e[-1] = 0

e_new = np.zeros(e.shape)
h_new = np.zeros(h.shape)
b_new = np.zeros(b.shape)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(x_m, e, 'b.-')
plt.plot(x_d, h, 'r.-')
plt.title('Condición inicial de $E(t,x)$ y $H(t,x)$')
plt.xlabel('x (m)')
plt.ylabel('$E(t,x)$ $H(t,x)$')
plt.grid()


# SIMULACIÓN

CFL = 1.0 # condición CFL de estabilidad

dt = CFL * dx / 1

t_range = np.arange(0, 10, dt)

for t in t_range:

    alpha = ( eps/dt + sigm/2 )
    beta =  ( eps/dt - sigm/2 )
    alphaH = ( mu/dt + sigmH/2 )
    betaH =  ( mu/dt - sigmH/2 )

    b_new   = b[:] + (dt/dx)*(e[1:] - e[:-1])
    h[:]    = h[:] + (dt/(2*eps**2))*(b_new[:]*alphaH - b[:]*betaH)
    e[1:-1] = beta[1:-1]/alpha[1:-1]*e[1:-1] + (1 / dx / alpha[1:-1])*( h[1:] - h[:-1] )
    b = b_new

    e[0] =  0
    e[-1] =  0

    plt.plot(x_m, e, linestyle = '-', marker = 's', markersize = 2, color = 'dodgerblue', label = 'E(x,t)')
    plt.plot(x_d, h, linestyle = '-', marker = 's', markersize = 2, color = 'orangered', label = 'H(x,t)')
    plt.vlines(4*L/5, -2, 2, color = 'black')
    plt.vlines(1*L/5, -2, 2, color = 'black')
    plt.xlabel('x (m)')
    plt.ylabel('Campos')
    plt.legend()
    plt.grid()
    plt.ylim(-2, 2)
    plt.xlim(x_m[0], x_m[-1])
    plt.pause(0.01)
    plt.cla()

print(np.sqrt((eps+sigm[1:])/(mu+sigmH)))