import numpy as np

# PARÁMETROS DE LA SIMULACIÓN

N = 121
L = 20

x_m = np.linspace(0, L, N) # grid primario (main)
x_d = (x_m[1:] + x_m[:-1])/2 # En esta construcción es claro que el campo magnético tiene un nodo menos que el eléctrico

dx = x_m[1] - x_m[0] # Definimos el paso espacial

# DEFINIMOS LOS PARÁMETROS DEL PROBLEMA

eps = np.ones(N)
eps[x_m>=L/2] = 3

sigm = np.zeros(N)
sigm[x_m>=L/2] = 0.0 # Al aumentar sigma observamos que la señal se atenua

mu = 1
#c = 1/np.sqrt(mu*eps)

# CONDICIÓN INICAL

x_0 = 5
s_0 = 0.75

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

CFL = 0.9 # condición CFL de estabilidad

dt = CFL * dx / 1

t_range = np.arange(0, 20, dt)

for t in t_range:

    alpha = ( sigm/2 + eps/dt)
    beta = ( sigm/2 - eps/dt)

    E[1:-1] = - beta[1:-1]/alpha[1:-1]*E[1:-1] - (1 / dx / alpha[1:-1])*( H[1:] - H[:-1] )
    H[:]    = H[:]    - (dt / dx / mu) *( E[1:] - E[:-1] )

    plt.plot(x_m, E, 'b.-', label = 'E(x,t)')
    plt.plot(x_d, H, 'r.-', label = 'H(x,t)')
    plt.vlines(L/2, -2, 2, color = 'black')
    plt.xlabel('x (m)')
    plt.ylabel('Campos')
    plt.legend()
    plt.grid()
    plt.ylim(-2, 2)
    plt.xlim(x_m[0], x_m[-1])
    plt.pause(0.01)
    plt.cla()