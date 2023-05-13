import numpy as np

# DEFINIMOS LOS PARÁMETROS DEL PROBLEMA

eps = 1 
mu = 1
c = 1/np.sqrt(mu*eps)

# PARÁMETROS DE LA SIMULACIÓN Y CONDICIÓN INICAL

N = 101

x_m = np.linspace(0, 10, N) # grid primario (main)
x_d = (x_m[1:] + x_m[:-1])/2 # En esta construcción es claro que el campo magnético tiene un nodo menos que el eléctrico

dx = x_m[1] - x_m[0] # Definimos el paso espacial

x_0 = 4
s_0 = 0.75

E = np.exp(-pow(x_m-x_0,2)/(2*s_0**2))
H = np.zeros(x_d.shape) 
#H = np.exp(-pow(x_d-x_0,2)/(2*s_0**2))


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

dt = CFL * dx / c

t_range = np.arange(0, 10, dt)

for t in t_range:

    E[1:-1] = E[1:-1] - (dt / dx / eps)*( H[1:] - H[:-1] )

    # Lado izquierdo -> PEC
    # E[0] = 0

    #E[0] = E[0] - 2*dt/dx/eps * H[0] # PMC fija el campo magnético
    #E[-1] = 0                        # PEC fija el campo eléctrico

    #E[0] = E[0] - (dt / dx / eps) * ( H[0] - H[-1] ) # Condiciones peródicas
    #E[-1] = E[0]
 
    H[:]    = H[:] - (dt / dx / mu) * ( E[1:] - E[:-1] )

    plt.plot(x_m, E, 'b.-', label = 'E(x,t)')
    plt.plot(x_d, H, 'r.-', label = 'H(x,t)')
    plt.title('$E(t,x)$ y $H(t,x)$')
    plt.xlabel('x (m)')
    plt.ylabel('Campos')
    plt.legend()
    plt.grid()
    plt.ylim(-1.1, 1.1)
    plt.xlim(x_m[0], x_m[-1])
    plt.pause(0.01)
    plt.cla()