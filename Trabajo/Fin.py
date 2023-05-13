import numpy as np
import matplotlib.pyplot as plt

# Mallado

N = 101
L = 10

vx, dx = np.linspace(0, L, N, retstep=True)
vy, dy = np.linspace(0, L, N, retstep=True)

vx_d = (vx[1:]+vx[:-1])/2
vy_d = (vy[1:]+vy[:-1])/2

x_ex, y_ex = np.meshgrid(vx_d, vy)
x_ey, y_ey = np.meshgrid(vx, vy_d)
x_h, y_h = np.meshgrid(vx_d, vy_d)

#Parámetros

eps = 1
mu = 1
c = 1/np.sqrt(eps*mu)

x0 = 1*L/10

#Funciones

CFL = 1.0
dt = CFL / ( c * np.sqrt( 1 / dx**2 + 1 / dy**2 ) )
print(dt)
t_range = np.arange(0, 120*dt, dt)


def aux(x, x0, Ny, par, yx):
    sig = np.zeros(x.shape)
    L0 = int(len(x)/2)
    sig[:L0] = np.where(x[:L0]<x0, 9*(x[:L0]-x0)**2, 0)
    sig[L0:] = np.where(x[L0:]>(L-x0), 9*(x[L0:]-(L-x0))**2, 0)

    if yx < 0 :
        sig = sig + np.zeros((Ny,1))
    else:
        sig = (sig + np.zeros((Ny,1))).transpose()
    plt.imshow(sig)
    plt.colorbar()
    plt.show()
    alpha = par/dt - sig/2
    beta = par/dt + sig/2

    return alpha, beta

alpha_xx, beta_xx = aux(vx_d, x0, N, eps, -1) # Sigma_x viviendo donde Ex
alpha_yx, beta_yx = aux(vy, x0, N-1, eps, 1)  # Sigma_y viviendo donde Ex

alpha_xy, beta_xy = aux(vx, x0, N-1, eps, -1) # Sigma_x viviendo donde Ey
alpha_yy, beta_yy = aux(vy_d, x0, N, eps, 1)  # Sigma_y viviendo donde Ey

alpha_xz, beta_xz = aux(vx_d, x0, N-1, mu, -1) # Sigma_x viviendo donde Hz
alpha_yz, beta_yz = aux(vy_d, x0, N-1, eps, 1) # Sigma_y viviendo donde Hz


#Condición incial

x_0 = L/2
s_0 = 0.2

E_x = np.zeros(x_ex.shape)
E_y = np.zeros(x_ey.shape)


D_x = np.zeros(x_ex.shape)
D_y = np.zeros(x_ey.shape)

#H = np.zeros(x_h.shape)
#H[int(1*N/4):int(3*N/4)] = np.exp(-(pow(x_h[int(1*N/4):int(3*N/4)]-x_0, 2))/(2*s_0**2))
H = np.exp(-(pow(x_h-x_0, 2) + pow(y_h-x_0, 2))/(2*s_0**2))
B = mu*H
print(H.shape)
d_xnew = np.zeros(x_ex.shape)
d_ynew = np.zeros(x_ey.shape)

#Condiciones de contorno

E_x[:, 0] = E_x[:, -1] = 0 #pec
E_y[0,:] = E_y[-1,:] = 0 #pec

level = np.linspace(-0.1, 0.6, 51)


#Ley de faraday

def update_h (e_x, e_y, h_z, b_z):

    b_znew = alpha_yz[:,:]/beta_yz[:,:]*b_z[:,:] + 1/beta_yz[:,:]*( -(e_y[:, 1:] - e_y[:, :-1])/dx + (e_x[1:, :] - e_x[:-1, :])/dy )

    h_z[:,:] = alpha_xz[:,:]/beta_xz[:,:]*h_z[:,:] + 1/beta_xz[:,:]*( b_znew[:,:] - b_z[:,:] )/dt
    
    b_z = b_znew

    return h_z, b_z

#Ley de Ampere


def update_e (e_x, e_y, h_z, d_x, d_y) : 

    d_xnew[1:-1,:] = d_x[1:-1,:] + dt/eps*( (h_z[1:,:] - h_z[:-1,:])/dy )
    e_x = alpha_yx[1:-1,:]/beta_yx[1:-1,:] * e_x[1:-1,:] + 1/beta_yx[1:-1,:]*( beta_xx[1:-1,:]*d_xnew[1:-1,:] - alpha_xx[1:-1,:]*d_x[1:-1,:] )

    d_ynew[:, 1:-1] = d_y[:,1:-1] - dt/eps*( (h_z[:,1:] - h_z[:,:-1])/dx )
    e_y = alpha_xy[:, 1:-1]/beta_xy[:, 1:-1] * e_y[:,1:-1] + 1/beta_xy[:,1:-1]*( beta_yy[:,1:-1]*d_ynew[:,1:-1] - alpha_yy[:,1:-1]*d_y[:,1:-1] )

    d_x[:,:] = d_xnew
    d_y[:,:] = d_ynew

    return e_x, e_y, d_x, d_y

#Bucle

c = 1
for t in t_range:
    
    E_x[:, 0] = E_x[:, -1] = 0 #pec
    E_y[0,:] = E_y[-1,:] = 0 #pec
    
    H, B = update_h(E_x, E_y, H, B)
    E_x[1:-1, :], E_y[:, 1:-1], D_x[:, :], D_y[:, :] = update_e(E_x, E_y, H, D_x, D_y)
    
    # plt.xlim(0,L)
    # plt.ylim(0,L)
    # plt.contourf(x_h, y_h, H, levels = level, cmap = 'viridis')
    plt.imshow(H, cmap = 'viridis', interpolation='gaussian',vmin = -0.05, vmax = 0.4)
    # plt.hlines(c*x1, c*x0, c*x1, color = 'black', linestyles='dashed')
    # plt.hlines(c*x0, c*x0, c*x1, color = 'black', linestyles='dashed')
    # plt.vlines(c*x1, c*x0, c*x1, color = 'black', linestyles='dashed')
    # plt.vlines(c*x0, c*x0, c*x1, color = 'black', linestyles='dashed')
    plt.axis('off')
    cb = plt.colorbar()
    plt.pause(0.0001)
    plt.cla()
    cb.remove()