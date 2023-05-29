import numpy as np
import matplotlib.pyplot as plt

eps = 1.0
mu = 1.0
c = 1/np.sqrt(eps*mu)

class FDTD_PML1D():
    def __init__(self, L=10, N=121):
        self.N = N
        self.L = L
        self.x, self.dx = np.linspace(0, L, N, retstep=True)       # Definimos el vector del espacio principal en x

        self.CFL = 1
        self.x_d = (self.x[1:] + self.x[:-1])/2             # Definimos el vector del espacio dual en x
        self.dt = self.CFL / ( c * 1/self.dx )              # Definimos el paso temporal

        # Al hacer la reducción de las ecuaciones de maxwell en 1D trabajamos con los campos (Ex, Hy)
        
        self.e = np.zeros( N )
        self.h = np.zeros( N-1 )

        self.w = 1*L/20 # anchura de la PML
        
        def aux(x, par):
            sig = np.zeros(x.shape)
            L0 = int(len(x)/2)
            sig[:L0] = np.where(x[:L0] < self.w, 90*(x[:L0]-self.w)**2, 0)
            sig[L0:] = np.where(x[L0:] > (L-self.w), 90*(x[L0:]-(L-self.w))**2, 0)

            sig += np.exp(-pow(x-2*L/3,2)/(2*0.2**2))
            alpha = par/self.dt + sig/2
            beta = par/self.dt - sig/2

            return alpha, beta, sig

        self.alpha_x, self.beta_x, sig0 = aux(self.x, eps)       # Sigma_z viviendo donde Ex
        self.sigma = sig0
        self.alpha_z, self.beta_z, sig1 = aux(self.x_d, eps)     # Sigma_z viviendo donde Hy

    def update(self):

        #alpha_x, beta_x = self.alpha_x, self.beta_x
        #alpha_z, beta_z = self.alpha_z, self.beta_z

        e, h = self.e, self.h

        # Imponemos las condiciones de contorno del problema, en el artículo se aplican condiciones PEC

        e[0] = e[-1] = 0 #pec

        # Actualizamos de manera normal e

        e[1:-1] =  self.beta_x[1:-1]/self.alpha_x[1:-1]*e[1:-1] - (1 / self.dx / self.alpha_x[1:-1])*( h[1:] - h[:-1] )

        # Actualizamos de manera noraml h
        
        h[:]    =  self.beta_z[:]/self.alpha_z[:]*h[:] - (1 / self.dx / self.alpha_z[:])*( e[1:] - e[:-1] )