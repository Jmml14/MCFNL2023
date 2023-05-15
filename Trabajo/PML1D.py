import numpy as np
import matplotlib.pyplot as plt

eps = 1.0
mu = 1.0
c = 1/np.sqrt(eps*mu)

class FDTD_PML1D():
    def __init__(self, L=10, CFL=1.0, N=101):

        self.L = L
        self.x, self.dx = np.linspace(0, L, N, retstep=True)       # Definimos el vector del espacio principal en x

        self.x_d = (self.x[1:] + self.x[:-1])/2             # Definimos el vector del espacio dual en x
        self.dt = CFL / ( c * 1/self.dx )                   # Definimos el paso temporal

        # Al hacer la reducción de las ecuaciones de maxwell en 1D trabajamos con los campos (Ex, Hy)
        
        self.e = np.zeros( N )
        self.h = np.zeros( N-1 )

        self.w = 1*L/20 # anchura de la PML
        
        def aux(x, par):
            sig = np.zeros(x.shape)
            L0 = int(len(x)/2)
            sig[:L0] = np.where(x[:L0] < self.w, 70*(x[:L0]-self.w)**2, 0)
            sig[L0:] = np.where(x[L0:] > (L-self.w), 70*(x[L0:]-(L-self.w))**2, 0)

            alpha = par/self.dt + sig/2
            beta = par/self.dt - sig/2

            return alpha, beta

        self.alpha_x, self.beta_x = aux(self.x, eps)       # Sigma_z viviendo donde Ex
        self.alpha_z, self.beta_z = aux(self.x_d, eps)     # Sigma_z viviendo donde Hy

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