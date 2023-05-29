import numpy as np
import matplotlib.pyplot as plt

eps = 1.0                   # Permitividad eléctrica
mu = 1.0                    # Permeabilidad magnética
c = 1/np.sqrt(eps*mu)       # Velocidad de la onda

porcentaje = 1/10           # Pocentaje que carcateriza el ancho de la PML ( El ancho será porcentaje*L )

class FDTD_PML1D():
    """

    Clase FDTD_PML1D: toma como argumentos:
    - L:   Espacio de la simulación
    - w:   Porcentaje del ancho de la PML
    - CFL: Valor de la condición CFL de la simulación
    - N:   Partición espacial de la simulación

    Esta clase construye los vectores de la conductividad sigma_x. Por otro lado, contruye 
    un método para actualizar los campos eléctrico y magnético. En este caso no es necesario usar el doble paso ya que los campos con los 
    que trabajamos son paralelos a la PML y estos se actualizan normal.

    """
    def __init__(self, L=10, w = porcentaje, CFL=1.0, N=121):

        self.L = L

        self.x, self.dx = np.linspace(0, L, N, retstep=True)    # Definimos el vector del espacio principal en x

        self.x_d = (self.x[1:] + self.x[:-1])/2                 # Definimos el vector del espacio dual en x
        self.dt = CFL / ( c * 1/self.dx )                       # Definimos el paso temporal

        """

        Al hacer la reducción de las ecuaciones de Maxwell en 1D trabajamos con Ey y Hz.
        
        """
        self.e = np.zeros( N )
        self.h = np.zeros( N-1 )

        self.w = w*L        # anchura de la PML
        
        def aux(x, par):
            """

            La función aux construye los vectores alpha y beta que aparecen en la discretización
            de las ecuaciones de Maxwell, estos parámetros dependen de sigma_x y dt.

            """

            sig = np.zeros(x.shape)
            L0 = int(len(x)/2)
            sig[:L0] = np.where(x[:L0] < self.w, 20*(x[:L0]-self.w)**2, 0)
            sig[L0:] = np.where(x[L0:] > (L-self.w), 20*(x[L0:]-(L-self.w))**2, 0)

            alpha = par/self.dt + sig/2
            beta = par/self.dt - sig/2

            return alpha, beta

        self.alpha_x, self.beta_x = aux(self.x, eps)       # Sigma_z viviendo donde Ex
        self.alpha_z, self.beta_z = aux(self.x_d, eps)     # Sigma_z viviendo donde Hy

    def update(self):
        """

        Actualización de E y H mediante un solo paso.
            
        """
        e, h = self.e, self.h

        # Imponemos las condiciones de contorno del problema, en el artículo se aplican condiciones PEC

        e[0] = e[-1] = 0 #pec

        # Actualizamos de manera normal e

        e[1:-1] =  self.beta_x[1:-1]/self.alpha_x[1:-1]*e[1:-1] - (1 / self.dx / self.alpha_x[1:-1])*( h[1:] - h[:-1] )

        # Actualizamos de manera normal h
        
        h[:]    =  self.beta_z[:]/self.alpha_z[:]*h[:] - (1 / self.dx / self.alpha_z[:])*( e[1:] - e[:-1] )