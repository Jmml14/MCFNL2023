import numpy as np
import matplotlib.pyplot as plt

eps = 1.0                   # Permitividad eléctrica
mu = 1.0                    # Permeabilidad magnética
c = 1/np.sqrt(eps*mu)       # Velocidad de la onda

class FDTD_PML2D():
    def __init__(self, L=20, CFL=1.0, N=101):
        """

        Clase FDTD_PML2D: toma como argumentos:
        - L:   Espacio de la simulación
        - w:   Porcentaje del ancho de la PML
        - CFL: Valor de la condición CFL de la simulación
        - N:   Partición espacial de la simulación

        Esta clase construye las matrices de las conductividades sigma_x y sigma_y, además añadimos
        una esfera absorvente en la zona de simulación como si fuera un obstáculo. Por otro lado, contruye 
        dos métodos para actualizar los campos eléctrico y magnético teniendo en cuenta que es necesario 
        usar un doble paso.

        """

        self.L = L
        self.N = N
        self.vx, self.dx = np.linspace(0, L, N, retstep=True)       # Definimos el vector del espacio principal en x
        self.vy, self.dy = np.linspace(0, L, N, retstep=True)       # Definimos el vector del espacio principal en y

        self.vx_d = (self.vx[1:] + self.vx[:-1])/2                          # Definimos el vector del espacio dual en x
        self.vy_d = (self.vy[1:] + self.vy[:-1])/2                          # Definimos el vector del espacio dual en y
        self.dt = CFL / ( c * np.sqrt( 1/self.dx**2 + 1/self.dy**2 ) )      # Definimos el paso temporal

        """

        Al hacer la reducción de las ecuaciones de Maxwell en 2D trabajamos con el modo TE (Ex, Ey, Hz). Debido
        a que vamos a usar un doble paso para actualizar Ex, Ey y Hz es necesario definir las matrices para el 
        Desplazamiento eléctrico D y el campo B.
        
        """
        
        self.e_x = np.zeros( (N, N-1) )
        self.d_x = np.zeros( (N, N-1) )
        self.e_y = np.zeros( (N-1, N) )
        self.d_y = np.zeros( (N-1, N) )
        self.h_z = np.zeros( (N-1, N-1) )
        self.b_z = np.zeros( (N-1, N-1) )

        self.d_xnew = np.zeros( (N, N-1) )
        self.d_ynew = np.zeros( (N-1, N) )
        self.b_znew = np.zeros( (N-1, N-1) )

        self.w = 1*L/10              # anchura de la PML
        
        def aux(x, Ny, par, yx, donde):
            """

            La función aux construye las matrices alpha y beta que aparecen en la discretización
            de las ecuaciones de Maxwell, estos parámetros dependen de sigma_xy y dt.

            """

            sig = np.zeros(x.shape)
            L0 = int(len(x)/2)
            sig[:L0] = np.where(x[:L0] < self.w, 70*(x[:L0]-self.w)**2, 0)
            sig[L0:] = np.where(x[L0:] > (L-self.w), 70*(x[L0:]-(L-self.w))**2, 0)
            
            x_ex, y_ex = np.meshgrid( self.vx_d, self.vy )
            x_ey, y_ey = np.meshgrid( self.vx, self.vy_d )
            x_hz, y_hz = np.meshgrid( self.vx_d, self.vy_d )

            if yx < 0 :
                sig = sig + np.zeros((Ny,1))
                if -2 < donde < 0:
                    sig = sig + 5*np.exp(-(pow(x_ex[:, :]-3*L/4, 2) + pow(y_ex[:, :]-1*L/2, 2))/(2*1**2))
                elif donde < -2:
                    sig = sig + 5*np.exp(-(pow(x_ey[:, :]-3*L/4, 2) + pow(y_ey[:, :]-1*L/2, 2))/(2*1**2))
                else:
                    sig = sig + 5*np.exp(-(pow(x_hz[:, :]-3*L/4, 2) + pow(y_hz[:, :]-1*L/2, 2))/(2*1**2))
            else:
                sig = (sig + np.zeros((Ny,1))).transpose()
                if -2 < donde < 0:
                    sig = sig + 5*np.exp(-(pow(x_ex[:, :]-3*L/4, 2) + pow(y_ex[:, :]-1*L/2, 2))/(2*1**2))
                elif donde < -2:
                    sig = sig + 5*np.exp(-(pow(x_ey[:, :]-3*L/4, 2) + pow(y_ey[:, :]-1*L/2, 2))/(2*1**2))
                else:
                    sig = sig + 5*np.exp(-(pow(x_hz[:, :]-3*L/4, 2) + pow(y_hz[:, :]-1*L/2, 2))/(2*1**2))
            alpha = par/self.dt - sig/2
            beta = par/self.dt + sig/2
            
            return alpha, beta

        self.alpha_xx, self.beta_xx = aux(self.vx_d, N, eps, -1, -1) # Sigma_x viviendo donde Ex
        self.alpha_yx, self.beta_yx = aux(self.vy, N-1, eps, 1, -1)   # Sigma_y viviendo donde Ex

        self.alpha_xy, self.beta_xy = aux(self.vx, N-1, eps, -1, -4) # Sigma_x viviendo donde Ey
        self.alpha_yy, self.beta_yy = aux(self.vy_d, N, eps, 1, -4)  # Sigma_y viviendo donde Ey

        self.alpha_xz, self.beta_xz = aux(self.vx_d, N-1, mu, -1, 1) # Sigma_x viviendo donde Hz
        self.alpha_yz, self.beta_yz = aux(self.vy_d, N-1, eps, 1, 1) # Sigma_y viviendo donde Hz

    def update_h(self):
        """

        Actualización de Hz usando doble paso.
            
        """

        e_x, e_y= self.e_x, self.e_y
        h_z, b_z = self.h_z, self.b_z

        b_znew = self.alpha_yz[:,:]/self.beta_yz[:,:]*b_z[:,:] + 1/self.beta_yz[:,:]*( -(e_y[:, 1:] - e_y[:, :-1])/self.dx + (e_x[1:, :] - e_x[:-1, :])/self.dy )
        h_z[:,:] = self.alpha_xz[:,:]/self.beta_xz[:,:]*h_z[:,:] + 1/self.beta_xz[:,:]*( b_znew[:,:] - b_z[:,:] )/self.dt
        b_z = b_znew

        return h_z, b_z

    def update_e(self):
        """

        Actualización de Ex, Ey usando doble paso.
            
        """

        e_x, d_x = self.e_x, self.d_x
        e_y, d_y = self.e_y, self.d_y
        d_xnew, d_ynew, h_z = self.d_xnew, self.d_ynew, self.h_z

        # Imponemos las condiciones de contorno del problema, en el artículo se aplican condiciones PEC

        e_x[:, 0] = e_y[:, -1] = 0 #pec
        e_y[0, :] = e_y[-1, :] = 0 #pec

        # Actualizamos con doble paso Ex

        d_xnew[1:-1,:] = d_x[1:-1,:] + self.dt/eps*( (h_z[1:,:] - h_z[:-1,:])/self.dy )
        e_x = self.alpha_yx[1:-1,:]/self.beta_yx[1:-1,:] * e_x[1:-1,:] + 1/self.beta_yx[1:-1,:]*( self.beta_xx[1:-1,:]*d_xnew[1:-1,:] - self.alpha_xx[1:-1,:]*d_x[1:-1,:] )

        # Actualizamos con doble paso Ey

        d_ynew[:, 1:-1] = d_y[:,1:-1] - self.dt/eps*( (h_z[:,1:] - h_z[:,:-1])/self.dx )
        e_y = self.alpha_xy[:, 1:-1]/self.beta_xy[:, 1:-1] * e_y[:,1:-1] + 1/self.beta_xy[:,1:-1]*( self.beta_yy[:,1:-1]*d_ynew[:,1:-1] - self.alpha_yy[:,1:-1]*d_y[:,1:-1] )

        d_x[:,:] = d_xnew
        d_y[:,:] = d_ynew

        return e_x, e_y, d_x, d_y