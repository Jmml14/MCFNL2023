import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 1001)

x_0 = 3
s_0 = 1
c = 1

for t in range(0, 10):
    f_t = np.exp(-pow(x-x_0-c*t,2)/(2*s_0**2))

    plt.plot(x, f_t)
    plt.grid()
    plt.ylim(-0.1, 1.1)
    plt.xlim(x[0], x[-1])
    plt.pause(0.1)
    plt.cla()