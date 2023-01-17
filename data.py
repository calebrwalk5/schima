# Create data to train on the Mandelbrot set 
import numpy as np
import matplotlib.pyplot as plt

def mandelbrot_set(xmin, xmax, ymin, ymax, xn, yn, max_iter, horizon=2.0):
    X = np.linspace(xmin, xmax, xn, dtype=np.float32)
    Y = np.linspace(ymin, ymax, yn, dtype=np.float32)
    C = X + Y[:, None]*1j
    N = np.zeros_like(C, dtype=int)
    Z = np.zeros_like(C)
    for n in range(max_iter):
        I = np.less(abs(Z), horizon)
        N[I] = n
        Z[I] = Z[I]**2 + C[I]
    N[N == max_iter-1] = 0
    return Z, N

xmin, xmax, ymin, ymax = -2.5, 1.5, -2, 2
xn, yn = (2000, 2000)
max_iter = 256
horizon = 2.0 ** 40
Z, N = mandelbrot_set(xmin, xmax, ymin, ymax, xn, yn, max_iter, horizon)

plt.imshow(N, extent=(xmin, xmax, ymin, ymax), cmap='jet', origin='lower')
plt.show()
