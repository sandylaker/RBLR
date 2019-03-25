from rblr import Huber
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('MacOSX')

font = {
    'family': 'Arial',
    'size': 16
}

h = Huber()
x = np.linspace(-5, 5, 1000)
rho_arr = h.rho(x)
psi_arr = h.psi(x)
weight_arr = h.m_weights(x)

f1 = plt.figure(1)
plt.plot(x, rho_arr)
plt.xlabel('x', fontdict=font)
plt.ylabel(r'$\rho$', fontdict=font)
plt.title(r'$\rho(x)$',fontdict=font)


f2 = plt.figure(2)
plt.plot(x, psi_arr)
plt.xlabel('x', fontdict=font)
plt.ylabel(r'$\psi$',fontdict=font)
plt.title(r'$\psi(x)$', fontdict=font)

f3 = plt.figure(3)
plt.plot(x, weight_arr)
plt.xlabel('x', fontdict=font)
plt.ylabel('weight', fontdict=font)
plt.title('Weight function', fontdict=font)
plt.show()


