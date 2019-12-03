import numpy as np
from sympy import *
from scipy.misc import derivative
import matplotlib.pyplot as plt


def get_k_beta(kz, kt, v):
    k_beta = kz - (kt/v)
    return k_beta


def get_k_eta(kz, kt, v):
    k_eta = kz + (kt/v)
    return k_eta


def fourier_fi(kt, v, potential):
    pot_derv = diff(potential, Z)
    fourier_voltage_derv = np.fft.fft(pot_derv)
    return fourier_voltage_derv


# def compute_source(voltage, kz, kt, v, L1, L2, z_i):
#     keta = get_k_eta(kz, kt, v)
#     kbeta = get_k_beta(kz, kt, v)
#
#     current = complex(0, 1)*kz*np.exp(-complex(0, 1)*kz*z_i)*fourier_potential_derv(kt, v, voltage)\
#               *(np.exp(-complex(0, 1)*keta*L1/2)*(np.sin(keta*L1/2))/(keta/2)
#                 - np.exp(complex(0, 1)*kbeta*L2/2)*(np.sin(kbeta*L2/2))/(kbeta/2))
#     return current


def pL1 (z, z0, L1):
    x = z-z0-L1/2
    if x >= -L1/2 and x <= L1/2:
        return 1
    else:
        return 0


def pL2 (z, z0, L2):
    x = z-z0+L2/2
    if x >= -L2/2 and x <= L2/2:
        return 1
    else:
        return 0


def plot_signal(z, I1, I2):
    #current = derivative(first_der, Z)
    plt.plot(z, I1)
    plt.plot(z, I2)
    plt.show()


z0 = 0
z = np.linspace(-90, 60, 1000)
v = 4000

L1 = 50
L2 = 80
I1 = []
I2 = []

Z = symbols('Z')

V = 96*(Z**3)*(exp(-Z)) - 90
t = 0.000001
x = fourier_fi(5,6, V)
print(x)

for zi in z:
    fi = diff(V, Z)
    fii = diff(fi, Z)
    #print(fi)
    z1 = z0+v*t
    z2 = z0-v*t
    pl1 = pL1(zi, z0, L1)
    pl2 = pL2(zi, z0, L2)
    fi1 = fi.subs({Z: Z - z1})*pl1
    fi2 = fi.subs({Z: -Z + z2})*pl2
    fi1 = fi1.subs({Z: zi}) * pl1
    fi2 = -fi2.subs({Z: zi}) * pl2
    #current1 = derivative(fi, zi+z1)*pl1
    #print(fii1)
    # fii1 = fii1.subs({Z:zi-z0})
    # fii2 = derivative(V, zi-z0) * pl2
    # #current2 = derivative(fi, zi+z2)*pl2
    I1.append(fi1)
    I2.append(fi2)
    #print('d')
plot_signal(z, I1, I2)
