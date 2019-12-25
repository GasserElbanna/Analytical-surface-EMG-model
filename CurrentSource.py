import numpy as np
from sympy import *
from scipy.misc import derivative
import matplotlib.pyplot as plt
import scipy.signal


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


def compute_windowed_intensity_1st_signal(I, v, t, z_step):
    reversed_I = np.zeros(len(I))
    shifted_I = np.zeros(len(I))
    windowed_I = np.zeros(len(I))
    # for i in range(len(I)):
    #     reversed_I[i] = I[len(I)-1-i]
    for i in range(len(I)):
        zi = i * z_step - 90
        if zi < 1 and zi >= 0:
            for k in range(len(I)-i):
                reversed_I[i-k] = I[i+k]
                #reversed_I[i+k] = 0
            break
    iterator = int((v * t) / z_step)
    if iterator == 0:
        shifted_I = reversed_I
    else:
        for j in range(len(reversed_I)):
            shifted_I[j] = reversed_I[j-iterator]
    for k in range(len(shifted_I)):
        zi = k * z_step - 90
        if zi > 50 or zi < 0:
            windowed_I[k] = 0
        else:
            windowed_I[k] = shifted_I[k]
    return windowed_I


def compute_windowed_intensity_2nd_signal(I, v, t, z_step):
    shifted_I = np.zeros(len(I))
    windowed_I = np.zeros(len(I))
    # for i in range(len(I)):
    #     reversed_I[i] = I[len(I)-1-i]
    flipped_I = np.negative(I)
    iterator = int((v * t) / z_step)
    if iterator == 0:
        shifted_I = flipped_I
    else:
        for j in range(len(flipped_I)):
            if j+iterator >= len(flipped_I):
                break
            else:
                shifted_I[j] = flipped_I[j + iterator]
    for k in range(len(shifted_I)):
        zi = k * z_step - 90
        if zi < -80 or zi > 0:
            windowed_I[k] = 0
        else:
            windowed_I[k] = shifted_I[k]
    return windowed_I


def compute_current_source(z_step, t_step, v, intensity, fii):
    for w1 in range(1000):
        t = w1 * t_step
        I1 = np.zeros(1000)
        for w2 in range(1000):
            zi = w2 * 0.151515152 - 90
            pl1 = pL1(zi, z0, L1)
            I1[w2] = fii.subs({Z: zi}) * pl1
        signal1 = compute_windowed_intensity_1st_signal(I1, v, t, 0.151515152)
        signal2 = compute_windowed_intensity_2nd_signal(I1, v, t, 0.151515152)
        intensity[w1, :] = signal1 + signal2
        plot_signal(z, signal1, signal2)
    return np.fft.fft2(intensity)


def plot_signal(z, I, I2):
    #current = derivative(first_der, Z)
    plt.plot(z, I)
    plt.plot(z, I2)
    plt.xlabel("Distance (mm)")
    plt.ylabel("Voltage V(Z)")
    plt.title("Signal at T = 50*t_step")
    plt.grid()
    plt.savefig('t02.png')
    plt.show()


z0 = 0

v = 4000
fs = 4096
t_step = 1/fs
tmax = 1000*t_step

L1 = 50
L2 = 80


Z = symbols('Z')
V = 96*(Z**3)*(exp(-Z)) - 90

z = np.linspace(-90, 60, 1000)
time = np.linspace(0, tmax, 1000)

fi = diff(V, Z)
fii = diff(fi, Z)
intensity = np.zeros((1000, 1000))
compute_current_source(0, t_step, v, intensity, fii)