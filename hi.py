import numpy as np
from sympy import *
from scipy.misc import derivative
import matplotlib.pyplot as plt
import scipy.signal
from scipy.special import iv, kn, ive
from VolumeConductor import *


# # signal
# input = np.random.rand(100)
# window = scipy.signal.boxcar(10)
# # Pad the window to make its size equal to signal size
# # I'm assuming your peak is between sample 45 and 55
# window = np.lib.pad(window, (45, 45), 'constant')
# output = input*window
#
# Z = symbols('Z')
# z = np.linspace(0, 50, 200)
# v = 96*(z**3)*(np.exp(-z)) - 90
# fi = -96*z**3*np.exp(-z) + 288*z**2*np.exp(-z)
# fii = 96*z**3*np.exp(-z) - 576*z**2*np.exp(-z) + 576*z*np.exp(-z)
#
#
# plt.plot(-z, fi, 'g')
# plt.title('Second Derivative')
# plt.xlabel('Distance (Z)')
# plt.ylabel('Voltage')
# #plt.savefig('secondDerv.png')
# plt.show()
volume_length = 125
velocity = 4000
f_sampling = 4096


kz_max = (np.pi*f_sampling)/velocity
bins = int((2*f_sampling*volume_length)/velocity)
kz_step = np.pi/volume_length
z = np.linspace(0, volume_length, int(bins/2))
th = np.linspace(0, np.pi, 26)
vol_cond_spatial_freq = np.zeros((26, (int(bins / 2))), dtype=np.complex)
vol_cond_spatial = np.zeros((26, (int(bins / 2))), dtype=np.complex)
coeff11 = np.linspace(0, 100000, 1000)
coeff22 = np.linspace(0, 100000, 1000)
count = 0


def save_result(z, th, potential, i):
    neg_z = get_negative_axes(z)
    neg_th = get_negative_axes(th)
    z_all = np.concatenate((neg_z, z), axis=0)
    th_all = np.concatenate((neg_th, th), axis=0)
    Z, TH = np.meshgrid(z_all, th_all)
    inv_potential = get_inverse_potential(potential)
    inv_potential_row = get_inverse_potential_row(potential)
    inv_potential_col = get_inverse_potential_col(potential)
    pot_1 = np.concatenate((inv_potential, inv_potential_row), axis=1)
    pot_2 = np.concatenate((inv_potential_col, potential), axis=1)
    total_pot = np.concatenate((pot_1, pot_2), axis=0)
    fig = pyplot.figure(figsize=(8, 4))
    ax = fig.gca(projection='3d')
    ax.plot_surface(TH, Z, total_pot)
    ax.set_xlabel('Theta')
    ax.set_ylabel('Z')
    ax.set_zlabel('Potential')
    pyplot.savefig('test/img' + str(i) + '.png')


for coeff1 in coeff11:
    for coeff2 in coeff22:
        for w1 in range(26):
            for w2 in range(int(bins/2)):
                if w2 == 0:
                    kz = 0.0000001*kz_step
                else:
                    kz = w2*kz_step
                kth = w1
                vol_cond_spatial_freq[w1, w2] = (coeff1 * iv(kth, kz * 50)) + (coeff2 * kn(kth, kz * 50))
        vol_cond_spatial = np.fft.ifft2(vol_cond_spatial_freq)
        mag = np.abs(vol_cond_spatial)
        mag_max = np.amax(mag)
        mag_normalized = normalize_potential(mag, mag_max)
        count += 1
        save_result(z, th, mag_normalized, count)