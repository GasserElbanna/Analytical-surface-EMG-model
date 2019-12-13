import numpy as np
from sympy import symbols
from VolumeConductor import *
#from Current_Source import *
from DetectionSys import *

#Spatial filter Input Parameters
z_elec = 3
th_elec = 3
L = 0.001
W = 0.001
z_dist = 0.005
dist = 0.005
Relec = 0.025
alpha = 5 #in degrees
elec_weighting = [[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]]
elec_weighting = np.array(elec_weighting)
##################################
#Source Input Parameters
z_spatial = symbols('z_spatial')
#membrane_voltage = (96*z_spatial**3*np.exp(-z_spatial)) - 90
velocity = 4 #in m/sec
fiber_length = 0.080 #in m
fiber_depth = 6 #in mm
f_sampling = 4096 #in Hz
no_fibers = 2
L1 = 0.025
L2 = 0.025
z0 = 0
##################################
#Volume Conductor Input Parameters
a = 0.020 #in m
b = 0.045 #in m
c = 0.048 #in m
d = 0.050 #in m

# a = 0.025 #in m
# b = 0.027 #in m
# c = 0.030 #in m

R = 0.044 #in m
volume_length = 0.125 #in m

sigsz = 1 #in S/m
sigsp = 1
sigfz = 0.05
sigfp = 0.05
sigmz = 0.5
sigmp = 0.1
sigbz = 0.02
sigbp = 0.02
####################################

a1 = get_distance(sigbz, sigbp, a)
a2 = get_distance(sigmz, sigmp, a)
b2 = get_distance(sigmz, sigmp, b)
b3 = get_distance(sigfz, sigfp, b)
c3 = get_distance(sigfz, sigfp, c)
c4 = get_distance(sigsz, sigsp, c)
d4 = get_distance(sigsz, sigsp, d)


Rm = get_distance(sigmz, sigmp, R)
layers = [a, b, c, d]
layers_radial_coord = [a1, a2, b2, b3, c3, c4, d4]
source_radial_coord = [R, Rm, R, R]
cond = [sigbp, sigbz, sigmp, sigmz, sigfp, sigfz, sigsp, sigsz]

# Rm = get_distance(sigmz, sigmp, R)
# layers = [a, b, c]
# layers_radial_coord = [a1, a2, b2, b3, c3]
# source_radial_coord = [Rm, R, R]
# cond = [sigmp, sigmz, sigfp, sigfz, sigsp, sigsz]

A, B, C, D, E, F, G, H, I, J, K = symbols('A B C D E F G H I J K')
sym = [A, B, C, D, E, F, G, H, I, J, K]


# Sampling and Resolutions
kz_max = (np.pi*f_sampling)/velocity
bins = int((2*f_sampling*volume_length)/velocity)
kz_step = (2*kz_max)/bins

kth_step = 1
kth_max = int((50*kth_step)/2)

z_step = 1/kz_max
th_step = int(1/kth_max)
############################


vol_cond_spatial_freq = np.zeros((26, (int(bins/2)+1)), dtype=np.complex)
vol_cond_spatial = np.zeros((26, (int(bins/2)+1)), dtype=np.complex)

z = np.linspace(0, volume_length, int(bins/2)+1)
th = np.linspace(0, np.pi, 26)


for w1 in range(26):
    for w2 in range(int(bins/2)+1):
        if w2 == 0:
            kz = 0.0000001*kz_step
        else:
            kz = w2*kz_step
        kth = w1

        vol_cond_spatial_freq[w1, w2] = compute_vol_cond(kz, kth, sym, layers, layers_radial_coord, source_radial_coord, cond)


print(vol_cond_spatial_freq)
vol_cond_spatial = np.fft.ifft2(vol_cond_spatial_freq)
mag = np.abs(vol_cond_spatial)
np.savetxt("mag_sim1.csv", mag, delimiter=",")
print(len(mag), len(mag[0]))
print(mag)
mag_max = np.amax(mag)
mag_normalized = normalize_potential(mag, mag_max)
show_impulse_response(mag_normalized, z, th)