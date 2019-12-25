import numpy as np
from sympy import *
from VolumeConductor import *
#from Current_Source import *
from DetectionSys import *
from test import *

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
Z = symbols('Z')
V = 96*(Z**3)*(exp(-Z)) - 90
fi = diff(V, Z)
fii = diff(fi, Z)
velocity = 4000 #in mm/sec
fiber_length = 0.080 #in m
fiber_depth = 6 #in mm
f_sampling = 4096 #in Hz
no_fibers = 2
L1 = 50 #in mm
L2 = 80 #in mm
z0 = 0
intensity = np.zeros((100, 100))
##################################
#Volume Conductor Input Parameters
# a = 0.020 #in m
# b = 0.045 #in m
# c = 0.048 #in m
# d = 0.050 #in m

a = 45 #in mm
b = 48 #in mm
c = 50 #in mm

R = 44 #in mm
volume_length = 125 #in mm

sigsz = 0.001 #in S/mm
sigsp = 0.001
sigfz = 0.00005
sigfp = 0.00005
sigmz = 0.0005
sigmp = 0.0001
sigbz = 0.00002
sigbp = 0.00002
####################################

# a1 = get_distance(sigbz, sigbp, a)
# a2 = get_distance(sigmz, sigmp, a)
# b2 = get_distance(sigmz, sigmp, b)
# b3 = get_distance(sigfz, sigfp, b)
# c3 = get_distance(sigfz, sigfp, c)
# c4 = get_distance(sigsz, sigsp, c)
# d4 = get_distance(sigsz, sigsp, d)

a1 = get_distance(sigmz, sigmp, a)
a2 = get_distance(sigfz, sigfp, a)
b2 = get_distance(sigfz, sigfp, b)
b3 = get_distance(sigsz, sigsp, b)
c3 = get_distance(sigsz, sigsp, c)


# Rm = get_distance(sigmz, sigmp, R)
# layers = [a, b, c, d]
# layers_radial_coord = [a1, a2, b2, b3, c3, c4, d4]
# source_radial_coord = [R, Rm, R, R]
# cond = [sigbp, sigbz, sigmp, sigmz, sigfp, sigfz, sigsp, sigsz]

Rm = get_distance(sigmz, sigmp, R)
layers = [a, b, c]
layers_radial_coord = [a1, a, b, b, c]
source_radial_coord = [Rm, R, R]
cond = [sigmp, sigmz, sigfp, sigfz, sigsp, sigsz]

A, B, C, D, E, F, G, H, I, J, K = symbols('A B C D E F G H I J K')
sym = [A, B, C, D, E, F, G, H, I, J, K]


# Sampling and Resolutions
kz_max = (np.pi*f_sampling)/velocity
bins = int((2*f_sampling*volume_length)/velocity)
kz_step = np.pi/volume_length

kth_step = 1
kth_max = int((50*kth_step)/2)

t_step = 1/f_sampling
t_max = 100*t_step
z_step = 1/kz_max
th_step = int(1/kth_max)


time = np.linspace(0, t_max, 100)
z = np.linspace(0, volume_length, int(bins/2))
th = np.linspace(0, np.pi, 26)
############################


vol_cond_spatial_freq = np.zeros((26, (int(bins/2))), dtype=np.complex)
vol_cond_spatial = np.zeros((26, (int(bins/2))), dtype=np.complex)




for w1 in range(26):
    for w2 in range(int(bins/2)):
        if w2 == 0:
            kz = 0.0000001*kz_step
        else:
            kz = w2*kz_step
        kth = w1

        vol_cond_spatial_freq[w1, w2] = compute_vol_cond(kz, kth, sym, layers, layers_radial_coord, source_radial_coord, cond)
        #vol_cond_spatial_freq[w1, w2] = compute_test(kz, kth, layers, layers_radial_coord, a2, Rm)


print(vol_cond_spatial_freq)
vol_cond_spatial = np.fft.ifft2(vol_cond_spatial_freq)
mag = np.abs(vol_cond_spatial)
np.savetxt("mag_sim8.csv", mag, delimiter=",")
print(len(mag), len(mag[0]))
print(mag)
mag_max = np.amax(mag)
mag_normalized = normalize_potential(mag, mag_max)
show_impulse_response(mag_normalized, z, th)