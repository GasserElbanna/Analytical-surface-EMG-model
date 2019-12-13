import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import iv, kn, ive, kve, kv
from sympy import *
from scipy.misc import derivative

def iv_derv(v, z):
    derv_I = (ive(v+1, z) + ive(v-1, z))/2
    return derv_I


def kn_derv(n, z):
    derv_K = -(kv(n+1, z) + kv(n-1, z))/2
    return derv_K


a = np.array([[0.28472*10**-4, -0.13749*10**9, 0.13345*10**-11],
             [0.29957*10**-4, -0.10583*10**10, -0.10144*10**-10],
             [0, 0.40524*10**12, -0.15633*10**-12]])


b = np.array([[ive(30, 2000*0.018*np.sqrt(0.5/0.1)), -ive(30, 2000*0.018), -kv(30, 2000*0.018)],
             [np.sqrt(0.5*0.1)*iv_derv(30, 2000*0.018*np.sqrt(0.5/0.1)), -np.sqrt(0.05*0.05)*iv_derv(30, 2000*0.018), -np.sqrt(0.05*0.05)*kn_derv(30, 2000*0.018)],
             [0, np.sqrt(0.05*0.05)*iv_derv(30, 2000*0.02), np.sqrt(0.05*0.05)*kn_derv(30, 2000*0.02)]])
cond1 = np.linalg.cond(a)
cond2 = np.linalg.cond(b)
print(iv(0,0), kn(0,0))
print(b)
for i in range(-2,3,1):
    print('a')
