import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import iv, kn, ive
from sympy import symbols, linear_eq_to_matrix
from VolumeConductor import *

# print(-(iv(25, 0.044*np.sqrt(0.5/0.1)*128*25.1327)*kn(25, 0.045*np.sqrt(0.5/0.1)*128*25.1327))/(0.1))
# print(-np.sqrt(0.5/0.1)*iv(25, 0.044*np.sqrt(0.5/0.1)*128*25.1327)*kn_derv(25, 0.045*np.sqrt(0.5/0.1)*128*25.1327))
def matrix(kth, kz, am, rm):
    A = [[1, -((iv(kth, 45*kz))/(iv(kth, 48*kz))), -((kn(kth, 45*kz))/(kn(kth, 48*kz))), 0, 0],
         [np.sqrt(0.0001*0.0005)*((iv_derv(kth, kz*am))/(iv(kth, kz*am))), -0.00005*((iv_derv(kth, kz*45))/(iv(kth, kz*48))), -0.00005*((kn_derv(kth, kz*45))/(kn(kth, kz*48))), 0, 0],
         [0, 1, 1, -((iv(kth, 48*kz))/(iv(kth, 50*kz))), -((kn(kth, 48*kz))/(kn(kth, 50*kz)))],
         [0, 0.05*((iv_derv(kth, kz*0.048))/(iv(kth, kz*0.048))), 0.05*((kn_derv(kth, kz*0.048))/(kn(kth, kz*0.048))), -((iv_derv(kth, kz*0.048))/(iv(kth, kz*0.05))), -((kn_derv(kth, kz*0.048))/(kn(kth, kz*0.05)))],
         [0, 0, 0, ((iv_derv(kth, kz*0.05))/(iv(kth, kz*0.05))), ((kn_derv(kth, kz*0.05))/(kn(kth, kz*0.05)))]]
    b = [[-(iv(kth, rm*kz)*kn(kth, am*kz))/(0.1)],
         [-np.sqrt(0.5/0.1)*iv(kth, rm*kz)*kn_derv(kth, am*kz)],
         [0],
         [0],
         [0]]
    # b = [[0],
    #      [0],
    #      [0],
    #      [0],
    #      [0]]
    #np.savetxt("Atest.csv", A, delimiter=",")
    #np.savetxt("btest.csv", b, delimiter=",")
    c = np.linalg.solve(A, b)
    return c


def compute_test(kz, kth, layers, layers_radial_coord, am, rm):
    mat = matrix(kth, kz, am, rm)
    #####################
    coeffs = get_coeff(kth, kz, mat, layers, layers_radial_coord)
    potential = get_potential(kz, kth, coeffs[len(coeffs) - 2], coeffs[len(coeffs) - 1],
                              layers_radial_coord[len(layers_radial_coord) - 1])
    return potential
