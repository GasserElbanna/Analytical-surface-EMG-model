import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import iv, kn, ive
from sympy import symbols, linear_eq_to_matrix


def get_distance(sigz, sigp, x):
    y = np.sqrt(sigz/sigp)*x
    return y


def iv_derv(v, z):
    derv_I = (iv(v+1, z) + iv(v-1, z))/2
    return derv_I


def kn_derv(n, z):
    derv_K = -(kn(n+1, z) + kn(n-1, z))/2
    return derv_K


##########################################Get Coefficients##############################################################
def field_potential_equation(kz, kth, coeff1, coeff2, layer_radial_coord, measured_layer_radial_coord, source_radial_coord, cond, i):
    pot = ((coeff1 * iv(kth, kz * layer_radial_coord)) / iv(kth, kz * measured_layer_radial_coord)) \
            + ((coeff2 * kn(kth, kz * layer_radial_coord)) / (kn(kth, kz * measured_layer_radial_coord)))
    if layer_radial_coord < source_radial_coord and i == 1:
        pot += (iv(kth, kz * layer_radial_coord) * kn(kth, kz * source_radial_coord)) / cond
    elif layer_radial_coord > source_radial_coord and i == 1:
        pot += (iv(kth, kz * source_radial_coord) * kn(kth, kz * layer_radial_coord)) / cond
    return pot


def field_potential_derv_equation(kz, kth, coeff1, coeff2, layer_radial_coord, measured_layer_radial_coord, source_radial_coord, cond_row, cond_z, i):
    pot = (coeff1 * np.sqrt(cond_row * cond_z) * iv_derv(kth, kz * layer_radial_coord)) / iv(kth, kz * measured_layer_radial_coord) \
            + (coeff2 * np.sqrt(cond_row * cond_z) * kn_derv(kth, kz * layer_radial_coord)) / kn(kth, kz *measured_layer_radial_coord)
    if layer_radial_coord < source_radial_coord and i == 1:
        pot += np.sqrt(cond_z / cond_row) * (iv_derv(kth, kz * layer_radial_coord) * kn(kth, kz * source_radial_coord))
    elif layer_radial_coord > source_radial_coord and i == 1:
        pot += np.sqrt(cond_z / cond_row) * (iv(kth, kz * source_radial_coord) * kn_derv(kth, kz * layer_radial_coord))
    return pot


def bound_cond(bound_conds, equ1, equ2):
    bound_conds.append(equ1 - equ2)


def linear_to_matrix(equs, coeff):
    A, b = linear_eq_to_matrix(equs, coeff)
    A = np.array(A).astype(np.float64)
    # cond = np.linalg.cond(A)
    # print(cond)
    b = np.array(b).astype(np.float64)
    # np.savetxt("A.csv", A, delimiter=",")
    # np.savetxt("b.csv", b, delimiter=",")
    #print(A, b)
    s = np.linalg.solve(A, b)
    return s


def get_coeff(kth, kz, s, row, rowm):
    #print(s)
    s1 = []
    for i in range(len(row)):
        if i == 0:
            s1.append(s[i] / iv(kth, kz * rowm[i]))
        else:
            s1.append(s[2 * i - 1] / iv(kth, kz * rowm[2 * i]))
            s1.append(s[2 * i] / kn(kth, kz * rowm[2 * i]))
    return s1
########################################################################################################################


###############################################Compute Field Potential##################################################
def get_potential(kz, kth, coeff1, coeff2, row1):
    pot = (coeff1 * iv(kth, kz * row1)) + (coeff2 * kn(kth, kz * row1))
    return pot


def normalize_potential(array, pot_max):
    for i in range(len(array)):
        for j in range(len(array[0])):
            array[i, j] = array[i, j] / pot_max
    return array
########################################################################################################################


#################################################Mirroring Functions####################################################
def get_negative_axes(array):
    neg_array = []
    for i in range(len(array)):
        neg_array.append(-array[len(array)-1-i])
    return neg_array


def get_inverse_potential(potential):
    inverse = np.zeros((len(potential), len(potential[0])))
    for i in range(len(potential)):
        for j in range(len(potential[0])):
            inverse[i, j] = potential[len(potential) - i - 1, len(potential[0]) - j - 1]
    return inverse


def get_inverse_potential_row(potential):
    inverse = np.zeros((len(potential), len(potential[0])))
    for i in range(len(potential)):
        for j in range(len(potential[0])):
            inverse[i, j] = potential[len(potential) - i - 1, j]
    return inverse


def get_inverse_potential_col(potential):
    inverse = np.zeros((len(potential), len(potential[0])))
    for i in range(len(potential)):
        for j in range(len(potential[0])):
            inverse[i, j] = potential[i, len(potential[0]) - j - 1]
    return inverse
########################################################################################################################


def show_impulse_response(potential, z, th):
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
    ax.plot_surface(Z, TH, total_pot)
    ax.set_xlabel('Z')
    ax.set_ylabel('Theta')
    ax.set_zlabel('Potential')
    pyplot.show()
    pyplot.savefig('Graph.png')


def compute_vol_cond(kz, kth, sym, layers, layers_radial_coord, source_radial_coord, cond):
    # Save list of equations
    equs_1 = []
    equs_2 = []
    bound_conds = []
    coeff = []
    for i in range(len(layers_radial_coord)):
        coeff.append(sym[i])
    #######################

    # Boundary Conditions
    for i in range(len(layers)):
        if i == 0:
            pot = field_potential_equation(kz, kth, coeff[i], 0, layers_radial_coord[i], layers_radial_coord[i],
                                           source_radial_coord[i], cond[i], i)
            derv_pot = field_potential_derv_equation(kz, kth, coeff[i], 0, layers_radial_coord[i],
                                                     layers_radial_coord[i], source_radial_coord[i], cond[i], cond[i + 1], i)
            equs_1.append(pot)
            equs_2.append(derv_pot)
        elif i == len(layers) - 1:
            pot = field_potential_equation(kz, kth, coeff[2 * i - 1], coeff[2 * i], layers_radial_coord[2 * i - 1],
                                           layers_radial_coord[2 * i], source_radial_coord[len(source_radial_coord) - 1],
                                           cond[len(cond) - 2], i)
            derv_pot1 = field_potential_derv_equation(kz, kth, coeff[2 * i - 1], coeff[2 * i], layers_radial_coord[2 * i - 1],
                                                      layers_radial_coord[2 * i], source_radial_coord[i], cond[len(cond) - 2],
                                                      cond[len(cond) - 1], i)
            derv_pot2 = field_potential_derv_equation(kz, kth, coeff[2 * i - 1], coeff[2 * i], layers_radial_coord[2 * i],
                                                      layers_radial_coord[2 * i], source_radial_coord[i], cond[len(cond) - 2],
                                                      cond[len(cond) - 1], i)
            equs_1.append(pot)
            equs_2.append(derv_pot1)
            equs_2.append(derv_pot2)
        else:
            pot1 = field_potential_equation(kz, kth, coeff[2 * i - 1], coeff[2 * i], layers_radial_coord[2 * i - 1],
                                            layers_radial_coord[2 * i], source_radial_coord[i], cond[2 * i], i)
            pot2 = field_potential_equation(kz, kth, coeff[2 * i - 1], coeff[2 * i], layers_radial_coord[2 * i],
                                            layers_radial_coord[2 * i], source_radial_coord[i], cond[2 * i], i)
            derv_pot1 = field_potential_derv_equation(kz, kth, coeff[2 * i - 1], coeff[2 * i],
                                                      layers_radial_coord[2 * i - 1], layers_radial_coord[2 * i],
                                                      source_radial_coord[i], cond[2 * i], cond[2 * i + 1], i)
            derv_pot2 = field_potential_derv_equation(kz, kth, coeff[2 * i - 1], coeff[2 * i],
                                                      layers_radial_coord[2 * i], layers_radial_coord[2 * i],
                                                      source_radial_coord[i], cond[2 * i], cond[2 * i + 1], i)
            equs_1.append(pot1)
            equs_1.append(pot2)
            equs_2.append(derv_pot1)
            equs_2.append(derv_pot2)

    for i in range(len(layers)):
        if i == len(layers) - 1:
            bound_cond(bound_conds, equs_2[len(equs_2) - 1], 0)
        else:
            bound_cond(bound_conds, equs_1[2 * i], equs_1[2 * i + 1])
            bound_cond(bound_conds, equs_2[2 * i], equs_2[2 * i + 1])
    ####################
    # Linear Equ to Matrix
    matrix = linear_to_matrix(bound_conds, coeff)
    #####################
    coeffs = get_coeff(kth, kz, matrix, layers, layers_radial_coord)
    potential = get_potential(kz, kth, coeffs[len(coeffs)-2], coeffs[len(coeffs)-1], layers_radial_coord[len(layers_radial_coord)-1])
    return potential
