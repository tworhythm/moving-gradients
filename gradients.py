import matplotlib.pyplot as plt
import numpy as np
from gco_python_master.pygco import cut_simple
from scipy import misc, ndimage

image_A = misc.imread('A.png')[:,:,0:3]
image_B = misc.imread('B.png')[:,:,0:3]
h, w, dim = image_A.shape

all_paths = generate_all_paths(-15, 15)


def generate_all_paths(min_displacement, max_displacement):
    range_displacement = max_displacement - min_displacement + 1
    all_paths = np.zeros((range_displacement*range_displacement, 2), 'int16')

    counter = 0
    for dy in range(min_displacement, max_displacement):
        for dx in range(min_displacement, max_displacement):
            all_paths[counter, 0] = dy
            all_paths[counter, 1] = dx
            counter += 1
    return all_paths

def matching_cost(pA, pB, A, B, gradient_A, gradient_B, variance_A, variance_B, beta):
    Ay, Ax = pA
    By, Bx = pB
    color_diff = np.sum(np.power(np.subtract(A[Ay, Ax, :].astype('int32'), B[By, Bx, :].astype('int32')), 2))
    gradient_diff = 0
    q = gradient_diff + 0.5*color_diff

    return np.power(np.true_divide(np.power(q, 2), variance_A[Ay, Ax] * variance_B[By, Bx]), beta)

def variance_4_neighbourhood(A):
    kernel = np.array([[0,1,0],[1,1,1],[0,1,0]])

    number_neighbours = ndimage.convolve(np.sum(np.ones_like(A), axis=2), kernel, mode='constant', cval=0)
    sum_neighbours = ndimage.convolve(np.sum(A, axis=2), kernel, mode='constant', cval=0)
    mean_neighbours = np.true_divide(sum_neighbours, number_neighbours)

    sum_squared_neighbours = ndimage.convolve(np.sum(np.power(A.astype('int32'), 2), axis=2), kernel, mode='constant', cval=0)

    return np.true_divide(sum_squared_neighbours, number_neighbours) - np.power(mean_neighbours, 2)