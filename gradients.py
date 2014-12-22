import matplotlib.pyplot as plt
import numpy as np
from gco_python_master.pygco import cut_simple
from scipy import misc, ndimage

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

def variance_4_neighbourhood(values):
    # numerically stable variance by using shifted data
    h, w, dim = values.shape
    variance = np.zeros((h,w), dtype='int32')
    for y in range(0, h):
        for x in range(0, w):
            K = values[y,x]
            n = 1
            sum_neighbourhood = 0
            sum_squared_neighbourhood = 0
            if y > 0:
                n += 1
                sum_neighbourhood += np.sum(np.subtract(values[y - 1, x, :], K))
                sum_squared_neighbourhood += np.sum(np.power(np.subtract(values[y - 1, x, :], K), 2))
            if y < (h - 1):
                n += 1
                sum_neighbourhood += np.sum(np.subtract(values[y + 1, x, :], K))
                sum_squared_neighbourhood += np.sum(np.power(np.subtract(values[y + 1, x, :], K), 2))
            if x > 0:
                n += 1
                sum_neighbourhood += np.sum(np.subtract(values[y, x - 1, :], K))
                sum_squared_neighbourhood += np.sum(np.power(np.subtract(values[y, x - 1, :], K), 2))
            if x < (w - 1):
                n += 1
                sum_neighbourhood += np.sum(np.subtract(values[y, x + 1, :], K))
                sum_squared_neighbourhood += np.sum(np.power(np.subtract(values[y, x + 1, :], K), 2))
            variance[y, x] = np.true_divide(sum_squared_neighbourhood - np.true_divide(np.power(sum_neighbourhood, 2), n), n)
    return variance

def main():
    image_A = misc.imread('A.png')[:,:,0:3]
    image_B = misc.imread('B.png')[:,:,0:3]
    h, w, dim = image_A.shape

    all_paths = generate_all_paths(-15, 15)

if __name__ == "__main__":
    main()