import matplotlib.pyplot as plt
import numpy as np
from gco_python_master.pygco import cut_simple
from scipy import misc, ndimage
import timeit
from numba import jit

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

def coherency_cost(path, neighbour_path, delta):
    return min(np.sqrt(np.sum(np.power(np.subtract(path, neighbour_path), 2))), delta)

def matching_cost(pA, pB, image_A, image_B, gradient_A, gradient_B, variance_A, variance_B, beta):
    Ay, Ax = pA
    By, Bx = pB
    color_diff = np.sum(np.power(np.subtract(image_A[Ay, Ax, :].astype('int32'), image_B[By, Bx, :].astype('int32')), 2))
    gradient_diff = np.sum(np.power(np.subtract(gradient_A[Ay, Ax, :].astype('int32'), gradient_B[By, Bx, :].astype('int32')), 2))
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

@jit(target='cpu', nopython=True)
def calculate_unary_costs(unary_cost_matrix, paths, image_A, image_B, gradient_A, gradient_B, variance_A, variance_B):
    h, w, dim = image_A.shape
    number_paths = paths.shape[0]
    for y in range(0, h):
        for x in range(0, w):
            for path_number in range(0, number_paths):
                path = paths[path_number, :]
                best_correspondence = 0
                dy, dx = path
                for point in range(0, 20):
                    True

def main():
    print("Initial setup...")
    start = timeit.default_timer()
    image_A = misc.imread('A.png')[:,:,0:3]
    image_B = misc.imread('B.png')[:,:,0:3]
    h, w, dim = image_A.shape
    beta = 2
    stop = timeit.default_timer()
    print("[%.4fs]" % (stop - start))


    print("Calculating local variance...")
    start = timeit.default_timer()
    variance_A = variance_4_neighbourhood(image_A)
    variance_B = variance_4_neighbourhood(image_B)
    stop = timeit.default_timer()
    print("[%.4fs]" % (stop - start))

    print("Calculating image gradients...")
    start = timeit.default_timer()
    gradient_A_y, gradient_A_x, gradient_A_c = np.gradient(image_A)
    gradient_A = np.concatenate((gradient_A_y, gradient_A_x, gradient_A_c), axis=2)
    gradient_B_y, gradient_B_x, gradient_B_c = np.gradient(image_B)
    gradient_B = np.concatenate((gradient_B_y, gradient_B_x, gradient_B_c), axis=2)
    stop = timeit.default_timer()
    print("[%.4fs]" % (stop - start))

    print("Generating paths...")
    start = timeit.default_timer()
    all_paths = generate_all_paths(-30, 30)
    stop = timeit.default_timer()
    print("[%.4fs]" % (stop - start))

    number_pixels = h*w
    number_paths = all_paths.shape[0]

    print("Calculate unary costs...")
    start = timeit.default_timer()
    unaries = np.zeros((number_pixels, number_paths))
    calculate_unary_costs(unaries, all_paths, image_A, image_B, gradient_A, gradient_B, variance_A, variance_B)
    stop = timeit.default_timer()
    print("[%.4fs]" % (stop - start))

if __name__ == "__main__":
    main()