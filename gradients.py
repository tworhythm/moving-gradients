import matplotlib.pyplot as plt
import numpy as np
from gco_python_master.pygco import cut_simple
from scipy import misc, ndimage
from bresenham import make_line
import timeit
from numba import jit

def generate_all_paths(min_displacement, max_displacement):
    range_displacement = max_displacement - min_displacement + 1
    all_paths = np.zeros((range_displacement*range_displacement, 2), 'int16')

    counter = 0
    for dy in range(min_displacement, max_displacement + 1):
        for dx in range(min_displacement, max_displacement + 1):
            all_paths[counter, 0] = dy
            all_paths[counter, 1] = dx
            counter += 1
    return all_paths

def coherency_cost(path, neighbour_path, delta):
    return min(np.sqrt(np.sum(np.power(np.subtract(path, neighbour_path), 2))), delta)

@jit('f8(i2, i2, i2, i2, i4[:,:,:], i4[:,:,:], f8[:,:,:], f8[:,:,:], i4[:,:], i4[:,:], i4)',target='cpu', nopython=True)
def matching_cost(Ay, Ax, By, Bx, image_A, image_B, gradient_A, gradient_B, variance_A, variance_B, beta):
    # beta is equivalent to alpha divided by 4, where alpha is the power used in the paper
    # examples use alpha = 8 => beta = 2
    color_diff = np.power(image_A[Ay, Ax, 0] - image_B[By, Bx, 0], 2) + np.power(image_A[Ay, Ax, 1] - image_B[By, Bx, 1], 2) + np.power(image_A[Ay, Ax, 2] - image_B[By, Bx, 2], 2)
    gradient_diff = np.power(gradient_A[Ay, Ax, 0] - gradient_B[By, Bx, 0], 2) + np.power(gradient_A[Ay, Ax, 1] - gradient_B[By, Bx, 1], 2) + np.power(gradient_A[Ay, Ax, 2] - gradient_B[By, Bx, 2], 2)
    q = gradient_diff + 0.5*color_diff

    # Add 1 to the variance to avoid dividing by zero - any neighborhood with no gradient is hence not normalized (I don't think this should make a big difference)
    return np.power(np.true_divide(np.power(q, 2), (variance_A[Ay, Ax] + 1) * (variance_B[By, Bx] + 1)), beta)

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

@jit('void(f8[:,:], i2, i4[:,:,:], i4[:,:,:], f8[:,:,:], f8[:,:,:], i4[:,:], i4[:,:], i4)',target='cpu', nopython=True)
def compute_matching_costs(all_costs, max_displacement, image_A, image_B, gradient_A, gradient_B, variance_A, variance_B, beta):
    h, w, dim = image_A.shape
    for y in range(0, h):
        for x in range(0, w):
            for dy in range(-max_displacement, max_displacement + 1):
                for dx in range(-max_displacement, max_displacement + 1):
                    if ((y + dy) < h) and ((y + dy) >= 0) and ((x + dx) < w) and ((x + dx) >= 0):
                        all_costs[y*h + x, (dy+max_displacement) * (2*max_displacement+1) + (dx+max_displacement)] = matching_cost(y, x, y + dy, x + dx, image_A, image_B, gradient_A, gradient_B, variance_A, variance_B, beta)


@jit(target='cpu', nopython=True)
def calculate_unary_costs(unary_cost_matrix, paths, single_path, all_costs, max_displacement, image_A, image_B, gradient_A, gradient_B, variance_A, variance_B, beta):
    # single_path is an array large enough to hold any single path since no array creation is possible inside this function due to numba
    h, w, dim = image_A.shape
    number_paths = paths.shape[0]
    for y in range(0, h):
        for x in range(0, w):
            for path_number in range(0, number_paths):
                path = paths[path_number, :]
                best_correspondence = np.inf
                best_transition_point_y = y
                best_transition_point_x = x
                dy, dx = path

                if ((y + dy) < h) and ((y + dy) >= 0) and ((x + dx) < w) and ((x + dx) >= 0):
                    make_line(y, x, y + dy, x + dx, single_path)
                    number_points = np.maximum(np.abs(dy), np.abs(dx)) + 1
                    for p in range(0, number_points):
                        dy2 = single_path[p, 0] - y
                        dx2 = single_path[p, 1] - x
                        current_correspondence = all_costs[y*h + x, (dy2 + max_displacement)*(2*max_displacement + 1) + (dx2 + max_displacement)]
                        if current_correspondence < best_correspondence:
                            best_correspondence = current_correspondence
                            best_transition_point_y = single_path[p, 0]
                            best_transition_point_x = single_path[p, 1]
                    unary_cost_matrix[y*h + x, path_number, 0] = best_correspondence
                    unary_cost_matrix[y*h + x, path_number, 1] = best_transition_point_y
                    unary_cost_matrix[y*h + x, path_number, 2] = best_transition_point_x
                else:
                    unary_cost_matrix[y*h + x, path_number, 0] = np.inf
                    unary_cost_matrix[y*h + x, path_number, 1] = y
                    unary_cost_matrix[y*h + x, path_number, 2] = x

@jit('void(i2[:,:], i4, f8[:,:])', target='cpu', nopython=True)
def calculate_pairwise_costs(all_paths, delta, smooth_costs):
    number_paths = all_paths.shape[0]
    for i in range(0, number_paths):
        for j in range(0, i):
            smooth_costs[i,j] = min(np.sqrt(np.power(all_paths[i,0] - all_paths[j,0],2) + np.power(all_paths[i,1] - all_paths[j,1],2)), delta)
            smooth_costs[j,i] = smooth_costs[i,j]

# def main():
print("Initial setup...")
start = timeit.default_timer()
image_A = misc.imread('A.png')[:,:,0:3].astype('int32')
image_B = misc.imread('B.png')[:,:,0:3].astype('int32')
h, w, dim = image_A.shape
beta = np.int32(2)
delta = np.int32(20) # see paper for other values used
max_displacement = np.int32(30)
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
all_paths = generate_all_paths(-max_displacement, max_displacement)
stop = timeit.default_timer()
print("[%.4fs]" % (stop - start))

number_pixels = h*w
number_paths = all_paths.shape[0]

print("Calculate unary costs...")
start = timeit.default_timer()
unaries = np.zeros((number_pixels, number_paths, 3))
single_path = np.zeros((max_displacement+1,2), dtype=np.int16)
all_costs = np.zeros((h*w, (2*max_displacement+1)*(2*max_displacement+1)))
compute_matching_costs(all_costs, max_displacement, image_A, image_B, gradient_A, gradient_B, variance_A, variance_B, beta)
calculate_unary_costs(unaries, all_paths, single_path, all_costs, max_displacement, image_A, image_B, gradient_A, gradient_B, variance_A, variance_B, beta)
stop = timeit.default_timer()
print("[%.4fs]" % (stop - start))

print("Calculate pairwise costs...")
start = timeit.default_timer()
smooth_costs = np.zeros((number_paths, number_paths))
calculate_pairwise_costs(all_paths, delta, smooth_costs)
stop = timeit.default_timer()
print("[%.4fs]" % (stop - start))

# py = 15
# px = 15
# pind = py*64 + px
# print("Pixel is (%d, %d)" % (py, px))
# print("Index is %d" % (pind))
# #find best correspondence
# bestcost = np.inf
# trans_y = 0
# trans_x = 0
# path_index = -1
# for i in range(0, unaries.shape[1]):
#     if unaries[pind, i, 0] < bestcost:
#         bestcost = unaries[pind, i, 0]
#         trans_y = unaries[pind, i, 1]
#         trans_x = unaries[pind, i, 2]
#         path_index = i
# path_dy = all_paths[path_index, 0]
# path_dx = all_paths[path_index, 1]

# print("Best correspondence score is %.4f, path index %d" % (bestcost, path_index))
# print("The path has displacement (%d, %d), and transition point (%d, %d)" % (path_dy, path_dx, trans_y, trans_x))
# trans_correspondence = matching_cost(py, px, trans_y, trans_x, image_A, image_B, gradient_A, gradient_B, variance_A, variance_B, beta)
# print("Actual correspondence is %.4f" % trans_correspondence)

# if __name__ == "__main__":
#     main()