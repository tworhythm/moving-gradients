import matplotlib.pyplot as plt
import numpy as np
from gco_python_master.pygco import cut_simple
from scipy import misc, ndimage
from bresenham import make_line
import timeit
from numba import jit

@jit(target='cpu', nopython=True)
def isSubmodular(A):
    h, w = A.shape
    n = 0
    for i in range(0, h):
        for j in range(0, h):
            for k in range(0, h):
                if (A[j,k] + A[i,i]) > (A[j,i] + A[i,k]):
                    n += 1
    return n

def generate_all_paths(min_displacement, max_displacement):
    range_displacement = max_displacement - min_displacement + 1
    all_paths = np.zeros((range_displacement**3, 4), 'int16')

    counter = 0
    for dy in range(min_displacement, max_displacement + 1):
        for dx in range(min_displacement, max_displacement + 1):
            number_points = np.maximum(np.abs(dy), np.abs(dx)) + 1
            single_path = np.zeros((number_points, 2), dtype=np.int16)
            make_line(0, 0, dy, dx, single_path)
            if(not(single_path[0, 0] == 0 and single_path[0, 1] == 0)):
                np.flipud(single_path)
            for p in range(0, number_points):
                all_paths[counter, 0] = single_path[p, 0]
                all_paths[counter, 1] = single_path[p, 1]
                all_paths[counter, 2] = dy - single_path[p, 0]
                all_paths[counter, 3] = dx - single_path[p, 1]
                # print dy, dx, all_paths[counter, :]
                counter += 1

    all_paths.resize((counter, 4))
    return all_paths

def coherency_cost(path, neighbour_path, delta):
    d = np.sqrt((path[0] + path[2])**2 + (path[1] + path[3])**2)
    dn = np.sqrt((neighbour_path[0] + neighbour_path[2])**2 + (neighbour_path[1] + neighbour_path[3])**2)
    vA = path[0:2]/np.sqrt(path[0]**2 + path[1]**2)
    vAn = neighbour_path[0:2]/np.sqrt(neighbour_path[0]**2 + neighbour_path[1]**2)
    return min(np.sqrt(np.sum(np.power(np.subtract(d*vA, dn*vAn), 2))), delta)

@jit(target='cpu', nopython=True)
def compute_gradient(image, gradient):
    # compute spatial gradient of grayscale image of size h x w, output is h x w x 2
    h = image.shape[0]
    w = image.shape[1]

    for y in range(0, h):
        for x in range(0, w):
            if (y > 0) and (y < (h - 1)):
                dy = 0.5*(image[y+1,x] - image[y-1,x])
            elif (y > 0):
                dy = image[y,x] - image[y-1,x]
            elif (y < (h - 1)):
                dy = image[y+1,x] - image[y,x]
            if (x > 0) and (x < (w - 1)):
                dx = 0.5*(image[y,x+1] - image[y,x-1])
            elif (x > 0):
                dx = image[y,x] - image[y,x-1]
            elif (x < (w - 1)):
                dx = image[y,x+1] - image[y,x]
            gradient[y,x,0] = dy
            gradient[y,x,1] = dx

@jit('f8(i2, i2, i2, i2, f8[:,:], f8[:,:], f8[:,:,:], f8[:,:,:], i4[:,:], i4[:,:], i4)',target='cpu', nopython=True)
def matching_cost(Ay, Ax, By, Bx, image_A, image_B, gradient_A, gradient_B, variance_A, variance_B, beta):
    # beta is equivalent to alpha divided by 4, where alpha is the power used in the paper
    # examples use alpha = 8 => beta = 2
    color_diff = np.power(image_A[Ay, Ax] - image_B[By, Bx], 2)
    gradient_diff = (gradient_A[Ay, Ax, 0] - gradient_B[By, Bx, 0])**2 + (gradient_A[Ay, Ax, 1] - gradient_B[By, Bx, 1])**2
    q = gradient_diff + 0.5*color_diff

    var_A = variance_A[Ay, Ax]
    if var_A == 0:
        var_A = 1
    var_B = variance_B[By, Bx]
    if var_B == 0:
        var_B = 1

    return np.power(np.true_divide(np.power(q, 2), var_A * var_B), beta)

def variance_4_neighbourhood(values):
    # numerically stable variance by using shifted data
    h, w = values.shape
    variance = np.zeros((h,w), dtype='int32')
    for y in range(0, h):
        for x in range(0, w):
            K = values[y,x]
            n = 1
            sum_neighbourhood = 0
            sum_squared_neighbourhood = 0
            if y > 0:
                n += 1
                sum_neighbourhood += np.sum(np.subtract(values[y - 1, x], K))
                sum_squared_neighbourhood += np.sum(np.power(np.subtract(values[y - 1, x], K), 2))
            if y < (h - 1):
                n += 1
                sum_neighbourhood += np.sum(np.subtract(values[y + 1, x], K))
                sum_squared_neighbourhood += np.sum(np.power(np.subtract(values[y + 1, x], K), 2))
            if x > 0:
                n += 1
                sum_neighbourhood += np.sum(np.subtract(values[y, x - 1], K))
                sum_squared_neighbourhood += np.sum(np.power(np.subtract(values[y, x - 1], K), 2))
            if x < (w - 1):
                n += 1
                sum_neighbourhood += np.sum(np.subtract(values[y, x + 1], K))
                sum_squared_neighbourhood += np.sum(np.power(np.subtract(values[y, x + 1], K), 2))
            variance[y, x] = np.true_divide(sum_squared_neighbourhood - np.true_divide(np.power(sum_neighbourhood, 2), n), n)
    return variance

@jit(target='cpu', nopython=True)
def in_bounds(y, x, h, w):
    if (y >= 0) and (y < h) and (x >= 0) and (x < w):
        return 1
    else:
        return 0

@jit('void(f8[:,:,:,:], i2, f8[:,:], f8[:,:], f8[:,:,:], f8[:,:,:], i4[:,:], i4[:,:], i4)',target='cpu', nopython=True)
def compute_matching_costs(all_costs, max_displacement, image_A, image_B, gradient_A, gradient_B, variance_A, variance_B, beta):
    h, w = image_A.shape
    for y in range(0, h):
        for x in range(0, w):
            for dy in range(-max_displacement, max_displacement + 1):
                for dx in range(-max_displacement, max_displacement + 1):
                    if ((y + dy) < h) and ((y + dy) >= 0) and ((x + dx) < w) and ((x + dx) >= 0):
                        all_costs[y, x, dy+max_displacement, dx+max_displacement] = matching_cost(y, x, y + dy, x + dx, image_A, image_B, gradient_A, gradient_B, variance_A, variance_B, beta)


@jit(target='cpu', nopython=True)
def calculate_unary_costs(unary_cost_matrix, paths, single_path, all_costs, max_displacement, image_A, image_B, gradient_A, gradient_B, variance_A, variance_B, beta):
    # single_path is an array large enough to hold any single path since no array creation is possible inside this function due to numba
    h, w = image_A.shape
    number_paths = paths.shape[0]
    for y in range(0, h):
        for x in range(0, w):
            for path_number in range(0, number_paths):
                path = paths[path_number, :]
                pAy = y + path[0]
                pAx = x + path[1]
                pBy = y - path[2]
                pBx = x - path[3]
                if (in_bounds(pAy, pAx, h, w) == 1) and (in_bounds(pBy, pBx, h, w) == 1):
                    unary_cost_matrix[y, x, path_number] = min(100000000000000000*(all_costs[pAy, pAx, pBy - pAy + max_displacement, pBx - pAx + max_displacement]), 10000000)
                else:
                    unary_cost_matrix[y, x, path_number] = 10000000

@jit('void(i2[:,:], i4, f8[:,:])', target='cpu', nopython=True)
def calculate_pairwise_costs(all_paths, delta, smooth_costs):
    number_paths = all_paths.shape[0]
    for i in range(0, number_paths):
        for j in range(0, i):
            smooth_costs[i,j] = min(np.sqrt(np.power((all_paths[i,0] + all_paths[i,2]) - (all_paths[j,0] + all_paths[j,2]),2) + np.power((all_paths[i,1] + all_paths[i,3]) - (all_paths[j,1] + all_paths[j,3]),2)), delta)
            smooth_costs[j,i] = smooth_costs[i,j]

# def main():
print("Initial setup...")
start = timeit.default_timer()
image_A = misc.imread('A.png')[:,:,0].astype('float64') / 255.0
image_B = misc.imread('B.png')[:,:,0].astype('float64') / 255.0
h, w = image_A.shape
beta = np.int32(2)
delta = np.int32(20) # see paper for other values used
max_displacement = np.int32(4)
intermediate_frames = 18
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
gradient_A= np.zeros((h, w, 2))
compute_gradient(image_A, gradient_A)
gradient_B= np.zeros((h, w, 2))
compute_gradient(image_B, gradient_B)
stop = timeit.default_timer()
print("[%.4fs]" % (stop - start))

print("Generating paths...")
start = timeit.default_timer()
all_paths = generate_all_paths(-max_displacement, max_displacement)
stop = timeit.default_timer()
print("[%.4fs]" % (stop - start))

number_paths = all_paths.shape[0]

print("Calculate unary costs...")
start = timeit.default_timer()
unaries = np.zeros((h, w, number_paths))
single_path = np.zeros((max_displacement+1,2), dtype=np.int16)
all_costs = np.ones((h, w, 2*max_displacement+1, 2*max_displacement+1))
compute_matching_costs(all_costs, max_displacement, image_A, image_B, gradient_A, gradient_B, variance_A, variance_B, beta)
calculate_unary_costs(unaries, all_paths, single_path, all_costs, max_displacement, image_A, image_B, gradient_A, gradient_B, variance_A, variance_B, beta)
stop = timeit.default_timer()
print("[%.4fs]" % (stop - start))

print("Calculate pairwise costs...")
start = timeit.default_timer()
smooth_costs = np.zeros((number_paths, number_paths))
calculate_pairwise_costs(all_paths, delta, smooth_costs)
smooth_costs_int = (np.ceil(10*smooth_costs)).astype('int32')
stop = timeit.default_timer()
print("[%.4fs]" % (stop - start))

print("Check pairwise matrix is submodular...")
start = timeit.default_timer()
num_nonsubmodular = isSubmodular(smooth_costs_int)
if num_nonsubmodular == 0:
    print "True"
else:
    print "False. %n combinations fail" % num_nonsubmodular
stop = timeit.default_timer()
print("[%.4fs]" % (stop - start))

print("Assigning labels...")
start = timeit.default_timer()
unaries_int = unaries.astype('int32')
result = cut_simple(unaries_int, smooth_costs_int)
stop = timeit.default_timer()
print("[%.4fs]" % (stop - start))

# if __name__ == "__main__":
#     main()