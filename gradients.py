import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from gco_python_master.pygco import cut_simple
from scipy import misc, ndimage
import timeit
import precompute
import interpolation

@jit('i4(i4[:,:])', target='cpu', nopython=True)
def isSubmodular(A):
    h, w = A.shape
    n = 0
    for i in range(0, h):
        for j in range(0, h):
            for k in range(0, h):
                if (A[j,k] + A[i,i]) > (A[j,i] + A[i,k]):
                    n += 1
    return n

@jit('i4(i4, i4, i4, i4)', target='cpu', nopython=True)
def in_bounds(y, x, h, w):
    if (y >= 0) and (y < h) and (x >= 0) and (x < w):
        return 1
    else:
        return 0

@jit('void(f8[:,:,:], i2[:,:], i2[:,:], f8[:,:,:,:], i2, f8[:,:], f8[:,:], f8[:,:,:], f8[:,:,:], f8[:,:], f8[:,:], i4)', target='cpu', nopython=True)
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
image_A = misc.imread('A.png')[:,:,0].astype('float64')
image_B = misc.imread('B.png')[:,:,0].astype('float64')
height, width = image_A.shape
beta = np.int32(2)
delta = np.int32(20) # see paper for other values used
max_displacement = np.int32(4)
intermediate_frames = 18
stop = timeit.default_timer()
print("[%.4fs]" % (stop - start))


print("Calculating local variance...")
start = timeit.default_timer()
variance_A, variance_B = np.zeros((height,width), dtype=np.float64), np.zeros((height,width), dtype=np.float64)
precompute.variance_4_neighbourhood(image_A, variance_A)
precompute.variance_4_neighbourhood(image_B, variance_B)
stop = timeit.default_timer()
print("[%.4fs]" % (stop - start))

print("Calculating image gradients...")
start = timeit.default_timer()
gradient_A, gradient_B = np.zeros((height, width, 2)), np.zeros((height, width, 2))
precompute.compute_gradient(image_A, gradient_A)
precompute.compute_gradient(image_B, gradient_B)
stop = timeit.default_timer()
print("[%.4fs]" % (stop - start))

print("Generating paths...")
start = timeit.default_timer()
all_paths = precompute.generate_all_paths(-max_displacement, max_displacement)
stop = timeit.default_timer()
print("[%.4fs]" % (stop - start))

number_paths = all_paths.shape[0]

print("Calculate unary costs...")
start = timeit.default_timer()
unaries = np.zeros((height, width, number_paths))
single_path = np.zeros((max_displacement+1,2), dtype=np.int16)
all_costs = np.ones((height, width, 2*max_displacement+1, 2*max_displacement+1))
precompute.compute_matching_costs(all_costs, max_displacement, image_A, image_B, gradient_A, gradient_B, variance_A, variance_B, beta)
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

print("Interpolate images...")
start = timeit.default_timer()
frame = np.zeros_like(image_A)
for f in range(0, intermediate_frames):
    interpolation.generate_interpolated_frame(frame, image_A, image_B, result, all_paths, f, intermediate_frames)
    misc.imsave('output/%02d.png' % f, frame)
stop = timeit.default_timer()
print("[%.4fs]" % (stop - start))

# if __name__ == "__main__":
#     main()