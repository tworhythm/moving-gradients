import numpy as np
from numba import jit

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