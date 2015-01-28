from numba import jit
import numpy as np
from bresenham import make_line

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
                counter += 1

    all_paths.resize((counter, 4))
    return all_paths

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

@jit('void(f8[:,:,:,:], i2, f8[:,:], f8[:,:], f8[:,:,:], f8[:,:,:], f8[:,:], f8[:,:], i4)',target='cpu', nopython=True)
def compute_matching_costs(all_costs, max_displacement, image_A, image_B, gradient_A, gradient_B, variance_A, variance_B, beta):
    h, w = image_A.shape
    for y in range(0, h):
        for x in range(0, w):
            for dy in range(-max_displacement, max_displacement + 1):
                for dx in range(-max_displacement, max_displacement + 1):
                    if ((y + dy) < h) and ((y + dy) >= 0) and ((x + dx) < w) and ((x + dx) >= 0):
                        all_costs[y, x, dy+max_displacement, dx+max_displacement] = matching_cost(y, x, y + dy, x + dx, image_A, image_B, gradient_A, gradient_B, variance_A, variance_B, beta)

@jit('void(f8[:,:],f8[:,:])',target='cpu', nopython=True)
def variance_4_neighbourhood(values, variance):
    # numerically stable variance by using shifted data
    h, w = values.shape
    for y in range(0, h):
        for x in range(0, w):
            K = values[y,x]
            n = 1
            sum_neighbourhood = 0
            sum_squared_neighbourhood = 0
            if y > 0:
                n += 1
                sum_neighbourhood += values[y - 1, x] - K
                sum_squared_neighbourhood += (values[y - 1, x] - K)**2
            if y < (h - 1):
                n += 1
                sum_neighbourhood += values[y + 1, x] - K
                sum_squared_neighbourhood += (values[y + 1, x] - K)**2
            if x > 0:
                n += 1
                sum_neighbourhood += values[y, x - 1] - K
                sum_squared_neighbourhood += (values[y, x - 1] - K)**2
            if x < (w - 1):
                n += 1
                sum_neighbourhood += values[y, x + 1] - K
                sum_squared_neighbourhood += (values[y, x + 1] - K)**2
            variance[y, x] = np.true_divide(sum_squared_neighbourhood - np.true_divide(np.power(sum_neighbourhood, 2), n), n)

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

@jit(target='cpu', nopython=True)
def compute_flow(flow, assigned_paths, all_paths):
    h, w, dim = flow.shape
    for y in range(0, h):
        for x in range(0, w):
            path_number = assigned_paths[y, x]
            flow[y, x, 0] = all_paths[path_number, 0] + all_paths[path_number, 2]
            flow[y, x, 1] = all_paths[path_number, 1] + all_paths[path_number, 3]