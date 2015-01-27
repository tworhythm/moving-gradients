from numba import jit
import numpy as np

@jit('f8(f8[:,:], i4, i4)',target='cpu', nopython=True)
def bilinear_interp(image, y, x):
    xmin = int(np.floor(x))
    xmax = int(np.ceil(x))
    ymin = int(np.floor(y))
    ymax = int(np.ceil(y))

    b1 = image[ymin, xmin]
    b2 = image[ymin, xmax] - image[ymin, xmin]
    b3 = image[ymax, xmin] - image[ymin, xmin]
    b4 = image[ymin, xmin] - image[ymin, xmax] - image[ymax, xmin] + image[ymax, xmax]

    return b1 + b2*(x - xmin) + b3*(y - ymin) + b4*(x - xmin)*(y - ymin)

@jit('void(f8[:,:], f8[:,:], f8[:,:], i4[:,:], i2[:,:], i4, i4)',target='cpu', nopython=True)
def generate_interpolated_frame(frame, image_A, image_B, result, all_paths, f, intermediate_frames):
    h, w = image_A.shape
    for y in range(0, h):
        for x in range(0, w):
            path = all_paths[result[y,x], :]
            path_A_length = np.sqrt(np.power(path[0], 2) + np.power(path[1], 2))
            path_B_length = np.sqrt(np.power(path[2], 2) + np.power(path[3], 2))
            full_path_length = path_A_length + path_B_length
            fraction_of_full_path = np.true_divide(f, intermediate_frames - 1)

            if full_path_length <= np.spacing(1):
                frame[y, x] = image_A[y,x]
            else:
                if full_path_length == path_A_length:
                    transition_point = 0
                else:
                    transition_point = np.true_divide(path_A_length, full_path_length)
                if path_A_length == 0 or transition_point > fraction_of_full_path:
                    image = image_B
                    y_interp = y - np.true_divide((1 - fraction_of_full_path) * full_path_length, path_B_length) * path[2]
                    x_interp = x - np.true_divide((1 - fraction_of_full_path) * full_path_length, path_B_length) * path[3]
                else:
                    image = image_A
                    y_interp = y + np.true_divide(fraction_of_full_path * full_path_length, path_A_length) * path[0]
                    x_interp = x + np.true_divide(fraction_of_full_path * full_path_length, path_A_length) * path[1]
                frame[y, x] = bilinear_interp(image, y_interp, x_interp)