import numpy as np
from numba import jit

@jit('void(i2, i2, i2, i2, i2[:, :])', target='cpu', nopython=True)
def make_line(y0, x0, y1, x1, line):
    is_steep = np.abs(y1 - y0) > np.abs(x1 - x0)
    if is_steep:
        tmp = x0
        x0 = y0
        y0 = tmp

        tmp = x1
        x1 = y1
        y1 = tmp

    if x0 > x1:
        tmp = x0
        x0 = x1
        x1 = tmp

        tmp = y0
        y0 = y1
        y1 = tmp

    if y0 > y1:
        y_step = -1
    else:
        y_step = 1

    currentPoint = 0

    err_increment = np.abs(y1 - y0)
    dx = x1 - x0

    err = dx >> 1
    y = y0

    for x in range(x0, x1 + 1):
        if is_steep:
            line[currentPoint, 0] = x
            line[currentPoint, 1] = y
        else:
            line[currentPoint, 0] = y
            line[currentPoint, 1] = x
        err -= err_increment
        if err < 0:
            y += y_step
            err += dx
        currentPoint += 1