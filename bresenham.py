import numpy as np

def make_line(y0, x0, y1, x1):
    is_steep = abs(y1 - y0) > abs(x1 - x0)
    if is_steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    if y0 > y1:
        y_step = -1
    else:
        y_step = 1

    line = np.zeros((max(abs(x1 - x0), abs(y1 - y0)) + 1, 2))
    currentPoint = 0

    err_increment = abs(y1 - y0)
    dx = x1 - x0

    err = dx >> 1
    y = y0

    print x0, y0, x1, y1, err, err_increment

    for x in range(x0, x1 + 1):
        if is_steep:
            line[currentPoint, :] = [x, y]
        else:
            line[currentPoint, :] = [y, x]
        err -= err_increment
        if err < 0:
            y += y_step
            err += dx
        currentPoint += 1

    return line