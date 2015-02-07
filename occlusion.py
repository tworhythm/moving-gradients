import numpy as np
import precompute
from numba import jit
from scipy.ndimage import label

def find_occlusions(assigned_paths, all_paths, occlusions):
    h, w = assigned_paths.shape
    forward_flow = np.zeros((h, w, 2))
    precompute.compute_forward_flow(forward_flow, assigned_paths, all_paths)
    backward_flow = np.zeros((h, w, 2))
    precompute.compute_backward_flow(backward_flow, assigned_paths, all_paths)
    forward_flow_verification(forward_flow, backward_flow, occlusions)
    backward_flow_verification(forward_flow, backward_flow, occlusions)

    occlusion_regions, num_regions = label(occlusions[:,:,0])
    print num_regions

@jit(target='cpu', nopython=True)
def forward_flow_verification(forward_flow, backward_flow, occlusions):
    h, w, dim = occlusions.shape
    for y in range(0, h):
        for x in range(0, w):
            vAy = forward_flow[y,x,0]
            vAx = forward_flow[y,x,1]
            vBy = backward_flow[y + vAy, x + vAx, 0]
            vBx = backward_flow[y + vAy, x + vAx, 1]
            if y == 20 and x == 20:
                print vAy, vAx, vBy, vBx
            if vAy == -vBy and vAx == -vBx:
                occlusions[y,x,0] = 0
            else:
                occlusions[y,x,0] = 1

@jit(target='cpu', nopython=True)
def backward_flow_verification(forward_flow, backward_flow, occlusions):
    h, w, dim = occlusions.shape
    for y in range(0, h):
        for x in range(0, w):
            vBy = backward_flow[y,x,0]
            vBx = backward_flow[y,x,1]
            vAy = forward_flow[y + vBy, x + vBx, 0]
            vAx = forward_flow[y + vBy, x + vBx, 1]

            if vAy == -vBy and vAx == -vBx:
                occlusions[y,x,1] = 0
            else:
                occlusions[y,x,1] = 1