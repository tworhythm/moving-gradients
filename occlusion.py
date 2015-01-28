import numpy as np
import precompute
from numba import jit

def find_occlusions(assigned_paths, all_paths, occlusions):
    h, w = assigned_paths.shape
    forward_flow = np.zeros((h, w, 2))
    precompute.compute_flow(forward_flow, assigned_paths, all_paths)
    forward_flow = - forward_flow
    backward_flow = np.zeros((h, w, 2))
    precompute.compute_flow(backward_flow, assigned_paths, all_paths)
    forward_flow_verification(forward_flow, backward_flow, occlusions)

@jit(target='cpu', nopython=True)
def forward_flow_verification(forward_flow, backward_flow, occlusions):
    h, w = occlusions.shape
    for y in range(0, h):
        for x in range(0, w):
            vAy = forward_flow[y,x,0]
            vAx = forward_flow[y,x,1]
            vBy = backward_flow[y + vAy, x + vAx, 0]
            vBx = backward_flow[y + vAy, x + vAx, 1]

            if vAy == -vBy and vAx == -vBx:
                occlusions[y,x] = 0
            else:
                occlusions[y,x] = 1