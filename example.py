import numpy as np
import matplotlib.pyplot as plt
from gco_python_master.pygco import cut_simple
import timeit
import precompute
import interpolation
import gradients
import occlusion
from cv2 import imread, imwrite

print("Initial setup...")
start = timeit.default_timer()
image_A = imread('A.png')[:,:,0].astype('float64')
image_B = imread('B.png')[:,:,0].astype('float64')
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
gradients.calculate_unary_costs(unaries, all_paths, single_path, all_costs, max_displacement, image_A, image_B, gradient_A, gradient_B, variance_A, variance_B, beta)
stop = timeit.default_timer()
print("[%.4fs]" % (stop - start))

print("Calculate pairwise costs...")
start = timeit.default_timer()
smooth_costs = np.zeros((number_paths, number_paths))
gradients.calculate_pairwise_costs(all_paths, delta, smooth_costs)
smooth_costs_int = (np.ceil(15*smooth_costs)).astype('int32')
stop = timeit.default_timer()
print("[%.4fs]" % (stop - start))

print("Check pairwise matrix is submodular...")
start = timeit.default_timer()
num_nonsubmodular = gradients.isSubmodular(smooth_costs_int)
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

print("Handling occlusions...")
start = timeit.default_timer()
occlusions = np.zeros((height, width, 2))
occlusion.find_occlusions(result, all_paths, occlusions)
imwrite('output/occlusions_forward.png', occlusions[:,:,0]*255)
imwrite('output/occlusions_backward.png', occlusions[:,:,1]*255)
stop = timeit.default_timer()
print("[%.4fs]" % (stop - start))

print("Interpolate images...")
start = timeit.default_timer()
frame = np.zeros_like(image_A)
for f in range(0, intermediate_frames):
    interpolation.generate_interpolated_frame(frame, image_A, image_B, result, all_paths, f, intermediate_frames)
    imwrite('output/%02d.png' % f, frame)
stop = timeit.default_timer()
print("[%.4fs]" % (stop - start))