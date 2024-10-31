import numpy as np
import matplotlib.pyplot as plt

# Fixed constants
r = 23.75
pitch = 1.2
depth_target_top = 40
depth_target_bottom = 50
offset_detector_to_coll = 5
offset_coll_to_obj = 5
w = 2  # Width of the collimators

# Parameters that want to be optimized (initial values)
h_L = 50
h_R = 50
ang_L = 0
ang_R = 0
det_y = offset_coll_to_obj + h_L + offset_detector_to_coll

# Fixed points
p3 = np.array([0, - depth_target_top])
p4 = np.array([0, - depth_target_bottom])

# Induced points based on initial guess
p1 = np.array([-r - (pitch / 2), det_y])
p2 = np.array([-r + (pitch / 2), det_y])

# Define points p5, p6, p7, p8 based on p1, p2 and collimator heights and offsets
p5 = np.array([p1[0], p1[1] - offset_detector_to_coll])
p6 = np.array([p1[0], p1[1] - offset_detector_to_coll - h_L])
p7 = np.array([p2[0], p2[1] - offset_detector_to_coll])
p8 = np.array([p2[0], p2[1] - offset_detector_to_coll - h_R])

# Create the plot
fig, ax = plt.subplots(figsize=(10, 10))

# Plot the points p1, p2, p3, p4
ax.scatter([p1[0], p2[0], p3[0], p4[0]], [p1[1], p2[1], p3[1], p4[1]], color='red', s=50, zorder=5)
# Plot the points p5, p6, p7, p8
ax.scatter([p5[0], p6[0], p7[0], p8[0]], [p5[1], p6[1], p7[1], p8[1]], color='blue', s=50, zorder=5)

# Label the points
ax.annotate('p1', (p1[0], p1[1]), xytext=(5, 5), textcoords='offset points')
ax.annotate('p2', (p2[0], p2[1]), xytext=(5, 5), textcoords='offset points')
ax.annotate('p3', (p3[0], p3[1]), xytext=(5, 5), textcoords='offset points')
ax.annotate('p4', (p4[0], p4[1]), xytext=(5, 5), textcoords='offset points')
ax.annotate('p5', (p5[0], p5[1]), xytext=(5, 5), textcoords='offset points')
ax.annotate('p6', (p6[0], p6[1]), xytext=(5, 5), textcoords='offset points')
ax.annotate('p7', (p7[0], p7[1]), xytext=(5, 5), textcoords='offset points')
ax.annotate('p8', (p8[0], p8[1]), xytext=(5, 5), textcoords='offset points')

# Draw lines connecting p1 to p3, and p2 to p4
ax.plot([p1[0], p3[0]], [p1[1], p3[1]], color='magenta', linestyle='--', linewidth=1.5)
ax.plot([p2[0], p4[0]], [p2[1], p4[1]], color='cyan', linestyle='--', linewidth=1.5)

# Add the rectangles for coll_L (left collimator) and coll_R (right collimator)
rect_L = plt.Rectangle((p5[0] - w, p6[1]), w, h_L, linewidth=1.5, edgecolor='black', facecolor='gray', alpha=0.5)
rect_R = plt.Rectangle((p7[0], p8[1]), w, h_R, linewidth=1.5, edgecolor='black', facecolor='gray', alpha=0.5)
ax.add_patch(rect_L)
ax.add_patch(rect_R)

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Show grid
ax.grid(True)

# Set equal aspect ratio
ax.set_aspect('equal', adjustable='box')

# Add title
plt.title('2D Coordinate System with Collimators')

# Show the plot
plt.show()
