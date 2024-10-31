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
ang_L = 20  # Rotation angle in degrees (counter-clockwise positive)
ang_R = 20
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

# Define the outer corners for the rectangles using width (w)
p5_outer = np.array([p5[0] - w, p5[1]])  # Left top of coll_L
p6_outer = np.array([p6[0] - w, p6[1]])  # Left bottom of coll_L
p7_outer = np.array([p7[0] + w, p7[1]])  # Right top of coll_R
p8_outer = np.array([p8[0] + w, p8[1]])  # Right bottom of coll_R


# Function to calculate the center gravity of four points
def calculate_center_gravity(p1, p2, p3, p4):
    return (p1 + p2 + p3 + p4) / 4


# Calculate center gravity for left and right collimators using four points
cg_L = calculate_center_gravity(p5, p6, p5_outer, p6_outer)
cg_R = calculate_center_gravity(p7, p8, p7_outer, p8_outer)


# Function to rotate a point around a center by a given angle (in degrees)
def rotate_point(point, center, angle):
    angle_rad = np.radians(angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    # Translate point to origin
    translated_point = point - center

    # Rotate point
    rotated_x = translated_point[0] * cos_angle - translated_point[1] * sin_angle
    rotated_y = translated_point[0] * sin_angle + translated_point[1] * cos_angle

    # Translate back
    rotated_point = np.array([rotated_x, rotated_y]) + center
    return rotated_point


# Rotate the points around their respective center gravity
p5_rot = rotate_point(p5, cg_L, ang_L)
p6_rot = rotate_point(p6, cg_L, ang_L)
p5_outer_rot = rotate_point(p5_outer, cg_L, ang_L)
p6_outer_rot = rotate_point(p6_outer, cg_L, ang_L)

p7_rot = rotate_point(p7, cg_R, ang_R)
p8_rot = rotate_point(p8, cg_R, ang_R)
p7_outer_rot = rotate_point(p7_outer, cg_R, ang_R)
p8_outer_rot = rotate_point(p8_outer, cg_R, ang_R)


# Function to calculate intersection of two lines given by points (p1, p3) and (p2, p4)
def calculate_intersection(p1, p3, p2, p4):
    # Line 1: p1 -> p3
    a1 = p3[1] - p1[1]
    b1 = p1[0] - p3[0]
    c1 = a1 * p1[0] + b1 * p1[1]

    # Line 2: p2 -> p4
    a2 = p4[1] - p2[1]
    b2 = p2[0] - p4[0]
    c2 = a2 * p2[0] + b2 * p2[1]

    # Determinant
    det = a1 * b2 - a2 * b1
    if det == 0:
        # Lines are parallel
        return None
    else:
        # Calculate intersection point
        x = (b2 * c1 - b1 * c2) / det
        y = (a1 * c2 - a2 * c1) / det
        return np.array([x, y])


# Calculate the intersection (crossp)
crossp = calculate_intersection(p1, p3, p2, p4)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 10))

# Plot the points p1, p2, p3, p4
ax.scatter([p1[0], p2[0], p3[0], p4[0]], [p1[1], p2[1], p3[1], p4[1]], color='red', s=50, zorder=5)
# Plot the rotated points p5, p6, p7, p8
ax.scatter([p5_rot[0], p6_rot[0], p7_rot[0], p8_rot[0]], [p5_rot[1], p6_rot[1], p7_rot[1], p8_rot[1]], color='blue',
           s=50, zorder=5)

# Plot the center gravity points cg_L and cg_R (in purple)
ax.scatter(cg_L[0], cg_L[1], color='purple', s=100, zorder=5)
ax.annotate('cg_L', (cg_L[0], cg_L[1]), xytext=(5, 5), textcoords='offset points')
ax.scatter(cg_R[0], cg_R[1], color='purple', s=100, zorder=5)
ax.annotate('cg_R', (cg_R[0], cg_R[1]), xytext=(5, 5), textcoords='offset points')

# Plot the cross point if it exists
if crossp is not None:
    ax.scatter(crossp[0], crossp[1], color='green', s=100, zorder=5)
    ax.annotate('crossp', (crossp[0], crossp[1]), xytext=(5, 5), textcoords='offset points')

# Draw lines connecting p1 to p3, and p2 to p4
ax.plot([p1[0], p3[0]], [p1[1], p3[1]], color='magenta', linestyle='--', linewidth=1.5)
ax.plot([p2[0], p4[0]], [p2[1], p4[1]], color='cyan', linestyle='--', linewidth=1.5)

# Add the rotated rectangles for coll_L and coll_R
rect_L_rot = plt.Polygon([[p5_outer_rot[0], p5_outer_rot[1]], [p5_rot[0], p5_rot[1]], [p6_rot[0], p6_rot[1]],
                          [p6_outer_rot[0], p6_outer_rot[1]]],
                         closed=True, linewidth=1.5, edgecolor='black', facecolor='gray', alpha=0.5)
rect_R_rot = plt.Polygon(
    [[p7_rot[0], p7_rot[1]], [p7_outer_rot[0], p7_outer_rot[1]], [p8_outer_rot[0], p8_outer_rot[1]],
     [p8_rot[0], p8_rot[1]]],
    closed=True, linewidth=1.5, edgecolor='black', facecolor='gray', alpha=0.5)

ax.add_patch(rect_L_rot)
ax.add_patch(rect_R_rot)

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Show grid
ax.grid(True)

# Set equal aspect ratio
ax.set_aspect('equal', adjustable='box')

# Add title
plt.title('2D Coordinate System with Rotated Collimators and Cross Point')

# Show the plot
plt.show()