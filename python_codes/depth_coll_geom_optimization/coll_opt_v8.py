import numpy as np
from scipy.optimize import minimize
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
p3 = np.array([0, -depth_target_top])
p4 = np.array([0, -depth_target_bottom])

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

# Function to calculate the intersection of two lines
def calculate_intersection(p1, p3, p2, p4):
    a1 = p3[1] - p1[1]
    b1 = p1[0] - p3[0]
    c1 = a1 * p1[0] + b1 * p1[1]

    a2 = p4[1] - p2[1]
    b2 = p2[0] - p4[0]
    c2 = a2 * p2[0] + b2 * p2[1]

    det = a1 * b2 - a2 * b1
    if det == 0:
        return None  # Lines are parallel
    else:
        x = (b2 * c1 - b1 * c2) / det
        y = (a1 * c2 - a2 * c1) / det
        return np.array([x, y])


# Function to rotate a point around a center by a given angle (in degrees)
def rotate_point(point, center, angle):
    angle_rad = np.radians(angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    translated_point = point - center
    rotated_x = translated_point[0] * cos_angle - translated_point[1] * sin_angle
    rotated_y = translated_point[0] * sin_angle + translated_point[1] * cos_angle
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


# Calculate the intersection (crossp)
crossp = calculate_intersection(p1, p3, p2, p4)

# Create the plot
fig1, ax1 = plt.subplots(figsize=(10, 10))

# Plot the points p1, p2, p3, p4
ax1.scatter([p1[0], p2[0], p3[0], p4[0]], [p1[1], p2[1], p3[1], p4[1]], color='red', s=50, zorder=5)
# Plot the rotated points p5, p6, p7, p8
ax1.scatter([p5_rot[0], p6_rot[0], p7_rot[0], p8_rot[0]], [p5_rot[1], p6_rot[1], p7_rot[1], p8_rot[1]], color='blue',
           s=50, zorder=5)

# Plot the center gravity points cg_L and cg_R (in purple)
ax1.scatter(cg_L[0], cg_L[1], color='purple', s=100, zorder=5)
ax1.annotate('cg_L', (cg_L[0], cg_L[1]), xytext=(5, 5), textcoords='offset points')
ax1.scatter(cg_R[0], cg_R[1], color='purple', s=100, zorder=5)
ax1.annotate('cg_R', (cg_R[0], cg_R[1]), xytext=(5, 5), textcoords='offset points')

# Plot the cross point if it exists
if crossp is not None:
    ax1.scatter(crossp[0], crossp[1], color='green', s=100, zorder=5)
    ax1.annotate('crossp', (crossp[0], crossp[1]), xytext=(5, 5), textcoords='offset points')

# Draw lines connecting p1 to p3, and p2 to p4
ax1.plot([p1[0], p3[0]], [p1[1], p3[1]], color='magenta', linestyle='--', linewidth=1.5)
ax1.plot([p2[0], p4[0]], [p2[1], p4[1]], color='cyan', linestyle='--', linewidth=1.5)

# Add the rotated rectangles for coll_L and coll_R
rect_L_rot = plt.Polygon([[p5_outer_rot[0], p5_outer_rot[1]], [p5_rot[0], p5_rot[1]], [p6_rot[0], p6_rot[1]],
                          [p6_outer_rot[0], p6_outer_rot[1]]],
                         closed=True, linewidth=1.5, edgecolor='black', facecolor='gray', alpha=0.5)
rect_R_rot = plt.Polygon(
    [[p7_rot[0], p7_rot[1]], [p7_outer_rot[0], p7_outer_rot[1]], [p8_outer_rot[0], p8_outer_rot[1]],
     [p8_rot[0], p8_rot[1]]],
    closed=True, linewidth=1.5, edgecolor='black', facecolor='gray', alpha=0.5)

ax1.add_patch(rect_L_rot)
ax1.add_patch(rect_R_rot)

# Set axis labels
ax1.set_xlabel('X')
ax1.set_ylabel('Y')

# Show grid
ax1.grid(True)

# Set equal aspect ratio
ax1.set_aspect('equal', adjustable='box')

# Add title
plt.title('2D Coordinate System with Rotated Collimators and Cross Point')


# Objective function for optimization
def objective(params):
    h_L, ang_L, h_R, ang_R, det_y = params

    # Calculate points p1, p2
    p1 = np.array([-r - (pitch / 2), det_y])
    p2 = np.array([-r + (pitch / 2), det_y])

    # Calculate p5, p6, p7, p8
    p5 = np.array([p1[0], p1[1] - offset_detector_to_coll])
    p6 = np.array([p1[0], p1[1] - offset_detector_to_coll - h_L])
    p7 = np.array([p2[0], p2[1] - offset_detector_to_coll])
    p8 = np.array([p2[0], p2[1] - offset_detector_to_coll - h_R])

    # Calculate center of gravity
    cg_L = (p5 + p6) / 2
    cg_R = (p7 + p8) / 2

    # Rotate points around center gravity
    p6_rot = rotate_point(p6, cg_L, ang_L)
    p8_rot = rotate_point(p8, cg_R, ang_R)

    # Calculate crossp (intersection of p1-p3 and p2-p4)
    crossp = calculate_intersection(p1, p3, p2, p4)

    # Minimize deviation of p6 and p8 from lines p1-p3 and p2-p4
    line1_dist = np.abs(np.cross(p3 - p1, p6_rot - p1)) / np.linalg.norm(p3 - p1)  # p6 on line p1-p3
    line2_dist = np.abs(np.cross(p4 - p2, p8_rot - p2)) / np.linalg.norm(p4 - p2)  # p8 on line p2-p4

    return line1_dist + line2_dist  # Minimize both deviations


# Constraints
def constraint1(params):
    h_L, ang_L, h_R, ang_R, det_y = params

    # Calculate points p1, p2
    p1 = np.array([-r - (pitch / 2), det_y])
    p2 = np.array([-r + (pitch / 2), det_y])

    # Calculate p5, p6, p7, p8
    p5 = np.array([p1[0], p1[1] - offset_detector_to_coll])
    p6 = np.array([p1[0], p1[1] - offset_detector_to_coll - h_L])
    p7 = np.array([p2[0], p2[1] - offset_detector_to_coll])
    p8 = np.array([p2[0], p2[1] - offset_detector_to_coll - h_R])

    # Calculate crossp (intersection of p1-p3 and p2-p4)
    crossp = calculate_intersection(p1, p3, p2, p4)

    # Constraint: crossp y must be smaller than p6 and p8 y values
    return min(p6[1], p8[1]) - crossp[1]


def constraint2(params):
    h_L, ang_L, h_R, ang_R, det_y = params

    # Calculate points p1, p2
    p1 = np.array([-r - (pitch / 2), det_y])
    p2 = np.array([-r + (pitch / 2), det_y])

    # Calculate p5, p6, p7, p8
    p5 = np.array([p1[0], p1[1] - offset_detector_to_coll])
    p6 = np.array([p1[0], p1[1] - offset_detector_to_coll - h_L])
    p7 = np.array([p2[0], p2[1] - offset_detector_to_coll])
    p8 = np.array([p2[0], p2[1] - offset_detector_to_coll - h_R])

    # Constraint: det_y must be greater than p5 and p7 y values
    return det_y - max(p5[1], p7[1])


# Bounds for variables (h_L, ang_L, h_R, ang_R, det_y)
bounds = [(20, 70), (0, 45), (20, 70), (0, 45), (offset_coll_to_obj + offset_detector_to_coll + 20, 80)]

# Constraints for the optimization
constraints = [{'type': 'ineq', 'fun': constraint1},  # crossp y < p6, p8 y
               {'type': 'ineq', 'fun': constraint2}]  # det_y > p5, p7 y

# Initial guess for the parameters
initial_guess = [50, 20, 50, 20, 60]

# Perform optimization
result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints)

# Output the results
if result.success:
    optimized_params = result.x
    h_L_opt, ang_L_opt, h_R_opt, ang_R_opt, det_y_opt = optimized_params
    print(f"Optimized h_L: {h_L_opt}, ang_L: {ang_L_opt}")
    print(f"Optimized h_R: {h_R_opt}, ang_R: {ang_R_opt}")
    print(f"Optimized det_y: {det_y_opt}")
else:
    print("Optimization failed.")


# Induced points based on initial guess
p1_opt = np.array([-r - (pitch / 2), det_y_opt])
p2_opt = np.array([-r + (pitch / 2), det_y_opt])

# Calculate the intersection (crossp)
crossp_opt = calculate_intersection(p1_opt, p3, p2_opt, p4)

# Define points p5, p6, p7, p8 based on p1, p2 and collimator heights and offsets
p5_opt = np.array([p1_opt[0], p1_opt[1] - offset_detector_to_coll])
p6_opt = np.array([p1_opt[0], p1_opt[1] - offset_detector_to_coll - h_L_opt])
p7_opt = np.array([p2_opt[0], p2_opt[1] - offset_detector_to_coll])
p8_opt = np.array([p2_opt[0], p2_opt[1] - offset_detector_to_coll - h_R_opt])

# Define the outer corners for the rectangles using width (w)
p5_outer_opt = np.array([p5_opt[0] - w, p5_opt[1]])  # Left top of coll_L
p6_outer_opt = np.array([p6_opt[0] - w, p6_opt[1]])  # Left bottom of coll_L
p7_outer_opt = np.array([p7_opt[0] + w, p7_opt[1]])  # Right top of coll_R
p8_outer_opt = np.array([p8_opt[0] + w, p8_opt[1]])  # Right bottom of coll_R


# Calculate center gravity for left and right collimators using four points
cg_L_opt = calculate_center_gravity(p5_opt, p6_opt, p5_outer_opt, p6_outer_opt)
cg_R_opt = calculate_center_gravity(p7_opt, p8_opt, p7_outer_opt, p8_outer_opt)


# Rotate the points around their respective center gravity
p5_rot_opt = rotate_point(p5_opt, cg_L_opt, ang_L_opt)
p6_rot_opt = rotate_point(p6_opt, cg_L_opt, ang_L_opt)
p5_outer_rot_opt = rotate_point(p5_outer_opt, cg_L_opt, ang_L_opt)
p6_outer_rot_opt = rotate_point(p6_outer_opt, cg_L_opt, ang_L_opt)

p7_rot_opt = rotate_point(p7_opt, cg_R_opt, ang_R_opt)
p8_rot_opt = rotate_point(p8_opt, cg_R_opt, ang_R_opt)
p7_outer_rot_opt = rotate_point(p7_outer_opt, cg_R_opt, ang_R_opt)
p8_outer_rot_opt = rotate_point(p8_outer_opt, cg_R_opt, ang_R_opt)


# Create the plot
fig2, ax2 = plt.subplots(figsize=(10, 10))

# Plot the points p1, p2, p3, p4
ax2.scatter([p1_opt[0], p2_opt[0], p3[0], p4[0]], [p1_opt[1], p2_opt[1], p3[1], p4[1]], color='red', s=50, zorder=5)
# Plot the rotated points p5, p6, p7, p8
ax2.scatter([p5_rot_opt[0], p6_rot_opt[0], p7_rot_opt[0], p8_rot_opt[0]], [p5_rot_opt[1], p6_rot_opt[1], p7_rot_opt[1], p8_rot_opt[1]], color='blue',
           s=50, zorder=5)

# Plot the center gravity points cg_L and cg_R (in purple)
ax2.scatter(cg_L_opt[0], cg_L_opt[1], color='purple', s=100, zorder=5)
ax2.annotate('cg_L_opt', (cg_L_opt[0], cg_L_opt[1]), xytext=(5, 5), textcoords='offset points')
ax2.scatter(cg_R_opt[0], cg_R_opt[1], color='purple', s=100, zorder=5)
ax2.annotate('cg_R_opt', (cg_R_opt[0], cg_R_opt[1]), xytext=(5, 5), textcoords='offset points')

# Plot the cross point if it exists
if crossp_opt is not None:
    ax2.scatter(crossp_opt[0], crossp_opt[1], color='green', s=100, zorder=5)
    ax2.annotate('crossp_opt', (crossp_opt[0], crossp_opt[1]), xytext=(5, 5), textcoords='offset points')

# Draw lines connecting p1 to p3, and p2 to p4
ax2.plot([p1_opt[0], p3[0]], [p1_opt[1], p3[1]], color='magenta', linestyle='--', linewidth=1.5)
ax2.plot([p2_opt[0], p4[0]], [p2_opt[1], p4[1]], color='cyan', linestyle='--', linewidth=1.5)

# Add the rotated rectangles for coll_L and coll_R
rect_L_rot_opt = plt.Polygon([[p5_outer_rot_opt[0], p5_outer_rot_opt[1]], [p5_rot_opt[0], p5_rot_opt[1]], [p6_rot_opt[0], p6_rot_opt[1]],
                          [p6_outer_rot_opt[0], p6_outer_rot_opt[1]]],
                         closed=True, linewidth=1.5, edgecolor='black', facecolor='gray', alpha=0.5)
rect_R_rot_opt = plt.Polygon(
    [[p7_rot_opt[0], p7_rot_opt[1]], [p7_outer_rot_opt[0], p7_outer_rot_opt[1]], [p8_outer_rot_opt[0], p8_outer_rot_opt[1]],
     [p8_rot_opt[0], p8_rot_opt[1]]],
    closed=True, linewidth=1.5, edgecolor='black', facecolor='gray', alpha=0.5)

ax2.add_patch(rect_L_rot_opt)
ax2.add_patch(rect_R_rot_opt)

# Set axis labels
ax2.set_xlabel('X')
ax2.set_ylabel('Y')

# Show grid
ax2.grid(True)

# Set equal aspect ratio
ax2.set_aspect('equal', adjustable='box')

# Add title
plt.title('Opt : 2D Coordinate System with Rotated Collimators and Cross Point')


# Show the plot
plt.show()