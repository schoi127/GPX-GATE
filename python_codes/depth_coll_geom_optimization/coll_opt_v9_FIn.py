import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Fixed constants
r = 23.75
pitch = 1.2
depth_target_top = 20
depth_target_bottom = 70
offset_detector_to_coll = 5
offset_coll_to_obj = 5
w = 2  # Width of the collimators

# Fixed points
p3 = np.array([0, -depth_target_top])
p4 = np.array([0, -depth_target_bottom])


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


def rotate_point(point, center, angle):
    angle_rad = np.radians(angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    translated_point = point - center
    rotated_x = translated_point[0] * cos_angle - translated_point[1] * sin_angle
    rotated_y = translated_point[0] * sin_angle + translated_point[1] * cos_angle
    rotated_point = np.array([rotated_x, rotated_y]) + center
    return rotated_point

def rotate_vector(vector, angle):
    angle_rad = np.radians(angle)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    return np.dot(rotation_matrix, vector)

# def objective(params):
#     cg_L_x, cg_L_y, cg_R_x, cg_R_y, h_L, ang_L, h_R, ang_R, det_y = params
#
#     # Calculate p1, p2
#     p1 = np.array([-r - (pitch / 2), det_y])
#     p2 = np.array([-r + (pitch / 2), det_y])
#
#     # Calculate p5, p6, p7, p8
#     p5 = np.array([cg_L_x, cg_L_y + h_L / 2])
#     p6 = np.array([cg_L_x, cg_L_y - h_L / 2])
#     p7 = np.array([cg_R_x, cg_R_y + h_R / 2])
#     p8 = np.array([cg_R_x, cg_R_y - h_R / 2])
#
#     # Rotate points
#     p5_rot = rotate_point(p5, [cg_L_x, cg_L_y], ang_L)
#     p6_rot = rotate_point(p6, [cg_L_x, cg_L_y], ang_L)
#     p7_rot = rotate_point(p7, [cg_R_x, cg_R_y], ang_R)
#     p8_rot = rotate_point(p8, [cg_R_x, cg_R_y], ang_R)
#
#     # Calculate distances from rotated points to lines
#     line1_dist = np.abs(np.cross(p3 - p1, p5_rot - p1)) / np.linalg.norm(p3 - p1) + \
#                  np.abs(np.cross(p3 - p1, p6_rot - p1)) / np.linalg.norm(p3 - p1)
#     line2_dist = np.abs(np.cross(p4 - p2, p7_rot - p2)) / np.linalg.norm(p4 - p2) + \
#                  np.abs(np.cross(p4 - p2, p8_rot - p2)) / np.linalg.norm(p4 - p2)
#
#     return line1_dist + line2_dist
#
#
# def constraint1(params):
#     cg_L_x, cg_L_y, cg_R_x, cg_R_y, h_L, ang_L, h_R, ang_R, det_y = params
#
#     # Calculate p1, p2
#     p1 = np.array([-r - (pitch / 2), det_y])
#     p2 = np.array([-r + (pitch / 2), det_y])
#
#     # Calculate p5, p6, p7, p8
#     p5 = np.array([cg_L_x, cg_L_y + h_L / 2])
#     p6 = np.array([cg_L_x, cg_L_y - h_L / 2])
#     p7 = np.array([cg_R_x, cg_R_y + h_R / 2])
#     p8 = np.array([cg_R_x, cg_R_y - h_R / 2])
#
#     # Rotate points
#     p5_rot = rotate_point(p5, [cg_L_x, cg_L_y], ang_L)
#     p6_rot = rotate_point(p6, [cg_L_x, cg_L_y], ang_L)
#     p7_rot = rotate_point(p7, [cg_R_x, cg_R_y], ang_R)
#     p8_rot = rotate_point(p8, [cg_R_x, cg_R_y], ang_R)
#
#     # Calculate crossp
#     crossp = calculate_intersection(p1, p3, p2, p4)
#
#     # Constraint: crossp y must be smaller than p6 and p8 y values
#     return min(p6_rot[1], p8_rot[1]) - crossp[1]

def objective(params):
    cg_L_x, cg_L_y, cg_R_x, cg_R_y, h_L, ang_L, h_R, ang_R, det_y = params

    # Calculate p1, p2
    p1 = np.array([-r - (pitch / 2), det_y])
    p2 = np.array([-r + (pitch / 2), det_y])

    # Calculate p5, p6, p7, p8
    p5 = np.array([cg_L_x, cg_L_y + h_L / 2])
    p6 = np.array([cg_L_x, cg_L_y - h_L / 2])
    p7 = np.array([cg_R_x, cg_R_y + h_R / 2])
    p8 = np.array([cg_R_x, cg_R_y - h_R / 2])

    # Rotate points
    p5_rot = rotate_point(p5, [cg_L_x, cg_L_y], ang_L)
    p6_rot = rotate_point(p6, [cg_L_x, cg_L_y], ang_L)
    p7_rot = rotate_point(p7, [cg_R_x, cg_R_y], ang_R)
    p8_rot = rotate_point(p8, [cg_R_x, cg_R_y], ang_R)

    # Calculate outer points
    normal_L = rotate_vector(np.array([-1, 0]), ang_L)
    normal_R = rotate_vector(np.array([1, 0]), ang_R)

    p5_rot_outer = p5_rot + w * normal_L
    p6_rot_outer = p6_rot + w * normal_L
    p7_rot_outer = p7_rot + w * normal_R
    p8_rot_outer = p8_rot + w * normal_R

    # Calculate distances from rotated points to lines
    line1_dist = np.abs(np.cross(p3 - p1, p5_rot - p1)) / np.linalg.norm(p3 - p1) + \
                 np.abs(np.cross(p3 - p1, p6_rot - p1)) / np.linalg.norm(p3 - p1)
    line2_dist = np.abs(np.cross(p4 - p2, p7_rot - p2)) / np.linalg.norm(p4 - p2) + \
                 np.abs(np.cross(p4 - p2, p8_rot - p2)) / np.linalg.norm(p4 - p2)

    return line1_dist + line2_dist


def constraint1(params):
    cg_L_x, cg_L_y, cg_R_x, cg_R_y, h_L, ang_L, h_R, ang_R, det_y = params

    # Calculate p1, p2
    p1 = np.array([-r - (pitch / 2), det_y])
    p2 = np.array([-r + (pitch / 2), det_y])

    # Calculate p5, p6, p7, p8
    p5 = np.array([cg_L_x, cg_L_y + h_L / 2])
    p6 = np.array([cg_L_x, cg_L_y - h_L / 2])
    p7 = np.array([cg_R_x, cg_R_y + h_R / 2])
    p8 = np.array([cg_R_x, cg_R_y - h_R / 2])

    # Rotate points
    p5_rot = rotate_point(p5, [cg_L_x, cg_L_y], ang_L)
    p6_rot = rotate_point(p6, [cg_L_x, cg_L_y], ang_L)
    p7_rot = rotate_point(p7, [cg_R_x, cg_R_y], ang_R)
    p8_rot = rotate_point(p8, [cg_R_x, cg_R_y], ang_R)

    # Calculate outer points
    normal_L = rotate_vector(np.array([-1, 0]), ang_L)
    normal_R = rotate_vector(np.array([1, 0]), ang_R)

    p5_rot_outer = p5_rot + w * normal_L
    p6_rot_outer = p6_rot + w * normal_L
    p7_rot_outer = p7_rot + w * normal_R
    p8_rot_outer = p8_rot + w * normal_R

    # Calculate crossp
    crossp = calculate_intersection(p1, p3, p2, p4)

    # Constraint: crossp y must be smaller than p6 and p8 y values
    return min(p6_rot[1], p8_rot[1]) - crossp[1]


def constraint2(params):
    cg_L_x, cg_L_y, cg_R_x, cg_R_y, h_L, ang_L, h_R, ang_R, det_y = params

    # Calculate p5, p7
    p5 = rotate_point(np.array([cg_L_x, cg_L_y + h_L / 2]), [cg_L_x, cg_L_y], ang_L)
    p7 = rotate_point(np.array([cg_R_x, cg_R_y + h_R / 2]), [cg_R_x, cg_R_y], ang_R)

    # Constraint: det_y must be greater than p5 and p7 y values
    return det_y - max(p5[1], p7[1])


# Initial guess
initial_guess = [
    -r - (pitch / 2), 0,  # cg_L
    -r + (pitch / 2), 0,  # cg_R
    50, 20,  # h_L, ang_L
    50, 20,  # h_R, ang_R
    60  # det_y
]

# Bounds for variables
bounds = [
    (-30, 0), (20, 70),  # cg_L
    (-30, 0), (20, 70),  # cg_R
    (20, 70), (0, 45),  # h_L, ang_L
    (20, 70), (0, 45),  # h_R, ang_R
    (offset_coll_to_obj + offset_detector_to_coll + 20, 80)  # det_y
]

# Constraints
constraints = [
    {'type': 'ineq', 'fun': constraint1},
    {'type': 'ineq', 'fun': constraint2}
]

# Perform optimization
result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

# Output the results
if result.success:
    optimized_params = result.x
    cg_L_x, cg_L_y, cg_R_x, cg_R_y, h_L, ang_L, h_R, ang_R, det_y = optimized_params
    print(f"Optimized cg_L: ({cg_L_x}, {cg_L_y})")
    print(f"Optimized cg_R: ({cg_R_x}, {cg_R_y})")
    print(f"Optimized h_L: {h_L}, ang_L: {ang_L}")
    print(f"Optimized h_R: {h_R}, ang_R: {ang_R}")
    print(f"Optimized det_y: {det_y}")

    # Calculate and print the outer points
    p5 = np.array([cg_L_x, cg_L_y + h_L / 2])
    p6 = np.array([cg_L_x, cg_L_y - h_L / 2])
    p7 = np.array([cg_R_x, cg_R_y + h_R / 2])
    p8 = np.array([cg_R_x, cg_R_y - h_R / 2])

    p5_rot = rotate_point(p5, [cg_L_x, cg_L_y], ang_L)
    p6_rot = rotate_point(p6, [cg_L_x, cg_L_y], ang_L)
    p7_rot = rotate_point(p7, [cg_R_x, cg_R_y], ang_R)
    p8_rot = rotate_point(p8, [cg_R_x, cg_R_y], ang_R)

    normal_L = rotate_vector(np.array([-1, 0]), ang_L)
    normal_R = rotate_vector(np.array([1, 0]), ang_R)

    p5_rot_outer = p5_rot + w * normal_L
    p6_rot_outer = p6_rot + w * normal_L
    p7_rot_outer = p7_rot + w * normal_R
    p8_rot_outer = p8_rot + w * normal_R

    print(f"p5_rot_outer: {p5_rot_outer}")
    print(f"p6_rot_outer: {p6_rot_outer}")
    print(f"p7_rot_outer: {p7_rot_outer}")
    print(f"p8_rot_outer: {p8_rot_outer}")
else:
    print("Optimization failed.")

# Calculate p1, p2
p1_opt = np.array([-r - (pitch / 2), det_y])
p2_opt = np.array([-r + (pitch / 2), det_y])

# Calculate p5, p6, p7, p8
p5_opt = np.array([cg_L_x, cg_L_y + h_L / 2])
p6_opt = np.array([cg_L_x, cg_L_y - h_L / 2])
p7_opt = np.array([cg_R_x, cg_R_y + h_R / 2])
p8_opt = np.array([cg_R_x, cg_R_y - h_R / 2])

# Rotate points
p5_rot_opt = rotate_point(p5_opt, [cg_L_x, cg_L_y], ang_L)
p6_rot_opt = rotate_point(p6_opt, [cg_L_x, cg_L_y], ang_L)
p7_rot_opt = rotate_point(p7_opt, [cg_R_x, cg_R_y], ang_R)
p8_rot_opt = rotate_point(p8_opt, [cg_R_x, cg_R_y], ang_R)
normal_L = rotate_vector(np.array([-1, 0]), ang_L)
normal_R = rotate_vector(np.array([1, 0]), ang_R)

p5_rot_outer_opt = p5_rot_opt + w * normal_L
p6_rot_outer_opt = p6_rot_opt + w * normal_L
p7_rot_outer_opt = p7_rot_opt + w * normal_R
p8_rot_outer_opt = p8_rot_opt + w * normal_R

cg_L_opt = [cg_L_x, cg_L_y]
cg_R_opt = [cg_R_x, cg_R_y]

# Calculate crossp
crossp_opt = calculate_intersection(p1_opt, p3, p2_opt, p4)

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
rect_L_rot_opt = plt.Polygon([[p5_rot_outer_opt[0], p5_rot_outer_opt[1]], [p5_rot_opt[0], p5_rot_opt[1]], [p6_rot_opt[0], p6_rot_opt[1]],
                          [p6_rot_outer_opt[0], p6_rot_outer_opt[1]]],
                         closed=True, linewidth=1.5, edgecolor='black', facecolor='gray', alpha=0.5)
rect_R_rot_opt = plt.Polygon(
    [[p7_rot_opt[0], p7_rot_opt[1]], [p7_rot_outer_opt[0], p7_rot_outer_opt[1]], [p8_rot_outer_opt[0], p8_rot_outer_opt[1]],
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