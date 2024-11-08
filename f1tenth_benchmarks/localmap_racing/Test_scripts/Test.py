import numpy as np
import matplotlib.pyplot as plt

# Define points
point1 = np.array([0, 0, 1])  # Homogeneous coordinate
point2 = np.array([1, 1, 1])  # Homogeneous coordinate
point3 = np.array([2, 2, 1])  # Homogeneous coordinate

# Define transformation function
def compute_relative_transformation(p1, p2):
    delta_x = p2[0] - p1[0]
    delta_y = p2[1] - p1[1]
    delta_theta = p2[2] - p1[2]
    
    cos_theta = np.cos(delta_theta)
    sin_theta = np.sin(delta_theta)
    
    transformation_matrix = np.array([
        [cos_theta, -sin_theta, delta_x],
        [sin_theta, cos_theta, delta_y],
        [0, 0, 1]
    ])
    return transformation_matrix

# Compute transformations
transformation_matrix1 = compute_relative_transformation(point1, point2)
transformation_matrix2 = compute_relative_transformation(point2, point3)
combined_transformation = np.dot(transformation_matrix2, transformation_matrix1)

# Apply transformations
first_transform = np.dot(transformation_matrix1, point1)
second_transform = np.dot(transformation_matrix2, first_transform)
direct_transform = np.dot(combined_transformation, point1)

# Print results
print("First Transform:", first_transform)
print("Second Transform:", second_transform)
print("Direct Transform:", direct_transform)

# Plot the results
plt.figure(figsize=(8, 8))
plt.arrow(point1[0], point1[1], np.cos(0), np.sin(0), head_width=0.1, head_length=0.1, fc='r', ec='r', label='Point 1')
plt.arrow(first_transform[0], first_transform[1], np.cos(np.pi/4), np.sin(np.pi/4), head_width=0.1, head_length=0.1, fc='b', ec='b', label='First Transform')
plt.arrow(second_transform[0], second_transform[1], np.cos(np.pi/2), np.sin(np.pi/2), head_width=0.1, head_length=0.1, fc='g', ec='g', label='Second Transform')
plt.arrow(direct_transform[0], direct_transform[1], np.cos(np.pi/2), np.sin(np.pi/2), head_width=0.1, head_length=0.1, fc='m', ec='m', label='Direct Transform')
plt.legend()
plt.grid()
plt.xlim(0, 4)
plt.ylim(0, 4)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Point Transformations')
plt.tight_layout()
plt.show()


