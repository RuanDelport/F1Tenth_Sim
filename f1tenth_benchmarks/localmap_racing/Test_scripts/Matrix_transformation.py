import numpy as np
import matplotlib.pyplot as plt

# Sample localtrack points
localtrack = np.array([[1, 2],
                       [2, 3],
                       [3, 5],
                       [4, 4]])

# Convert to homogeneous coordinates
localtrack_homogeneous = np.hstack((localtrack, np.ones((localtrack.shape[0], 1))))

# Define transformation matrices
local_to_global_matrix = np.array([[1, 0, 2],  # Translation by 2 in x
                                   [0, 1, 3],  # Translation by 3 in y
                                   [0, 0, 1]])

local_transformation_matrix = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4), 0],  # 45-degree rotation
                                        [np.sin(np.pi/4),  np.cos(np.pi/4), 0],
                                        [0, 0, 1]])

# Combine transformations
combined_transformation = np.dot(local_to_global_matrix, local_transformation_matrix)

# Apply local and combined transformations
local_transformed = np.dot(local_transformation_matrix, localtrack_homogeneous.T).T
global_transformed = np.dot(combined_transformation, localtrack_homogeneous.T).T

# Extract Cartesian coordinates
local_transformed_cartesian = local_transformed[:, :2] / local_transformed[:, 2][:, np.newaxis]
global_transformed_cartesian = global_transformed[:, :2] / global_transformed[:, 2][:, np.newaxis]

# Plotting
plt.figure(figsize=(8, 8))
plt.scatter(localtrack[:, 0], localtrack[:, 1], color='blue', label='Original Points')
plt.scatter(local_transformed_cartesian[:, 0], local_transformed_cartesian[:, 1], color='orange', label='After Local Transformation')
plt.scatter(global_transformed_cartesian[:, 0], global_transformed_cartesian[:, 1], color='green', label='After Combined Transformation')

# Connect points for clarity
for i in range(localtrack.shape[0]):
    plt.plot([localtrack[i, 0], local_transformed_cartesian[i, 0]], [localtrack[i, 1], local_transformed_cartesian[i, 1]], 'r--')
    plt.plot([local_transformed_cartesian[i, 0], global_transformed_cartesian[i, 0]], [local_transformed_cartesian[i, 1], global_transformed_cartesian[i, 1]], 'g--')

plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.legend()
plt.title('Transformation Process of localtrack Points')
# plt.xlim(-1, 10)
# plt.ylim(-1, 10)
plt.show()
