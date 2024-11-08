import numpy as np
import matplotlib.pyplot as plt

# Define a function to calculate the Bézier curve points
def bezier_curve(control_points, num_points=100):
    t = np.linspace(0, 1, num_points).reshape(-1, 1)  # Reshape t to (100, 1) for broadcasting
    curve_points = (
        (1 - t)**3 * control_points[0] +
        3 * (1 - t)**2 * t * control_points[1] +
        3 * (1 - t) * t**2 * control_points[2] +
        t**3 * control_points[3]
    )
    return curve_points

# Define an affine transformation function
def affine_transform(points, matrix, translation):
    return np.dot(points, matrix.T) + translation

# Original control points
control_points = np.array([
    [0, 0],
    [1, 2],
    [3, 3],
    [4, 0]
])

# Calculate the centroid of the original control points
original_centroid = np.mean(control_points, axis=0)

# Generate the original Bézier curve
original_curve = bezier_curve(control_points)

# Define rotation around the centroid
rotation_angle = np.radians(30)  # 30 degrees
scale_factor = 1.2
rotation_matrix = np.array([
    [scale_factor * np.cos(rotation_angle), -scale_factor * np.sin(rotation_angle)],
    [scale_factor * np.sin(rotation_angle),  scale_factor * np.cos(rotation_angle)]
])

# Step 1: Translate control points to place the centroid at the origin
translated_points = control_points - original_centroid

# Step 2: Rotate the translated points
rotated_points = affine_transform(translated_points, rotation_matrix, translation=np.array([0, 0]))

# Step 3: Translate back to the original centroid position
transformed_control_points = rotated_points + original_centroid

# Calculate the centroid of the transformed control points
transformed_centroid = np.mean(transformed_control_points, axis=0)

# Generate the transformed Bézier curve
transformed_curve = bezier_curve(transformed_control_points)

# Plot the original and transformed curves along with their centroids
plt.figure(figsize=(10, 6))
plt.plot(original_curve[:, 0], original_curve[:, 1], label="Original Bézier Curve", color="blue")
plt.plot(control_points[:, 0], control_points[:, 1], 'o--', label="Original Control Points", color="blue")
plt.scatter(*original_centroid, color="blue", marker="x", s=100, label="Original Centroid")

plt.plot(transformed_curve[:, 0], transformed_curve[:, 1], label="Transformed Bézier Curve", color="red")
plt.plot(transformed_control_points[:, 0], transformed_control_points[:, 1], 'o--', label="Transformed Control Points", color="red")
plt.scatter(*transformed_centroid, color="red", marker="x", s=100, label="Transformed Centroid")

# Labels and legend
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Bézier Curve, Control Points, and Centroids with Rotation around Centroid")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()
