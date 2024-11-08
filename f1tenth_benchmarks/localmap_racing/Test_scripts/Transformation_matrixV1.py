import matplotlib.pyplot as plt
import numpy as np

# Define two particles with (x, y) position and heading (in radians)
particle1 = np.array([1, 2, np.radians(30)])  # Particle A: (x, y, heading)
particle2 = np.array([4, 5, np.radians(75)])  # Particle B: (x, y, heading)

# Function to compute the transformation matrix from one particle to another
def compute_relative_transformation(p1, p2):
    # Relative translation
    delta_x = p2[0] - p1[0]
    delta_y = p2[1] - p1[1]
    
    # Relative rotation (heading difference)
    delta_theta = p2[2] - p1[2]
    
    # Rotation matrix for delta_theta
    cos_theta = np.cos(delta_theta)
    sin_theta = np.sin(delta_theta)
    rotation_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])
    
    # Construct the full transformation matrix (3x3)
    transformation_matrix = np.array([
        [cos_theta, -sin_theta, delta_x],
        [sin_theta, cos_theta, delta_y],
        [0, 0, 1]
    ])
    
    return transformation_matrix, delta_x, delta_y, np.degrees(delta_theta)

# Calculate the relative transformation from particle1 to particle2
transformation_matrix, delta_x, delta_y, delta_theta_deg = compute_relative_transformation(particle1, particle2)

# Visualization
def plot_particle(ax, particle, label, color):
    x, y, theta = particle
    dx, dy = np.cos(theta), np.sin(theta)
    ax.plot(x, y, marker='o', color=color, label=label)
    ax.arrow(x, y, dx, dy, color=color, head_width=0.2, head_length=0.2)

plt.figure(figsize=(8, 8))
ax = plt.gca()
ax.set_aspect('equal')
plt.axhline(0, color='grey', lw=0.5)
plt.axvline(0, color='grey', lw=0.5)

# Plot original particles
plot_particle(ax, particle1, 'Particle 1 (A)', 'blue')
plot_particle(ax, particle2, 'Particle 2 (B)', 'red')

# Visualize the transformation from particle1 to particle2
transformed_particle1 = np.dot(transformation_matrix, np.array([particle1[0], particle1[1], 1]))
plot_particle(ax, [*transformed_particle1[:2], particle2[2]], 'Transformed A to B', 'green')

# Display transformation details
ax.arrow(particle1[0], particle1[1], delta_x, delta_y, color='purple', linestyle='--', head_width=0.2)
plt.annotate(f"Rotation: {delta_theta_deg:.2f}°\nTranslation: ({delta_x:.2f}, {delta_y:.2f})",
             xy=(particle1[0] + delta_x, particle1[1] + delta_y), 
             xytext=(particle1[0] + delta_x + 0.5, particle1[1] + delta_y + 0.5),
             arrowprops=dict(arrowstyle="->", color='purple'))

# Final plot adjustments
plt.title('Relative Transformation from Particle 1 (A) to Particle 2 (B)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()


