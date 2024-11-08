import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Function to convert Cartesian coordinates to Frenet coordinates
def cartesian_to_frenet(x, y, path):
    diffs = path - np.array([x, y])
    dists = np.linalg.norm(diffs, axis=1)
    closest_index = np.argmin(dists)
    
    # Arc length along the path
    s = np.sum(np.linalg.norm(np.diff(path[:closest_index + 1], axis=0), axis=1))
    
    # Lateral distance from the path
    tangent = path[min(closest_index + 1, len(path) - 1)] - path[closest_index]
    normal = np.array([-tangent[1], tangent[0]])
    normal /= np.linalg.norm(normal)
    d = np.dot(np.array([x, y]) - path[closest_index], normal)
    
    return s, d, normal, tangent, closest_index

# Function to transform covariance matrix from Cartesian to Frenet
def transform_covariance(cov_xy, normal, tangent):
    J = np.array([tangent, normal]).T
    cov_sd = J.T @ cov_xy @ J
    return cov_sd

# Visualization function for uncertainty ellipses and robot orientation
def plot_ellipse(ax, mean, cov, color='blue', label=None):
    eigvals, eigvecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(eigvals)
    
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, 
                      edgecolor=color, fc='none', lw=2, label=label)
    ax.add_patch(ellipse)

# Generate a simple curved path (a circle segment)
theta = np.linspace(0, np.pi / 2, 100)
path = np.array([10 * np.cos(theta), 10 * np.sin(theta)]).T

# Define robot position, orientation, and Cartesian uncertainty
x, y, theta_robot = 9, 3, np.pi / 4
cov_xy = np.array([[0.5, 0.2], [0.2, 0.3]])

# Convert to Frenet frame
s, d, normal, tangent, idx = cartesian_to_frenet(x, y, path)
cov_sd = transform_covariance(cov_xy, normal, tangent)

# Visualization
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Cartesian Frame Visualization
ax[0].plot(path[:, 0], path[:, 1], 'k-', label='Reference Path')
ax[0].plot(x, y, 'ro', label='Robot Position')
plot_ellipse(ax[0], [x, y], cov_xy, color='blue', label='Cartesian Uncertainty')
# Orientation arrow
ax[0].arrow(x, y, 0.5 * np.cos(theta_robot), 0.5 * np.sin(theta_robot), 
            head_width=0.2, head_length=0.3, fc='red', ec='red', label='Orientation')
ax[0].set_title('Cartesian Frame')
ax[0].set_aspect('equal')
ax[0].legend()

# Frenet Frame Visualization
ax[1].plot(path[:, 0], path[:, 1], 'k-', label='Reference Path')
ax[1].plot(path[idx, 0], path[idx, 1], 'go', label='Closest Point on Path')
# Plot Frenet frame vectors
ax[1].arrow(path[idx, 0], path[idx, 1], tangent[0], tangent[1], color='purple', 
            head_width=0.2, head_length=0.3, label='Tangent (s)')
ax[1].arrow(path[idx, 0], path[idx, 1], normal[0], normal[1], color='orange', 
            head_width=0.2, head_length=0.3, label='Normal (d)')
# Extend normal vector to robot position in Frenet frame
robot_proj = path[idx] + d * normal
ax[1].plot([path[idx, 0], robot_proj[0]], [path[idx, 1], robot_proj[1]], 
           'r--', label='Lateral Offset (d)')
ax[1].plot(robot_proj[0], robot_proj[1], 'ro')
# Show transformed uncertainty in Frenet frame
plot_ellipse(ax[1], [s, d], cov_sd, color='green', label='Frenet Uncertainty')
ax[1].text(s + 0.2, d, f's: {s:.2f}\nd: {d:.2f}', fontsize=10)
ax[1].set_title('Frenet Frame')
ax[1].set_aspect('equal')
ax[1].legend()

plt.show()


