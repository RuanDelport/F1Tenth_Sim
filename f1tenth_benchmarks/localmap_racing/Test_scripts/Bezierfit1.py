import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# Function to calculate a cubic Bezier curve from four control points
def cubic_bezier(p0, p1, p2, p3, num_points=100):
    t = np.linspace(0, 1, num_points)[:, None]  # Reshape to (100, 1) for broadcasting
    bezier = (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3
    return bezier

# Example lidar-like data points (replace with actual lidar points)
np.random.seed(0)
num_points = 50
theta = np.linspace(0, 2*np.pi, num_points)
x = 5 * np.cos(theta) + np.random.normal(0, 0.2, num_points)
y = 5 * np.sin(theta) + np.random.normal(0, 0.2, num_points)

# Select a segment of lidar points (you can adjust these indices)
segment_indices = range(10, 14)
lidar_segment = np.column_stack((x[segment_indices], y[segment_indices]))

# Use the first and last point as endpoints, and the middle two as control points
p0, p1, p2, p3 = lidar_segment[0], lidar_segment[1], lidar_segment[2], lidar_segment[3]

# Fit a cubic Bezier curve to the four points
bezier_points = cubic_bezier(p0, p1, p2, p3)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'ro', label='Lidar Points', markersize=5)
plt.plot(lidar_segment[:, 0], lidar_segment[:, 1], 'bo-', label='Segment', markersize=8)
plt.plot(bezier_points[:, 0], bezier_points[:, 1], 'g-', label='Cubic Bezier Curve', linewidth=2)

# Plot control points
control_points = np.array([p0, p1, p2, p3])
plt.plot(control_points[:, 0], control_points[:, 1], 'kx-', label='Control Points', markersize=8)
for i, (x, y) in enumerate(control_points):
    plt.text(x, y, f'P{i}', fontsize=12, color='black')

plt.title('Cubic Bezier Curve Fit to Lidar Segment')
plt.legend()
plt.grid(True)
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.show()

