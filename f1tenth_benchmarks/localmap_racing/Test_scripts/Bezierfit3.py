import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define a cubic Bézier curve function
def cubic_bezier(t, control_points):
    return (
        (1 - t)**3 * control_points[0] +
        3 * (1 - t)**2 * t * control_points[1] +
        3 * (1 - t) * t**2 * control_points[2] +
        t**3 * control_points[3]
    )

# Objective function to minimize the distance between Lidar points and Bézier curve
def objective(control_points, points):
    control_points = control_points.reshape(-1, 2)  # Reshape control points
    t_values = np.linspace(0, 1, len(points))  # Parameter values
    bezier_points = np.array([cubic_bezier(t, control_points) for t in t_values])
    return np.sum(np.linalg.norm(bezier_points - points, axis=1))

# Generate simulated Lidar points along a 90-degree curve
t_values = np.linspace(0, np.pi / 2, 20)  # 20 points from 0 to π/2
curve_points = np.array([[np.cos(t), np.sin(t)] for t in t_values])  # Circular curve
noise = np.random.normal(0, 0.05, curve_points.shape)  # Adding noise
lidar_points = curve_points + noise  # Simulated Lidar points

# Initial guess for control points (based on the first and last Lidar points)
initial_control_points = np.array([lidar_points[0], [0.5, 1.5], [1.5, 0.5], lidar_points[-1]])

# Optimize control points
result = minimize(objective, initial_control_points.flatten(), args=(lidar_points,))
optimized_control_points = result.x.reshape(-1, 2)

# Generate Bézier curve points with optimized control points
bezier_points = np.array([cubic_bezier(t, optimized_control_points) for t in np.linspace(0, 1, 100)])

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(lidar_points[:, 0], lidar_points[:, 1], 'ro', label='Lidar Points', markersize=8)  # Lidar points
plt.plot(bezier_points[:, 0], bezier_points[:, 1], 'b-', label='Cubic Bézier Curve')  # Fitted Bézier curve
plt.plot(optimized_control_points[:, 0], optimized_control_points[:, 1], 'go--', label='Control Points', markersize=8)  # Control points
plt.title('Cubic Bézier Curve Fitting to Lidar Points on a 90-Degree Curve')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.grid()
plt.legend()
plt.show()

# Output the optimized control points
print("Optimized Control Points:")
for i, point in enumerate(optimized_control_points):
    print(f"Control Point {i + 1}: (x: {point[0]:.2f}, y: {point[1]:.2f})")



