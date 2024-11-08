import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# Function to calculate Bezier curve from control points
def bezier_curve(control_points, num_points=100):
    t_values = np.linspace(0, 1, num_points)[:, None]  # Reshape t_values to (num_points, 1) once
    curve = np.zeros((num_points, 2))  # 2D array for x and y coordinates

    n = len(control_points) - 1
    for i in range(n + 1):
        binom_coeff = np.math.comb(n, i)
        curve += binom_coeff * (1 - t_values)**(n - i) * t_values**i * control_points[i]

    return curve

# Example set of curved points (you can replace these with your own data)
np.random.seed(0)
points = np.array([[0, 0], [1, 2], [2, 3], [3, 1], [4, 0]])

# Fit a spline to the points (this generates the control points)
tck, u = splprep(points.T, s=0)
control_points = np.column_stack([splev(u, tck, der=0)[0], splev(u, tck, der=0)[1]])

# Calculate the Bezier curve from control points
bezier_points = bezier_curve(control_points)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(points[:, 0], points[:, 1], 'ro-', label='Original Points', markersize=5)
plt.plot(control_points[:, 0], control_points[:, 1], 'bx-', label='Control Points', markersize=7)
plt.plot(bezier_points[:, 0], bezier_points[:, 1], 'g-', label='Bezier Curve', linewidth=2)

# Display control points
for i, (x, y) in enumerate(control_points):
    plt.text(x, y, f'P{i}', fontsize=12, color='blue')

plt.title('Bezier Curve Fit to Points with Control Points')
plt.legend()
plt.grid(True)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
