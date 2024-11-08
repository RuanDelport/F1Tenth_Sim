import matplotlib.pyplot as plt
import numpy as np

# Define initial square points
square = np.array([
    [-1, -1],
    [1, -1],
    [1, 1],
    [-1, 1],
    [-1, -1]  # Close the square
])

# Define transformation matrices
def stretch_horizontal(factor):
    return np.array([
        [factor, 0],
        [0, 1]
    ])

def rotate(theta_deg):
    theta_rad = np.radians(theta_deg)
    return np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad), np.cos(theta_rad)]
    ])

def reflect_y_axis():
    return np.array([
        [-1, 0],
        [0, 1]
    ])

# Apply transformations in sequence
transformations = [
    stretch_horizontal(2),
    rotate(-45),
    stretch_horizontal(1/3),
    reflect_y_axis()
]

# Plot each transformation step
plt.figure(figsize=(10, 8))
ax = plt.gca()
ax.set_aspect('equal')
plt.axhline(0, color='grey', lw=0.5)
plt.axvline(0, color='grey', lw=0.5)

colors = ['b', 'g', 'r', 'm', 'c']
label_text = ["Original"]

# Plot original square
x, y = square[:, 0], square[:, 1]
plt.plot(x, y, label=label_text[0], color=colors[0])

# Apply and plot each transformation step
current_shape = square
for i, transformation in enumerate(transformations):
    current_shape = np.dot(current_shape, transformation.T)
    x, y = current_shape[:, 0], current_shape[:, 1]
    label_text.append(f"Step {i+1}")
    plt.plot(x, y, label=label_text[-1], color=colors[(i+1) % len(colors)])

# Final plot adjustments
plt.title('Transformation Steps and Final Combined Transformation')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()
