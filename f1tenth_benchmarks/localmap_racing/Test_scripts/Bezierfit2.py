import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Function to calculate a cubic Bezier curve from four control points
def cubic_bezier(p0, p1, p2, p3, num_points=100):
    t = np.linspace(0, 1, num_points)[:, None]  # Reshape to (num_points, 1) for broadcasting
    bezier = (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3
    return bezier

# Loss function to minimize the distance between the lidar points and the Bezier curve
def bezier_loss(control_points, lidar_points, p0, p3):
    # Reshape control points array to extract p1 and p2
    p1, p2 = np.array(control_points[:2]), np.array(control_points[2:])
    
    # Generate the Bezier curve for current control points
    bezier = cubic_bezier(p0, p1, p2, p3, num_points=len(lidar_points))
    
    # Calculate the squared distance between lidar points and Bezier points
    loss = np.sum(np.linalg.norm(bezier - lidar_points, axis=1)**2)
    return loss

# Function to fit a Bezier curve to lidar points
def fit_bezier_to_lidar(lidar_segment):
    p0, p3 = lidar_segment[0], lidar_segment[-1]  # Endpoints of the curve

    # Initial guess for control points p1 and p2 (midway between p0 and p3)
    initial_guess = np.concatenate([p0, p3])  # Start with simple midpoint assumption

    # Minimize the loss function to find the best control points p1 and p2
    result = minimize(bezier_loss, initial_guess, args=(lidar_segment, p0, p3), method='BFGS')

    # Extract the optimized control points
    p1, p2 = np.array(result.x[:2]), np.array(result.x[2:])
    
    return p0, p1, p2, p3  # Return all control points

# Generate lidar points simulating a racetrack with a straight and curved section
def generate_racetrack_lidar():
    # Straight section (e.g., y=0, x from 0 to 10)
    straight_x = np.linspace(0, 10, 50)
    straight_y = np.zeros_like(straight_x)

    # Curved section (e.g., a quarter circle arc)
    theta = np.linspace(0, np.pi/2, 50)  # Quarter circle
    curve_x = 10 + 5 * np.cos(theta)  # Radius 5
    curve_y = 5 * np.sin(theta)

    # Combine the straight and curved sections
    lidar_x = np.concatenate((straight_x, curve_x))
    lidar_y = np.concatenate((straight_y, curve_y))

    return lidar_x, lidar_y

# Segment lidar points into straight and curved sections
def segment_lidar_points(lidar_x, lidar_y):
    # Define the cutoff where the curve starts (for simplicity, using index)
    cutoff_index = 50  # This separates straight from curved (first 50 points are straight)
    
    # Segment the lidar points into straight and curved
    straight_segment = np.column_stack((lidar_x[:cutoff_index], lidar_y[:cutoff_index]))
    curved_segment = np.column_stack((lidar_x[cutoff_index:], lidar_y[cutoff_index:]))
    
    return straight_segment, curved_segment

# Main function
def main():
    # Generate lidar points for a simulated racetrack
    lidar_x, lidar_y = generate_racetrack_lidar()

    # Segment the lidar points into straight and curved sections
    straight_segment, curved_segment = segment_lidar_points(lidar_x, lidar_y)

    # Fit a cubic Bezier curve to the straight segment
    p0_straight, p1_straight, p2_straight, p3_straight = fit_bezier_to_lidar(straight_segment)
    bezier_straight = cubic_bezier(p0_straight, p1_straight, p2_straight, p3_straight)

    # Fit a cubic Bezier curve to the curved segment
    p0_curve, p1_curve, p2_curve, p3_curve = fit_bezier_to_lidar(curved_segment)
    bezier_curve = cubic_bezier(p0_curve, p1_curve, p2_curve, p3_curve)

    # Plotting the lidar points and the Bezier curves
    plt.figure(figsize=(10, 6))

    # Plot original lidar points
    plt.plot(lidar_x, lidar_y, 'ro', label='Lidar Points', markersize=5)

    # Plot straight segment
    plt.plot(bezier_straight[:, 0], bezier_straight[:, 1], 'b-', label='Bezier Straight Curve', linewidth=2)
    plt.plot(straight_segment[:, 0], straight_segment[:, 1], 'bo-', label='Straight Segment', markersize=5)

    # Plot curved segment
    plt.plot(bezier_curve[:, 0], bezier_curve[:, 1], 'g-', label='Bezier Curved Curve', linewidth=2)
    plt.plot(curved_segment[:, 0], curved_segment[:, 1], 'go-', label='Curved Segment', markersize=5)

    # Plot control points for both Bezier curves
    control_points_straight = np.array([p0_straight, p1_straight, p2_straight, p3_straight])
    control_points_curve = np.array([p0_curve, p1_curve, p2_curve, p3_curve])

    plt.plot(control_points_straight[:, 0], control_points_straight[:, 1], 'kx-', label='Control Points Straight', markersize=8)
    plt.plot(control_points_curve[:, 0], control_points_curve[:, 1], 'kx-', label='Control Points Curve', markersize=8)

    plt.title('Cubic Bezier Curves Fitted to Straight and Curved Segments')
    plt.legend()
    plt.grid(True)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    main()
