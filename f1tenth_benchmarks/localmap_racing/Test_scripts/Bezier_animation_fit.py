import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import glob
from scipy.optimize import minimize
import re

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

def filter_and_add_point(data):
    # Find indices where x < 0
    negative_x_indices = np.where(data[:, 0] < 0)[0]
    
    # Check if there are any negative x-values at the beginning of the array
    if len(negative_x_indices) > 0 and negative_x_indices[0] == 0:
        # Determine the last index of consecutive negative x-values at the start
        end_neg_index = np.where(np.diff(negative_x_indices) != 1)[0]
        last_neg_index = negative_x_indices[end_neg_index[0]] if end_neg_index.size > 0 else negative_x_indices[-1]
        
        # Filter out only the negative x-values at the beginning
        filtered_data = data[last_neg_index + 1:]
        
        # Get the last removed point and the next valid point
        if filtered_data.size > 0:
            closest_point = data[last_neg_index]
            next_point = filtered_data[0]
            # Interpolate y between the last removed point and the next valid point
            y_interp = np.interp(0, [closest_point[0], next_point[0]], [closest_point[1], next_point[1]])
        else:
            # No valid points left, use the y-value of the last removed point
            y_interp = data[last_neg_index, 1]
        
        # Create the new point with interpolated y-value at x = 0
        new_point = np.array([[0, y_interp]])
        # Add the new point at the start of the filtered data
        filtered_data = np.vstack((new_point, filtered_data))
    
    else:
        # No negative x-values at the start, return the original data
        filtered_data = data
    
    return filtered_data

def numerical_sort(value):
    numbers = re.findall(r'\d+', value)
    return int(numbers[-1]) if numbers else float('inf')

# Modified plot_lines_animation function with Bezier fitting and control points plotting
def Bezier_animation():
    # Load the boundary files
    local_track_files = sorted(glob.glob("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/local_map_*.npy"), key=numerical_sort)
    right_files = sorted(glob.glob("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line1_*.npy"), key=numerical_sort)
    left_files = sorted(glob.glob("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line2_*.npy"), key=numerical_sort)

    # Preload the data
    local_data = [np.load(file) for file in local_track_files]
    left_data = [np.load(file) for file in left_files]
    right_data = [np.load(file) for file in right_files]

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot initial data to set up the plot objects
    local_line, = ax.plot([], [], 'o-', label='Local Track')
    left_line, = ax.plot([], [], 'o-', label='Left Line')
    right_line, = ax.plot([], [], 'o-', label='Right Line')
    
    # Bezier lines and control points
    local_bezier_line, = ax.plot([], [], 'r--', label='Local Bezier')
    left_bezier_line, = ax.plot([], [], 'g--', label='Left Bezier')
    right_bezier_line, = ax.plot([], [], 'b--', label='Right Bezier')
    control_points, = ax.plot([], [], 'kx', label='Control Points')

    # Set labels, title, legend, and grid
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Left and Right Line Coordinates Over Time')
    ax.legend()
    ax.grid(True)

    # Set fixed axis limits based on initial data range (adjust according to your data range)
    ax.set_xlim(-1, 17)  # Example limits, adjust according to your data range
    ax.set_ylim(-10, 10)  # Example limits, adjust according to your data range

    # Define the update function
    def update(frame):
        # Filter and load the data for this frame
        local_segment = filter_and_add_point(local_data[frame][:, :2])
        left_segment = filter_and_add_point(left_data[frame])
        right_segment = filter_and_add_point(right_data[frame])
        
        # Fit Bézier curves for local, left, and right data
        p0_local, p1_local, p2_local, p3_local = fit_bezier_to_lidar(local_segment)
        p0_left, p1_left, p2_left, p3_left = fit_bezier_to_lidar(left_segment)
        p0_right, p1_right, p2_right, p3_right = fit_bezier_to_lidar(right_segment)
        
        # Generate Bézier curves
        bezier_local = cubic_bezier(p0_local, p1_local, p2_local, p3_local)
        bezier_left = cubic_bezier(p0_left, p1_left, p2_left, p3_left)
        bezier_right = cubic_bezier(p0_right, p1_right, p2_right, p3_right)

        # Set the local, left, and right lines
        local_x, local_y = local_segment[:, 0], local_segment[:, 1]
        left_x, left_y = left_segment[:, 0], left_segment[:, 1]
        right_x, right_y = right_segment[:, 0], right_segment[:, 1]
        
        local_line.set_data(local_x, local_y)
        left_line.set_data(left_x, left_y)
        right_line.set_data(right_x, right_y)

        # Set the Bézier curves
        local_bezier_line.set_data(bezier_local[:, 0], bezier_local[:, 1])
        left_bezier_line.set_data(bezier_left[:, 0], bezier_left[:, 1])
        right_bezier_line.set_data(bezier_right[:, 0], bezier_right[:, 1])
        
        # Plot the control points for each curve
        control_x = [p0_local[0], p1_local[0], p2_local[0], p3_local[0], 
                     p0_left[0], p1_left[0], p2_left[0], p3_left[0],
                     p0_right[0], p1_right[0], p2_right[0], p3_right[0]]
        control_y = [p0_local[1], p1_local[1], p2_local[1], p3_local[1], 
                     p0_left[1], p1_left[1], p2_left[1], p3_left[1],
                     p0_right[1], p1_right[1], p2_right[1], p3_right[1]]
        control_points.set_data(control_x, control_y)

        # Update the title to indicate the current frame
        ax.set_title(f'Left and Right Line Coordinates (Frame {frame})')

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(left_data), repeat=False, interval=50)

    plt.show()

def main():
    Bezier_animation()

if __name__ == '__main__':
     main()