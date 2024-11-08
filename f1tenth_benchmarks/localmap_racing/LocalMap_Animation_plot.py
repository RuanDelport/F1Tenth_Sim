"""
The following code is used to plot the local map data that is used for mapless racing.
The algorithm predicts and extends the left and right boundaries of the track from local track data.
This data is stored in the the logs folder when the simulator is running and can be used to plot the boundaries afterwords.
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re
from matplotlib.animation import FuncAnimation
import matplotlib
import math
from scipy.optimize import curve_fit
import trajectory_planning_helpers as tph
from trajectory_planning_helpers.calc_head_curv_num import calc_head_curv_num
matplotlib.use('TkAgg')  # or another suitable backend like 'Qt5Agg', 'Agg', etc.
from scipy.optimize import minimize

#Plotting functions
# ------------------------------------------------------------------------------------------------------------------
# Custom sort function to sort filenames numerically
def numerical_sort(value):
    numbers = re.findall(r'\d+', value)
    return int(numbers[-1]) if numbers else float('inf')

def close_event(event):
    if event.key == 'escape':
        plt.close(event.canvas.figure)
        

# # Define a function to filter and add interpolated points
# def filter_and_add_point(data):
#     # Find points with negative x-values
#     negative_x_points = data[data[:, 0] < 0]
    
#     # Filter out points with negative x-values
#     filtered_data = data[data[:, 0] >= 0]
    
#     # If there are any negative x points, find the one closest to the origin
#     if negative_x_points.size > 0:
#         closest_point = negative_x_points[np.argmin(np.abs(negative_x_points[:, 0]))]
        
#         # If filtered_data has remaining points, interpolate y based on the next non-removed point
#         if filtered_data.size > 0:
#             next_point = filtered_data[0]
#             y_interp = np.interp(0, [closest_point[0], next_point[0]], [closest_point[1], next_point[1]])
#             new_point = np.array([[0, y_interp]])
#             filtered_data = np.vstack((new_point, filtered_data))
        
#         # If no non-removed points are left, add two interpolated points
#         else:
#             # The first point at x = 0
#             y_interp_1 = closest_point[1]
#             new_point_1 = np.array([[0, y_interp_1]])
            
#             # The second point further out on the line between the removed point and the origin
#             x_offset = closest_point[0] / 2  # Halfway to origin for spacing
#             y_interp_2 = np.interp(x_offset, [closest_point[0], 0], [closest_point[1], y_interp_1])
#             new_point_2 = np.array([[x_offset, y_interp_2]])
            
#             # Add both points at the beginning of the filtered data
#             filtered_data = np.vstack((new_point_1, new_point_2, filtered_data))
    
#     return filtered_data





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



def PrintDataArray(n):

    local_track = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/local_map_"+ str(n) +".npy")
    right_line = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line1_"+ str(n) +".npy")
    left_line = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line2_"+ str(n) +".npy")
    boundaries = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/boundaries_"+ str(n) +".npy")
    bound_extension = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/boundExtension_"+ str(n) +".npy")

    # Print the arrays
    print("Local Track Data:")
    print(local_track)

    print("\nLeft Line Data:")
    print(left_line)

    print("\nRight Line Data:")
    print(right_line)

    print("\nBoundaries Data:")
    print(boundaries)

    print("\nBoundary Extension Data:")
    print(bound_extension)

def plot_lines_once(n):
    local_track = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/local_map_"+ str(n) +".npy")
    right_line = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line1_"+ str(n) +".npy")
    left_line = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line2_"+ str(n) +".npy")
    boundaries = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/boundaries_"+ str(n) +".npy")
    bound_extension = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/boundExtension_"+ str(n) +".npy")

    try:
        scan_dir = f"/home/ruan/Documents/f1tenth_benchmarks/Logs/LocalMPCC/RawData_mu60/ScanLog_gbr_0.npy"
        scans = np.load(scan_dir)
        print("Scan data found")
    except:
        print("No scan data found")

    angles = np.linspace(-2.35619449615, 2.35619449615, 1080)
    coses = np.cos(angles)
    sines = np.sin(angles)

    scan_xs, scan_ys = scans[n+1] * np.array([coses, sines])


    # Extract x and y coordinates
    left_x, left_y = left_line[:, 0], left_line[:, 1]
    right_x, right_y = right_line[:, 0], right_line[:, 1]
    
    fig, ax = plt.subplots()

    # Plot the coordinates
    ax.plot(local_track[:, 0], local_track[:, 1], label=f'Local track ()', marker='o')
    ax.plot(right_x, right_y, label=f'Right Line ()', marker='o')
    ax.plot(left_x, left_y, label=f'Left Line ()', marker='o')
    
    # ax.plot(scan_xs, scan_ys, label=f'Scan Data', marker='o')

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Left and Right Line Coordinates Over Time for n = ' + str(n))
    ax.legend()
    ax.grid(True)

    # Set fixed axis limits based on initial data range (adjust according to your data range)
    # ax.set_xlim([-1, 20])  # Example limits, adjust according to your data range
    # ax.set_ylim([-20, 10])  # Example limits, adjust according to your data range
    ax.set_xlim([-1, 10])  # Example limits, adjust according to your data range
    ax.set_ylim([-5, 5])  # Example limits, adjust according to your data range

    fig.canvas.mpl_connect('key_press_event', close_event)
    plt.show()

def func(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def plot_Polyfit(n):
    right_line = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line1_"+ str(n) +".npy")
    left_line = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line2_"+ str(n) +".npy")
    
    # Extract x and y coordinates
    left_x, left_y = left_line[:, 0], left_line[:, 1]
    right_x, right_y = right_line[:, 0], right_line[:, 1]
    
    poptL, pcovL = curve_fit(func, left_x, left_y)
    poptR, pcovR = curve_fit(func, right_x, right_y)
    
    fig, ax = plt.subplots()

    # Plot the coordinates
    ax.plot(left_x, left_y, label=f'Left Line ()', marker='o')
    ax.plot(right_x, right_y, label=f'Right Line ()', marker='o')
    ax.plot(left_x, func(left_x, *poptL), 'r-', 
            label='Lfit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f' % tuple(poptL))
    ax.plot(right_x, func(right_x, *poptR), 'r-', 
            label='Rfit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f' % tuple(poptR))

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Left and Right Line Coordinates Over Time for n = ' + str(n))
    ax.legend()
    ax.grid(True)

    # Set fixed axis limits based on initial data range (adjust according to your data range)
    ax.set_xlim([-1, 20])  # Example limits, adjust according to your data range
    ax.set_ylim([-20, 10])  # Example limits, adjust according to your data range

    fig.canvas.mpl_connect('key_press_event', close_event)
    plt.show()
    
def plot_lines_and_curvature(n):
    local_track = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/local_map_"+ str(n) +".npy")
    right_line = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line1_"+ str(n) +".npy")
    left_line = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line2_"+ str(n) +".npy")
    boundaries = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/boundaries_"+ str(n) +".npy")
    bound_extension = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/boundExtension_"+ str(n) +".npy")

    try:
        scan_dir = f"/home/ruan/Documents/f1tenth_benchmarks/Logs/LocalMPCC/RawData_mu60/ScanLog_gbr_0.npy"
        scans = np.load(scan_dir)
        print("Scan data found")
    except:
        print("No scan data found")

    angles = np.linspace(-2.35619449615, 2.35619449615, 1080)
    coses = np.cos(angles)
    sines = np.sin(angles)
    
    # local_track = filter_and_add_point(local_track[:,:2])
    # left_line = filter_and_add_point(left_line)
    # right_line = filter_and_add_point(right_line)   
    
    # Extract x and y coordinates
    scan_xs, scan_ys = scans[n+1] * np.array([coses, sines])
    local_x, local_y = local_track[:, 0], local_track[:, 1]
    left_x, left_y = left_line[:, 0], left_line[:, 1]
    right_x, right_y = right_line[:, 0], right_line[:, 1]
    
    LocalLine = local_track[:, 0:2]
    
    # Ca;culate the curvature using cirlce through three points method
    radiiLocal, centersLocal, curvatureLocal = FeatureExtraction.calculate_circle_radius_and_center(LocalLine)
    radiiR, centersR, curvatureR = FeatureExtraction.calculate_circle_radius_and_center(right_line)
    radiiL, centersL, curvatureL = FeatureExtraction.calculate_circle_radius_and_center(left_line)

    #Calculate the curvature using the calc_head_curv_num function
    # Calculate element lengths (distances between consecutive points)
    pathL = np.vstack((left_x, left_y)).T
    pathR = np.vstack((right_x, right_y)).T
    el_lengthsLocal = np.sqrt(np.sum(np.diff(local_track, axis=0)**2, axis=1))
    el_lengthsL = np.sqrt(np.sum(np.diff(pathL, axis=0)**2, axis=1))
    el_lengthsR = np.sqrt(np.sum(np.diff(pathR, axis=0)**2, axis=1))
    # print(el_lengthsLocal)

    # Call the calc_head_curv_num function
    
    psiLocal, kappaLocal = calc_head_curv_num(
        path=local_track,
        el_lengths=el_lengthsLocal,
        is_closed=False,
        stepsize_psi_preview=0.1,
        stepsize_psi_review=0.1,
        stepsize_curv_preview=0.2,
        stepsize_curv_review=0.2,
        calc_curv=True
    )
    try:
        psiLocal, kappaLocal = tph.calc_head_curv_num.calc_head_curv_num(np.column_stack((local_track[:,1],local_track[:,0])), el_lengthsLocal, False)
        psiLocal = -psiLocal #Issue for some reason the psi values are negative
    except:
        print("Error in local calc_head_curv_num")
    
    try:
         psiL, kappaL = tph.calc_head_curv_num.calc_head_curv_num(np.column_stack((pathL[:,1],pathL[:,0])), el_lengthsL, False)
         psiL = -psiL
    except:
        print("Error in left calc_head_curv_num")
  
    try:
        psiR, kappaR = tph.calc_head_curv_num.calc_head_curv_num(np.column_stack((pathR[:,1],pathR[:,0])), el_lengthsR, False)
        psiR = -psiR
    except:
        print("Error in right calc_head_curv_num")
        
    
    # Visualization
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # Plot the coordinates
    axs[0].plot(left_x, left_y, label=f'Left Line ()', marker='o')
    axs[0].plot(right_x, right_y, label=f'Right Line ()', marker='o')
    axs[0].plot(local_x, local_y, label=f'Local Line ()', marker='o')
    # axs[0].plot(scan_xs, scan_ys, label=f'Scan Data', marker='o')
    axs[0].set_title('Left and Right Line Coordinates Over Time for n = ' + str(n))
    axs[0].set_xlabel('X Coordinate')
    axs[0].set_ylabel('Y Coordinate')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].axis('equal')

    # # Plot the heading (psi)
    # axs[1].plot(np.arange(len(psi)), psi, label='Heading (psi)')
    # axs[1].set_title('Heading (psi) along the Path')
    # axs[1].set_xlabel('Point Index')
    # axs[1].set_ylabel('Heading (radians)')
    # axs[1].legend()
    
    # Plot the curvature using the circle through three points method
    try:
        axs[1].plot(psiL, label='Left line kappa')
    except:
        print("Error in plotting left line")
    try:
        axs[1].plot(psiR, label='Right line kappa')
    except:
        print("Error in plotting right line")
    try:
        axs[1].plot(psiLocal, label='Local line kappa')
    except:
        print("Error in plotting local line")
    axs[1].set_title('Heading (psi) along the Path using calc_head_curv_num')
    axs[1].set_xlabel('Index')
    axs[1].set_ylabel('Heading')
    axs[1].legend()
    axs[1].grid(True)
    # axs[1].set_ylim([-2, 2])

    # Plot the curvature (kappa)
    try:
        axs[2].plot(np.arange(len(kappaL)), kappaL, label='Curvature (kappa) left')
    except:
        print("Error in plotting left curvature")
    try:
        axs[2].plot(np.arange(len(kappaR)), kappaR, label='Curvature (kappa) right')
    except:
        print("Error in plotting right curvature")
    try:
        axs[2].plot(np.arange(len(kappaLocal)), kappaLocal, label='Curvature (kappa) Local')
    except:
        print("Error in plotting local curvature")
    axs[2].set_title('Curvature (kappa) along the Path using calc_head_curv_num')
    axs[2].set_xlabel('Point Index')
    axs[2].set_ylabel('Curvature (1/m)')
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    fig.canvas.mpl_connect('key_press_event', close_event)
    plt.show()
    
def plot_Poly_and_curvature(n):
    local_track = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/local_map_"+ str(n) +".npy")
    right_line = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line1_"+ str(n) +".npy")
    left_line = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line2_"+ str(n) +".npy")
    boundaries = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/boundaries_"+ str(n) +".npy")
    bound_extension = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/boundExtension_"+ str(n) +".npy")

    try:
        scan_dir = f"/home/ruan/Documents/f1tenth_benchmarks/Logs/LocalMPCC/RawData_mu60/ScanLog_gbr_0.npy"
        scans = np.load(scan_dir)
        print("Scan data found")
    except:
        print("No scan data found")

    angles = np.linspace(-2.35619449615, 2.35619449615, 1080)
    coses = np.cos(angles)
    sines = np.sin(angles)
    
    # Extract x and y coordinates
    scan_xs, scan_ys = scans[n+1] * np.array([coses, sines])
    left_x, left_y = left_line[:, 0], left_line[:, 1]
    right_x, right_y = right_line[:, 0], right_line[:, 1]
    local_x, local_y = local_track[:, 0], local_track[:, 1]
    
    poptL, pcovL = curve_fit(func, left_x, left_y)
    poptR, pcovR = curve_fit(func, right_x, right_y)
    poptLocal, pcovLocal = curve_fit(func, local_x, local_y)
    left_y_new = func(left_x, *poptL)
    right_y_new = func(right_x, *poptR)
    local_y_new = func(local_x, *poptLocal)
    


    #Calculate the curvature using the calc_head_curv_num function
    # Calculate element lengths (distances between consecutive points)
    pathL = np.vstack((left_x, left_y_new)).T
    pathR = np.vstack((right_x, right_y_new)).T
    pathLocal = np.vstack((local_x, local_y_new)).T
    el_lengthsL = np.sqrt(np.sum(np.diff(pathL, axis=0)**2, axis=1))
    el_lengthsR = np.sqrt(np.sum(np.diff(pathR, axis=0)**2, axis=1))
    el_lengthsLocal = np.sqrt(np.sum(np.diff(pathLocal, axis=0)**2, axis=1))
    
    # Ca;culate the curvature using cirlce through three points method
    radiiR, centersR, curvatureR = FeatureExtraction.calculate_circle_radius_and_center(pathR)
    radiiL, centersL, curvatureL = FeatureExtraction.calculate_circle_radius_and_center(pathL)
    radiiLocal, centersLocal, curvatureLocal = FeatureExtraction.calculate_circle_radius_and_center(pathLocal)

    # Call the calc_head_curv_num function
    psi, kappaL = calc_head_curv_num(
        path=pathL,
        el_lengths=el_lengthsL,
        is_closed=False,
        stepsize_psi_preview=0.1,
        stepsize_psi_review=0.1,
        stepsize_curv_preview=0.2,
        stepsize_curv_review=0.2,
        calc_curv=True
    )
    psi, kappaR = calc_head_curv_num(
        path=pathR,
        el_lengths=el_lengthsR,
        is_closed=False,
        stepsize_psi_preview=0.1,
        stepsize_psi_review=0.1,
        stepsize_curv_preview=0.2,
        stepsize_curv_review=0.2,
        calc_curv=True
    )
    psiLocal, kappaLocal = calc_head_curv_num(
        path=pathLocal,
        el_lengths=el_lengthsLocal,
        is_closed=False,
        stepsize_psi_preview=0.1,
        stepsize_psi_review=0.1,
        stepsize_curv_preview=0.2,
        stepsize_curv_review=0.2,
        calc_curv=True
    )
    
    # Visualization
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # Plot the coordinates
    axs[0].plot(left_x, left_y, label=f'Left Line ()', marker='o')
    axs[0].plot(right_x, right_y, label=f'Right Line ()', marker='o')
    axs[0].plot(local_x, local_y, label=f'Local Line ()', marker='o')
    axs[0].plot(left_x, left_y_new, 'r-', 
            label='Lfit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f' % tuple(poptL))
    axs[0].plot(right_x, right_y_new, 'r-', 
            label='Rfit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f' % tuple(poptR))
    axs[0].plot(local_x, local_y_new, 'r-', 
            label='Rfit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f' % tuple(poptR))
    # axs[0].plot(scan_xs, scan_ys, label=f'Scan Data', marker='o')
    axs[0].set_title('Left and Right Line Coordinates Over Time for n = ' + str(n))
    axs[0].set_xlabel('X Coordinate')
    axs[0].set_ylabel('Y Coordinate')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].axis('equal')

    # # Plot the heading (psi)
    # axs[1].plot(np.arange(len(psi)), psi, label='Heading (psi)')
    # axs[1].set_title('Heading (psi) along the Path')
    # axs[1].set_xlabel('Point Index')
    # axs[1].set_ylabel('Heading (radians)')
    # axs[1].legend()
    
    # Plot the curvature using the circle through three points method
    axs[1].plot(curvatureL, label='Left line Curvature')
    axs[1].plot(curvatureR, label='Right line Curvature')
    axs[1].plot(curvatureLocal, label='Local line Curvature')
    axs[1].set_title('Curvature using circle through three points method')
    axs[1].set_xlabel('Index')
    axs[1].set_ylabel('Curvature')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_ylim([-1, 1])

    # Plot the curvature (kappa)
    axs[2].plot(np.arange(len(kappaL)), kappaL, label='Curvature (kappa) left')
    axs[2].plot(np.arange(len(kappaR)), kappaR, label='Curvature (kappa) right')
    axs[2].plot(np.arange(len(kappaLocal)), kappaLocal, label='Curvature (kappa) Local')
    axs[2].set_title('Curvature (kappa) along the Path using calc_head_curv_num')
    axs[2].set_xlabel('Point Index')
    axs[2].set_ylabel('Curvature (1/m)')
    axs[2].grid(True)
    axs[2].legend()
    axs[2].set_ylim([-1, 1])

    plt.tight_layout()
    fig.canvas.mpl_connect('key_press_event', close_event)
    plt.show()
    
def plot_boundaries_once(n):
    boundaries = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/boundaries_"+ str(n) +".npy")
    bound_extension = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/boundExtension_"+ str(n) +".npy")
    pass

def plot_boundaries_animation():

    # Load the boundary files
    boundaries = sorted(glob.glob("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/boundaries_*.npy"), key=numerical_sort)
    boundaries_ext = sorted(glob.glob("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/boundExtension_*.npy"), key=numerical_sort)

    # Preload the data
    Bound_data = [np.load(file) for file in boundaries]
    Bound_ext_data = [np.load(file) for file in boundaries_ext]

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot initial data to set up the plot objects
    left_line, = ax.plot([], [], 'o-', label='Left bound')
    right_line, = ax.plot([], [], 'o-', label='Right bound')
    left_ext, = ax.plot([], [], 'o-', label='Left ext')
    right_ext, = ax.plot([], [], 'o-', label='Right ext')

    # Set labels, title, legend, and grid
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Left and Right Line Coordinates Over Time')
    ax.legend()
    ax.grid(True)

    # Set fixed axis limits based on initial data range (adjust according to your data range)
    ax.set_xlim(-10, 17)  # Example limits, adjust according to your data range
    ax.set_ylim(-10, 10)  # Example limits, adjust according to your data range

    # Define the update function
    def update(frame):
        # Update data in plot objects
        right_x, right_y = Bound_data[frame][:, 0], Bound_data[frame][:, 1]
        left_x, left_y = Bound_data[frame][:, 2], Bound_data[frame][:, 3]

        # Check if Bound_ext_data[frame] is not empty
        if Bound_ext_data[frame].size > 0:
            right_ext_x, right_ext_y = Bound_ext_data[frame][:, 0], Bound_ext_data[frame][:, 1]
            left_ext_x, left_ext_y = Bound_ext_data[frame][:, 2], Bound_ext_data[frame][:, 3]
        else:
            left_ext_x, left_ext_y = [], []
            right_ext_x, right_ext_y = [], []

        left_line.set_data(left_x, left_y)
        right_line.set_data(right_x, right_y)
        left_ext.set_data(left_ext_x, left_ext_y)
        right_ext.set_data(right_ext_x, right_ext_y)

        # Update the title to indicate the current frame
        ax.set_title(f'Left and Right Line Coordinates (Frame {frame})')

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(Bound_data), repeat=False, interval=50)
    plt.show()

def plot_lines_animation():
  
    # Load the boundary files
    local_track_files = sorted(glob.glob("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/local_map_*.npy"), key=numerical_sort)
    right_files = sorted(glob.glob("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line1_*.npy"), key=numerical_sort)
    left_files = sorted(glob.glob("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line2_*.npy"), key=numerical_sort)

    # # Preload the data
    local_data = [np.load(file) for file in local_track_files]
    left_data = [np.load(file) for file in left_files]
    right_data = [np.load(file) for file in right_files]

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))

    # # Plot initial data to set up the plot objects
    local_line, = ax.plot([], [], 'o-', label='Local Track')
    left_line, = ax.plot([], [], 'o-', label='Left Line')
    right_line, = ax.plot([], [], 'o-', label='Right Line')

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
        # Update data in plot objects
        
        local_data[frame] = filter_and_add_point(local_data[frame][:,:2])
        left_data[frame] = filter_and_add_point(left_data[frame])
        right_data[frame] = filter_and_add_point(right_data[frame])   
        
        local_x, local_y = local_data[frame][:, 0], local_data[frame][:, 1]
        left_x, left_y = left_data[frame][:, 0], left_data[frame][:, 1]
        right_x, right_y = right_data[frame][:, 0], right_data[frame][:, 1]
        
        local_line.set_data(local_x, local_y)
        left_line.set_data(left_x, left_y)
        right_line.set_data(right_x, right_y)

        # Update the title to indicate the current frame
        ax.set_title(f'Left and Right Line Coordinates (Frame {frame})')

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(left_data), repeat=False, interval=50)

    plt.show()

def plot_lines_animation_with_polyfit():
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
    left_line, = ax.plot([], [], 'o-', label='Left Line')
    right_line, = ax.plot([], [], 'o-', label='Right Line')
    local_line, = ax.plot([], [], 'o-', label='Local Track')
    left_poly, = ax.plot([], [], 'r-', label='Left Polyfit')
    right_poly, = ax.plot([], [], 'g-', label='Right Polyfit')
    local_poly, = ax.plot([], [], 'b-', label='Local Polyfit')

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
        
        # local_data[frame] = filter_and_add_point(local_data[frame][:,:2])
        # left_data[frame] = filter_and_add_point(left_data[frame])
        # right_data[frame] = filter_and_add_point(right_data[frame])   
        
        # Update data in plot objects
        left_x, left_y = left_data[frame][:, 0], left_data[frame][:, 1]
        right_x, right_y = right_data[frame][:, 0], right_data[frame][:, 1]
        local_x, local_y = local_data[frame][:, 0], local_data[frame][:, 1]
        
        # Update line data
        left_line.set_data(left_x, left_y)
        right_line.set_data(right_x, right_y)
        local_line.set_data(local_x, local_y)

        # Fit polynomials
        try:
            poptL, _ = curve_fit(func, left_x, left_y)
            poptR, _ = curve_fit(func, right_x, right_y)
            poptLocal, _ = curve_fit(func, local_x, local_y)
            left_poly.set_data(left_x, func(left_x, *poptL))
            right_poly.set_data(right_x, func(right_x, *poptR))
            local_poly.set_data(local_x, func(local_x, *poptLocal))
        except Exception as e:
            print(f"Error fitting polynomials: {e}")
            left_poly.set_data([], [])
            right_poly.set_data([], [])

        # Update the title to indicate the current frame
        ax.set_title(f'Left and Right Line Coordinates (Frame {frame})')

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(left_data), repeat=False, interval=50)

    plt.show()

def plot_lines_and_curvature_animation():
    
     # Load the boundary files
    local_track_files = sorted(glob.glob("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/local_map_*.npy"), key=numerical_sort)
    right_files = sorted(glob.glob("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line1_*.npy"), key=numerical_sort)
    left_files = sorted(glob.glob("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line2_*.npy"), key=numerical_sort)
    try:
        scan_files = sorted(glob.glob("/home/ruan/Documents/f1tenth_benchmarks/Logs/LocalMPCC/RawData_mu60/ScanLog_gbr_*.npy"), key=numerical_sort)
        print("Scan data found")
    except:
        print("No scan data found")

    angles = np.linspace(-2.35619449615, 2.35619449615, 1080)
    coses = np.cos(angles)
    sines = np.sin(angles)

    # # Preload the data
    local_track_data = [np.load(file) for file in local_track_files]
    left_data = [np.load(file) for file in left_files]
    right_data = [np.load(file) for file in right_files]
    scan_data = [np.load(file) for file in scan_files]

    # Create the figure and axes
    # fig, ax = plt.subplots(figsize=(12, 8))
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # # Plot initial data to set up the plot objects
    left_line, = axs[0].plot([], [], 'o-', label='Left Line')
    right_line, = axs[0].plot([], [], 'o-', label='Right Line')
    local_track_line, = axs[0].plot([], [], 'o-', label='Local Track')
    ThreeP_curveL, = axs[1].plot([],[], label='Curvature Left')
    ThreeP_curveR, = axs[1].plot([],[], label='Curvature Right')
    ThreeP_curveLocal, = axs[1].plot([],[], label='Curvature Local')
    Kappa_curveL, = axs[2].plot([],[], label='Curvature (kappa) Left')
    Kappa_curveR, = axs[2].plot([], [],label='Curvature (kappa) Right')
    Kappa_curveLocal, = axs[2].plot([], [],label='Curvature (kappa) Local')
    
     # Plot the coordinates
    axs[0].set_xlabel('X Coordinate')
    axs[0].set_ylabel('Y Coordinate')
    axs[0].legend()
    axs[0].grid(True)
     # Set fixed axis limits based on initial data range (adjust according to your data range)
    axs[0].set_xlim([-1, 20])  # Example limits, adjust according to your data range
    axs[0].set_ylim([-10, 10])  # Example limits, adjust according to your data range
    # axs[0].axis('equal')

    # Plot the curvature using the circle through three points method
    axs[1].set_title('Curvature using circle through three points method')
    axs[1].set_xlabel('Index')
    axs[1].set_ylabel('Curvature')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_ylim([-1, 1])
    axs[1].set_xlim([0, 50])  # Example limits, adjust according to your data range

    # Plot the curvature (kappa)
    axs[2].set_title('Curvature (kappa) along the Path using calc_head_curv_num')
    axs[2].set_xlabel('Point Index')
    axs[2].set_ylabel('Curvature (1/m)')
    axs[2].grid(True)
    axs[2].legend()
    axs[2].set_ylim([-1, 1])
    axs[2].set_xlim([0, 50])  # Example limits, adjust according to your data range

    plt.tight_layout()
    fig.canvas.mpl_connect('key_press_event', close_event)

    # Define the update function
    def update(frame):
        # Update data in plot objects
        left_x, left_y = left_data[frame][:, 0], left_data[frame][:, 1]
        right_x, right_y = right_data[frame][:, 0], right_data[frame][:, 1]
        local_x, local_y = local_track_data[frame][:, 0], local_track_data[frame][:, 1]
        pathL = np.vstack((left_x, left_y)).T
        pathR = np.vstack((right_x, right_y)).T
        pathLocal = np.vstack((local_x, local_y)).T
        # LocalLine = local_track_data[frame][:, 0:2]
        
        # Calculate the curvature using cirlce through three points method
        radiiR, centersR, curvatureR = FeatureExtraction.calculate_circle_radius_and_center(pathR)
        radiiL, centersL, curvatureL = FeatureExtraction.calculate_circle_radius_and_center(pathL)
        radiiLocal, centersLocal, curvatureLocal = FeatureExtraction.calculate_circle_radius_and_center(pathLocal)

        #Calculate the curvature using the calc_head_curv_num function
        # Calculate element lengths (distances between consecutive points)
        el_lengthsL = np.sqrt(np.sum(np.diff(pathL, axis=0)**2, axis=1))
        el_lengthsR = np.sqrt(np.sum(np.diff(pathR, axis=0)**2, axis=1))
        el_lengthsLocal = np.sqrt(np.sum(np.diff(pathLocal, axis=0)**2, axis=1))

        # Call the calc_head_curv_num function
        psi, kappaL = calc_head_curv_num(
            path=pathL,
            el_lengths=el_lengthsL,
            is_closed=False,
            stepsize_psi_preview=0.1,
            stepsize_psi_review=0.1,
            stepsize_curv_preview=0.2,
            stepsize_curv_review=0.2,
            calc_curv=True
        )
        psi, kappaR = calc_head_curv_num(
            path=pathR,
            el_lengths=el_lengthsR,
            is_closed=False,
            stepsize_psi_preview=0.1,
            stepsize_psi_review=0.1,
            stepsize_curv_preview=0.2,
            stepsize_curv_review=0.2,
            calc_curv=True
        )
        psiLocal, kappaLocal = calc_head_curv_num(
            path=pathLocal,
            el_lengths=el_lengthsLocal,
            is_closed=False,
            stepsize_psi_preview=0.1,
            stepsize_psi_review=0.1,
            stepsize_curv_preview=0.2,
            stepsize_curv_review=0.2,
            calc_curv=True
        )      
        
        left_line.set_data(left_x, left_y)
        right_line.set_data(right_x, right_y)
        local_track_line.set_data(local_x, local_y)
        ThreeP_curveL.set_data(np.arange(len(curvatureL)), curvatureL)
        ThreeP_curveR.set_data(np.arange(len(curvatureR)),curvatureR)
        ThreeP_curveLocal.set_data(np.arange(len(curvatureLocal)),curvatureLocal)
        Kappa_curveL.set_data(np.arange(len(kappaL)), kappaL)
        Kappa_curveR.set_data(np.arange(len(kappaR)), kappaR)
        Kappa_curveLocal.set_data(np.arange(len(kappaLocal)), kappaLocal)

        # Update the title to indicate the current frame
        axs[0].set_title(f'Left and Right Line Coordinates (Frame {frame})')

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(left_data), repeat=False, interval=100)

    plt.tight_layout()
    fig.canvas.mpl_connect('key_press_event', close_event)
    plt.show()

def plot_Poly_and_curvature_animation():
    
     # Load the boundary files
    local_track_files = sorted(glob.glob("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/local_map_*.npy"), key=numerical_sort)
    right_files = sorted(glob.glob("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line1_*.npy"), key=numerical_sort)
    left_files = sorted(glob.glob("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line2_*.npy"), key=numerical_sort)
    try:
        scan_files = sorted(glob.glob("/home/ruan/Documents/f1tenth_benchmarks/Logs/LocalMPCC/RawData_mu60/ScanLog_gbr_*.npy"), key=numerical_sort)
        print("Scan data found")
    except:
        print("No scan data found")

    angles = np.linspace(-2.35619449615, 2.35619449615, 1080)
    coses = np.cos(angles)
    sines = np.sin(angles)

    # # Preload the data
    local_track_data = [np.load(file) for file in local_track_files]
    left_data = [np.load(file) for file in left_files]
    right_data = [np.load(file) for file in right_files]
    scan_data = [np.load(file) for file in scan_files]

    # Create the figure and axes
    # fig, ax = plt.subplots(figsize=(12, 8))
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # # Plot initial data to set up the plot objects
    left_line, = axs[0].plot([], [], 'o-', label='Left Line')
    right_line, = axs[0].plot([], [], 'o-', label='Right Line')
    local_track_line, = axs[0].plot([], [], 'o-', label='Local Track')
    left_poly, = axs[0].plot([], [], 'r-', label='Left Polyfit')
    right_poly, = axs[0].plot([], [], 'g-', label='Right Polyfit')
    local_poly, = axs[0].plot([], [], 'b-', label='Local Polyfit')
    ThreeP_curveL, = axs[1].plot([],[], label='Curvature Left')
    ThreeP_curveR, = axs[1].plot([],[], label='Curvature Right')
    ThreeP_curveLocal, = axs[1].plot([],[], label='Curvature Local')
    Kappa_curveL, = axs[2].plot([],[], label='Curvature (kappa) Left')
    Kappa_curveR, = axs[2].plot([], [],label='Curvature (kappa) Right')
    Kappa_curveLocal, = axs[2].plot([], [],label='Curvature (kappa) Local')
    
     # Plot the coordinates
    axs[0].set_xlabel('X Coordinate')
    axs[0].set_ylabel('Y Coordinate')
    axs[0].legend()
    axs[0].grid(True)
     # Set fixed axis limits based on initial data range (adjust according to your data range)
    axs[0].set_xlim([-1, 20])  # Example limits, adjust according to your data range
    axs[0].set_ylim([-10, 10])  # Example limits, adjust according to your data range
    # axs[0].axis('equal')

    # Plot the curvature using the circle through three points method
    axs[1].set_title('Curvature using circle through three points method')
    axs[1].set_xlabel('Index')
    axs[1].set_ylabel('Curvature')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_ylim([-1, 1])
    axs[1].set_xlim([0, 50])  # Example limits, adjust according to your data range

    # Plot the curvature (kappa)
    axs[2].set_title('Curvature (kappa) along the Path using calc_head_curv_num')
    axs[2].set_xlabel('Point Index')
    axs[2].set_ylabel('Curvature (1/m)')
    axs[2].grid(True)
    axs[2].legend()
    axs[2].set_ylim([-1, 1])
    axs[2].set_xlim([0, 50])  # Example limits, adjust according to your data range

    plt.tight_layout()
    fig.canvas.mpl_connect('key_press_event', close_event)

    # Define the update function
    def update(frame):
        # Update data in plot objects
        left_x, left_y = left_data[frame][:, 0], left_data[frame][:, 1]
        right_x, right_y = right_data[frame][:, 0], right_data[frame][:, 1]
        local_x, local_y = local_track_data[frame][:, 0], local_track_data[frame][:, 1]
        
         # Fit polynomials
        try:
            poptL, _ = curve_fit(func, left_x, left_y)
            poptR, _ = curve_fit(func, right_x, right_y)
            poptLocal, _ = curve_fit(func, local_x, local_y)
            left_y_new = func(left_x, *poptL)
            right_y_new = func(right_x, *poptR)
            local_y_new = func(local_x, *poptLocal)
            left_poly.set_data(left_x, left_y_new)
            right_poly.set_data(right_x, right_y_new)
            local_poly.set_data(local_x, local_y_new )
        except Exception as e:
            print(f"Error fitting polynomials: {e}")
            left_poly.set_data([], [])
            right_poly.set_data([], [])
            
        pathL = np.vstack((left_x, left_y_new)).T
        pathR = np.vstack((right_x, right_y_new)).T
        pathLocal = np.vstack((local_x, local_y_new)).T
        # LocalLine = local_track_data[frame][:, 0:2]
        
       
        
        # Calculate the curvature using cirlce through three points method
        radiiR, centersR, curvatureR = FeatureExtraction.calculate_circle_radius_and_center(pathR)
        radiiL, centersL, curvatureL = FeatureExtraction.calculate_circle_radius_and_center(pathL)
        radiiLocal, centersLocal, curvatureLocal = FeatureExtraction.calculate_circle_radius_and_center(pathLocal)

        #Calculate the curvature using the calc_head_curv_num function
        # Calculate element lengths (distances between consecutive points)
        el_lengthsL = np.sqrt(np.sum(np.diff(pathL, axis=0)**2, axis=1))
        el_lengthsR = np.sqrt(np.sum(np.diff(pathR, axis=0)**2, axis=1))
        el_lengthsLocal = np.sqrt(np.sum(np.diff(pathLocal, axis=0)**2, axis=1))

        # Call the calc_head_curv_num function
        psi, kappaL = calc_head_curv_num(
            path=pathL,
            el_lengths=el_lengthsL,
            is_closed=False,
            stepsize_psi_preview=0.1,
            stepsize_psi_review=0.1,
            stepsize_curv_preview=0.2,
            stepsize_curv_review=0.2,
            calc_curv=True
        )
        psi, kappaR = calc_head_curv_num(
            path=pathR,
            el_lengths=el_lengthsR,
            is_closed=False,
            stepsize_psi_preview=0.1,
            stepsize_psi_review=0.1,
            stepsize_curv_preview=0.2,
            stepsize_curv_review=0.2,
            calc_curv=True
        )
        psiLocal, kappaLocal = calc_head_curv_num(
            path=pathLocal,
            el_lengths=el_lengthsLocal,
            is_closed=False,
            stepsize_psi_preview=0.1,
            stepsize_psi_review=0.1,
            stepsize_curv_preview=0.2,
            stepsize_curv_review=0.2,
            calc_curv=True
        )      
        
        left_line.set_data(left_x, left_y)
        right_line.set_data(right_x, right_y)
        local_track_line.set_data(local_x, local_y)
        ThreeP_curveL.set_data(np.arange(len(curvatureL)), curvatureL)
        ThreeP_curveR.set_data(np.arange(len(curvatureR)),curvatureR)
        ThreeP_curveLocal.set_data(np.arange(len(curvatureLocal)),curvatureLocal)
        Kappa_curveL.set_data(np.arange(len(kappaL)), kappaL)
        Kappa_curveR.set_data(np.arange(len(kappaR)), kappaR)
        Kappa_curveLocal.set_data(np.arange(len(kappaLocal)), kappaLocal)
        
        # Update the title to indicate the current frame
        axs[0].set_title(f'Left and Right Line Coordinates (Frame {frame})')

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(left_data), repeat=False, interval=100)

    plt.tight_layout()
    fig.canvas.mpl_connect('key_press_event', close_event)
    plt.show()  
   
def plot_points_and_circles_together(n):
    right_line_data = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line1_" + str(n) + ".npy")
    radii, centers, curvature = FeatureExtraction.calculate_circle_radius_and_center(right_line_data)

    right_x, right_y = right_line_data[:, 0], right_line_data[:, 1]
    print(right_x)
    print(right_y)
    
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot initial data to set up the plot objects
    right_line_plot, = ax.plot([], [], 'o-', label='Right Line')

    # Set labels, title, legend, and grid
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Right Line Coordinates and Curvature Over Time')
    ax.legend()
    ax.grid(True)

    ax.set_xlim(-15, 15)  # Example limits, adjust according to your data range
    ax.set_ylim(-15, 15)
    # Set equal scaling
    # ax.set_aspect('equal', 'box')

    # Define the update function
    def update(frame):
        # Update data in plot objects
        right_line_plot.set_data(right_x, right_y)

        # Update the title to indicate the current frame
        ax.set_title(f'Right Line Coordinates (Frame {frame})')

        radius, (x_center, y_center) = radii[frame], centers[frame]

        circle = plt.Circle((x_center, y_center), radius, color='red', fill=False, label='Circle')
        ax.add_patch(circle)

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(curvature), repeat=False, interval=200)
    fig.canvas.mpl_connect('key_press_event', close_event)

    plt.show()

#------------------------------------------------------------------------------------------------------------------

class FeatureExtraction:
    def __init__(self, path):
        #self.radii = np.array([])
        #self.centers = np.array([])
        self.radii = []
        self.centers = []
        
    def getCurvature(self):
        pass
    
    def calculate_circle_radius_and_center(points):
        # if len(points) < 3:
        #     raise ValueError("At least three points are required to form a circle.")
        
        radii = []
        centers = []
        curvatures = []
        
        if len(points) < 3:
            radii.append(1)
            centers.append((0, 0))
            curvatures.append(0)
            return radii, centers, curvatures
        for i in range(len(points) - 2):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            x3, y3 = points[i + 2]

            # Calculate the determinant (related to twice the area of the triangle)
            A = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
            
            if A == 0:
                raise ValueError("The points are collinear, so a unique circle cannot be formed.")

            # Calculate the squared lengths of the sides of the triangle
            a_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
            b_sq = (x3 - x2) ** 2 + (y3 - y2) ** 2
            c_sq = (x1 - x3) ** 2 + (y1 - y3) ** 2

            # Calculate the circumradius using the correct formula
            a = math.sqrt(a_sq)
            b = math.sqrt(b_sq)
            c = math.sqrt(c_sq)
            
            # Area of the triangle (using determinant method)
            area = abs(A) / 2

            # Circumradius
            radius = (a * b * c) / (4 * area)

            # Calculate the circumcenter coordinates
            x_center = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / (2 * A)
            y_center = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / (2 * A)

            # Determine the sign of curvature based on the orientation of the triangle
            # Sign of the area A will determine the direction of the turn:
            # A > 0: counterclockwise (positive curvature)
            # A < 0: clockwise (negative curvature)
            curvature = 1.0 / radius
            curvature = curvature if A > 0 else -curvature

            radii.append(radius)
            centers.append((x_center, y_center))
            curvatures.append(curvature)

        return np.array(radii), np.array(centers), np.array(curvatures)

def plot_points_and_circles(points, radii, centers):
    fig, ax = plt.subplots()
    
    # Plot the points
    ax.scatter(points[:, 0], points[:, 1], color='blue', label='Points')
    
    # Plot the circles
    for radius, (x_center, y_center) in zip(radii, centers):
        circle = plt.Circle((x_center, y_center), radius, color='red', fill=False, label='Circle')
        ax.add_patch(circle)
    
    # Set equal scaling
    ax.set_aspect('equal', 'box')
    
    # Add labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    
    # Show plot
    plt.show()

def plot_curvature(x_list, y_list, heading_list, curvature,
                   k=0.01, c="-c", label="Curvature"):
    """
    Plot curvature on 2D path. This plot is a line from the original path,
    the lateral distance from the original path shows curvature magnitude.
    Left turning shows right side plot, right turning shows left side plot.
    For straight path, the curvature plot will be on the path, because
    curvature is 0 on the straight path.

    Parameters
    ----------
    x_list : array_like
        x position list of the path
    y_list : array_like
        y position list of the path
    heading_list : array_like
        heading list of the path
    curvature : array_like
        curvature list of the path
    k : float
        curvature scale factor to calculate distance from the original path
    c : string
        color of the plot
    label : string
        label of the plot
    """
    cx = [x + d * k * np.cos(yaw - np.pi / 2.0) for x, y, yaw, d in
          zip(x_list, y_list, heading_list, curvature)]
    cy = [y + d * k * np.sin(yaw - np.pi / 2.0) for x, y, yaw, d in
          zip(x_list, y_list, heading_list, curvature)]

    plt.plot(cx, cy, c, label=label)
    for ix, iy, icx, icy in zip(x_list, y_list, cx, cy):
        plt.plot([ix, icx], [iy, icy], c)
    

def main():
    
    points = np.array([
    [-1.13979656, -1.1504081],
    [-0.88752856, -1.08038959],
    [-0.64219188, -1.05320218],
    [-0.38959808, -1.0280293],
    [-0.13795937, -0.96584606],
    [0.11108701, -0.96568564]
    ])

    # n = 200
    # n = 400
    # n = 85
    # n = 240
    # n = 415 # Good example
    # n = 395 # Very noisy
    # n = 140 #Nice example
    # n = 600
    # n = 40
    n =267 # messed up centre line on aut
    # n =345 # messed up centre line on esp
    # n =68 # messed up centre line on gbr
    # n = 13
    # n =580 #edge case gbr
    # n =255
    
    # n = 115 #Edgecase aut
    right_line = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line1_"+ str(n) +".npy")
    
    
    radii, centers, curvature = FeatureExtraction.calculate_circle_radius_and_center(right_line)
    # print("Radii:", radii)
    # print("Centers:", centers)
    # print("Curvature:", len(curvature))
    # print("Right Line:", len(right_line))

    # PrintDataArray(n)
    # plot_lines_once(n)
    # plot_Polyfit(n)
    plot_lines_and_curvature(n)
    # plot_Poly_and_curvature(n)
    # plot_boundaries_once(n)
    # plot_boundaries_animation()
    # plot_lines_animation()
    # Bezier_animation()
    # plot_lines_animation_with_polyfit()
    # plot_lines_and_curvature_animation()
    # plot_Poly_and_curvature_animation()
    # plot_points_and_circles(right_line, radii, centers)
    # plot_curvature(curvature, right_line)  #Not working
    # plot_points_and_circles_together(n)
    
    # # Sample path data
    # x_list, y_list = right_line[:, 0], right_line[:, 1]
  

    # # Simulate heading (in radians)
    # heading_list = np.arctan2(np.gradient(y_list), np.gradient(x_list))

    # # Simulate curvature
    # curvature = np.gradient(heading_list) / np.gradient(np.sqrt(np.gradient(x_list)**2 + np.gradient(y_list)**2))

    # # Plot the original path
    # plt.plot(x_list, y_list, label="Original Path")

    # # Plot the curvature using the function
    # plot_curvature(x_list, y_list, heading_list, curvature, k=0.001, c="r-", label="Curvature Plot")

    # # Show the plot
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.title("Curvature Plot Example")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

if __name__ == '__main__':
     main()