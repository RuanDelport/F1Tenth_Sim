import os
import numpy as np
import matplotlib.pyplot as plt
from f1tenth_benchmarks.utils.MapData import MapData
from f1tenth_benchmarks.utils.track_utils import CentreLine

# Ensure the directory exists
output_dir = f"Results"
os.makedirs(output_dir, exist_ok=True)
script_dir = os.path.dirname(os.path.abspath(__file__)) # Get the current script's directory
parent_dir = os.path.abspath(os.path.join(script_dir, "../../../")) # # Move three directories back
print(parent_dir)

map_name = "aut"
lap_number = 0

# Construct the file paths using the suffix and number
npy_file_path = os.path.join(parent_dir, f"Logs/FullStackPP/RawData_full_stack_pp/cf_estimates_{map_name}_{lap_number}.npy")
npy_file_path1 = os.path.join(parent_dir, f"Logs/FullStackPP/RawData_full_stack_pp/pf_estimates_{map_name}_{lap_number}.npy")
Timestep_file_path = os.path.join(parent_dir, f"Logs/FullStackPP/RawData_full_stack_pp/cf_steptimes_{map_name}_{lap_number}.npy")
Time_file_path1 = os.path.join(parent_dir, f"Logs/FullStackPP/RawData_full_stack_pp/pf_steptimes_{map_name}_{lap_number}.npy")

# # Print paths to verify correctness
# print("NPY File Path:", npy_file_path)
# print("NPY File Path 1:", npy_file_path1)
# print("Timestep File Path:", Timestep_file_path)
# print("Time File Path 1:", Time_file_path1)

# Load the NumPy file
data = np.load(npy_file_path)  # Assuming the data shape is (N, 3) where N is the number of samples
data1 = np.load(npy_file_path1)  
Timedata = np.load(Timestep_file_path)  
Timedata1 = np.load(Time_file_path1)  

map_data = MapData(map_name)
track = CentreLine(map_name)

track.widths = track.widths
l1 = track.path[:, :2] + track.nvecs * track.widths[:, 0][:, None]
l2 = track.path[:, :2] - track.nvecs * track.widths[:, 1][:, None]

# Extract x, y, and orientation (theta)
x = data[:, 0]
y = data[:, 1]
theta = data[:, 2]  # Orientation angle in radians
x1 = data1[:, 0]
y1 = data1[:, 1]
theta1 = data1[:, 2]  

#Prepare data for plotting with map image
orig_x = map_data.map_origin[0]
orig_y = map_data.map_origin[1]
map_resolution = map_data.map_resolution
l1 = track.path[:, :2] + track.nvecs * track.widths[:, 0][:, None]
l2 = track.path[:, :2] - track.nvecs * track.widths[:, 1][:, None]

startX = int((0-orig_x)/map_resolution)
startY = int((0-orig_y)/map_resolution)
scaled_x = (track.path[:, 0] - orig_x) / map_resolution
scaled_y = (track.path[:, 1] - orig_y) / map_resolution

scaled_L1_x = (l1[:, 0] - orig_x) / map_resolution
scaled_L1_y = (l1[:, 1] - orig_y) / map_resolution
scaled_L2_x = (l2[:, 0] - orig_x) / map_resolution
scaled_L2_y = (l2[:, 1] - orig_y) / map_resolution

particle_x = (x1 - orig_x) / map_resolution
particle_y = (y1 - orig_y) / map_resolution
curvature_x = (x - orig_x) / map_resolution
curvature_y = (y - orig_y) / map_resolution

# Path trajectory plot
plt.figure()
plt.plot(track.path[:, 0], track.path[:, 1], '--', linewidth=2, color='black', label='Track Centerline')
plt.plot(l1[:, 0], l1[:, 1], color='black', label='Track Boundary')
plt.plot(l2[:, 0], l2[:, 1], color='black')
plt.plot(x1, y1, label="Particle Filter Path")
plt.plot(x, y, label="Curvature Filter Path")
arrow_length = 0.1  # Adjust the length of the orientation arrows
for i in range(len(x)):
    plt.arrow(x[i], y[i], arrow_length * np.cos(theta[i]), arrow_length * np.sin(theta[i]), 
              head_width=0.05, head_length=0.1, fc='red', ec='red')
for i in range(len(x1)):
    plt.arrow(x1[i], y1[i], arrow_length * np.cos(theta1[i]), arrow_length * np.sin(theta1[i]), 
              head_width=0.05, head_length=0.1, fc='blue', ec='blue')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.title('Object Path with Orientation')
plt.legend(loc='upper right')
plt.grid(True)
plt.axis('equal')  # Ensure equal scaling on both axes
plt.show()

# Time iteration plot
plt.figure()
plt.plot(Timedata, label="Curvature Filter Time")
plt.plot( Timedata1, label="Particel Filter Time")
plt.xlabel('Iteration')
plt.ylabel('Time (s)')
# plt.ylim([0, 0.04])
plt.title('Time for each step')
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig(f"{output_dir}/{map_name}_Iteration_Time.svg")
plt.show()

# Map localisation plot
plt.figure( num=f'{map_name}_centreline')
plt.title(f"Localization results on {map_name}")
map_data.plot_map_img()
plt.plot(scaled_x, scaled_y, '--', linewidth=2, color='black', label = 'Centreline') # Plotting the track centerline
plt.plot(startX, startY, 'ro',label = "Starting Point") # Plotting the start point
plt.plot(particle_x, particle_y, '--', label="Particle Filter Path")
plt.plot(curvature_x, curvature_y, '--', label="Scan Match Filter Path")
plt.legend(loc='lower left')
plt.savefig(f"{output_dir}/{map_name}_localisation_comparison.svg")
plt.show()
