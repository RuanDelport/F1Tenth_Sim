import numpy as np
import yaml
from numba import njit 
from PIL import Image
import os 
from scipy.ndimage import distance_transform_edt as edt
import matplotlib.pyplot as plt

from f1tenth_benchmarks.utils.BasePlanner import load_parameter_file_with_extras
from f1tenth_benchmarks.utils.track_utils import CentreLine
from scipy import interpolate
from trajectory_planning_helpers.calc_head_curv_num import calc_head_curv_num
from f1tenth_benchmarks.localmap_racing.LocalMapGenerator import LocalMapGenerator
from f1tenth_benchmarks.utils.MapData import MapData
from scipy.spatial import KDTree
from scipy.optimize import minimize
import time
from f1tenth_benchmarks.utils.MapData import MapData
from matplotlib.patches import Ellipse
import trajectory_planning_helpers as tph
from f1tenth_benchmarks.utils.Frenet_coordinates_class import FrenetConverter
from scipy.stats import norm, expon
from scipy.signal import correlate, find_peaks

MOTION_DISPERSION_X = 0.8
MOTION_DISPERSION_Y = 0.05
MOTION_DISPERSION_THETA = 0.05

class CurvatureFilter:
    def __init__(self, planner_name, test_id, extra_params={}) -> None:
        self.params = load_parameter_file_with_extras("particle_filter_params", extra_params)
        self.planner_name = planner_name
        self.test_id = test_id
        self.data_path = f"Logs/{planner_name}/RawData_{test_id}/"
        self.estimates = None
        self.scan_simulator = None
        self.Q = np.diag(self.params.motion_q_stds) **2 
        self.NP = self.params.number_of_particles
        self.dt = self.params.dt
        self.num_beams = self.params.number_of_beams
        self.lap_number = 0
        self.map_name = None

        self.particles = None
        self.proposal_distribution = None
        self.weights = np.ones(self.NP) / self.NP
        self.particle_indices = np.arange(self.NP)
        
        self.Full_map_curvature = None
        self.Times = []
        self.Mapdata = None
        self.estimate = None
        self.i = 0
        self.start = 0
        
        self

    def init_pose(self, init_pose):
        '''Randomly generate a bunch of particles'''
        self.estimates = [init_pose]
        self.estimate = init_pose
        self.proposal_distribution = init_pose + np.random.multivariate_normal(np.zeros(3), self.Q*self.params.init_distribution, self.NP)
        self.particles = self.proposal_distribution
        # plt.plot(self.particles[:,0], self.particles[:,1], 'ro') #(initiallise the filter assuming the car is at the initial pose)
        # plt.show()

        return init_pose

    def set_map(self, map_name):
        self.map_name = map_name
        self.scan_simulator = SensorModel(f"maps/{map_name}", self.test_id, self.num_beams, self.params.fov)
        self.Mapdata = MapData(map_name)
        self.Full_map_curvature = self.scan_simulator.kappaSmooth

        image = self.scan_simulator.map_img
        image_np = np.array(image, dtype=int)
        image_np[image_np <= 128.] = 0
        image_np[image_np > 128.] = 1
        occupancy_grid = image_np
        # print(image_np)
        # # Visualize the occupancy grid
        # plt.imshow(occupancy_grid, cmap='gray', interpolation='nearest')
        # plt.title('Occupancy Grid')
        # plt.show()

    def localise(self, action, observation):
        '''MCL algorithm'''
        t0 = time.time()
        vehicle_speed = observation["vehicle_speed"] 
        self.particle_control_update(action, vehicle_speed)
        predict_state, pdf1, x, s, d, d_theta = self.motion_model_update(action, vehicle_speed, self.estimate)

        self.measurement_update(observation["scan"]) 
        pdf2, localtrack, psiLocal, local_tranformation_matrix, delta_xL, delta_yL, delta_thetaL = self.curve_measurement_update(observation["scan"])
        
        pdf_product = pdf1 * pdf2
        pdf_product /= np.trapz(pdf_product, x)
        peaks, properties = find_peaks(pdf_product, prominence=0.2)  # Adjust prominence as needed
        estimated_position_idx = np.argmax(pdf_product)       # Find the peak closest to the actual segment position (for visualization purposes)
        
        estimate = np.dot(self.particles.T, self.weights)
        self.estimates.append(estimate)
        self.estimate = estimate
        
        # Find s and heading at the estimated position
        if self.start == 0:
            estimated_position_idx = 0
            self.start = 1
            
        est_s = x[estimated_position_idx]
        est_centreline_heading = self.scan_simulator.psiSmooth[estimated_position_idx]
        x_center_est, y_center_est = self.scan_simulator.centreline_resampled[estimated_position_idx]
        x_con, y_con = self.scan_simulator.FrennetCon.to_cartesian(est_s, d)
        
    
        self.estimate = np.array([x_con, y_con, self.estimate[2]])
        
       

        
        
        
        First_point_local = np.array([localtrack[0][0], localtrack[0][1], psiLocal[0]])
        Global_point = np.array([x_center_est,y_center_est,est_centreline_heading])
        local_to_global_matrix, delta_x, delta_y, delta_theta = compute_relative_transformation( First_point_local, Global_point)
        
        # Total_transformation = np.dot(local_tranformation_matrix, local_to_global_matrix)
        combined_transformation = np.dot(local_to_global_matrix, local_tranformation_matrix)
        x_trans = combined_transformation[0,2]
        y_trans = combined_transformation[1,2]
        Rotation = np.array([[combined_transformation[0,0], combined_transformation[0,1]], [combined_transformation[1,0], combined_transformation[1,1]]])
        # Apply transformations
        direct_transform = np.dot(combined_transformation, Global_point)

        # local_pts = reoreintate_pts_with_R(localtrack[:, :2], np.array([x_trans,y_trans]), Rotation)
        centreline_to_zero_in_local= reoreintate_pts(localtrack[:, :2], np.array([delta_xL, delta_yL]), delta_thetaL)
        Zero_centre_to_global = reoreintate_pts(centreline_to_zero_in_local[:, :2], np.array([x_center_est,y_center_est]), est_centreline_heading)
        
        Scanmatch_transformed_points = reoreintate_pts(localtrack[:, :2], np.array([self.estimate[0], self.estimate[1]]), self.estimate[2])
     
        
        # local_pts = reoreintate_pts(localtrack[:, :2], np.array([delta_x,delta_y]), delta_theta)
        
        
        
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        axs[0].set_title('Multiplication of Normal and Three-Peak Distributions')
        axs[0].plot(x, pdf1, label='Prediction update', color='blue', linestyle='--')
        axs[0].plot(x, pdf2, label='Measurement update', color='green', linestyle='--')
        axs[0].plot(x, pdf_product, label='Estimation', color='red')
        axs[0].plot(x[estimated_position_idx], pdf_product[estimated_position_idx], "x", label="Detected Peaks", color='red')
        axs[0].set_xlabel('Track location in s(m)')
        axs[0].set_ylabel('Probability')
        axs[0].legend(loc='upper right')
        axs[0].grid(True)

        points = self.scan_simulator.centreline_resampled
        heading = self.scan_simulator.psiSmooth
        axs[1].plot(self.scan_simulator.L1[:, 0], self.scan_simulator.L1[:, 1], color='black')
        axs[1].plot(self.scan_simulator.L2[:, 0], self.scan_simulator.L2[:, 1], color='black')
        axs[1].scatter(points[:, 0], points[:, 1], color='orange', label='Smoothed centreline points', s=0.5)
   
        axs[1].scatter(localtrack[:, 0], localtrack[:, 1], color='red', label='Local Track in own reference frame ', s=0.5)
        axs[1].scatter(Scanmatch_transformed_points[:, 0], Scanmatch_transformed_points[:, 1], color='blue', label='Local-to-global (Scanmatch) ', s=0.5)
        # axs[1].scatter(centreline_to_zero_in_local[:, 0], centreline_to_zero_in_local[:, 1], color='green', label='Centreline_to_zero_in_local', s=0.5)
        # axs[1].scatter(Zero_centre_to_global[:, 0], Zero_centre_to_global[:, 1], color='pink', label='Zero_centre_to_global', s=0.5)
        
        axs[1].arrow(estimate[0], estimate[1], np.cos(estimate[2]),  np.sin(estimate[2]), color='blue', label='Estimated position using scanmatch ', head_width=0.2, head_length=0.2)
        axs[1].arrow(x_center_est, y_center_est, np.cos(est_centreline_heading), np.sin(est_centreline_heading), color='green', label='Estimated Progress point on centreline', head_width=0.2, head_length=0.2)
        axs[1].arrow(x_con, y_con, np.cos(estimate[2]),  np.sin(estimate[2]), color='purple', label='Frennet to cartesian', head_width=0.2, head_length=0.2)
        axs[1].scatter(points[0, 0], points[0, 1], color='red', label='Start', s=0.5)
        axs[1].legend(loc='upper right')
        axs[1].set_title("Track Points")

        plt.tight_layout()
        output_dir = f"Results/Cuvature_localise"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/{self.map_name}_iteration_plot_{self.i}.svg")
        plt.close(fig)  # Close the figure after saving

        self.i += 1
        
        t1 = time.time()
        localise_step_time = t1 - t0
        self.Times.append(localise_step_time)
        # name = save_path + f"LocalMapGeneration_{i}"
        # plt.savefig(name + ".svg", bbox_inches="tight")

        # image = self.scan_simulator.map_img
        # image_np = np.array(image, dtype=int)
        # image_np[image_np <= 128.] = 0
        # image_np[image_np > 128.] = 1
        # occupancy_grid = image_np
        
        # x, y = self.Mapdata.pts2rc(self.particles)
        # xe, ye = self.Mapdata.xy2rc(estimate[0], estimate[1])
        # mean = np.array([xe, ye])
        # cov = np.cov(x, y)
        
        # plt.imshow(occupancy_grid, cmap='gray', interpolation='nearest')
        # # plt.plot(self.particles[:,0], self.particles[:,1], 'ro', markersize=0.2)
        # plt.plot(estimate[0], estimate[1], 'yo', markersize=5)
        # plot_covariance_ellipse(mean, cov, n_std=2, edgecolor='blue', facecolor='none', linestyle='--')
        # plt.show()

        return estimate
    
    def motion_model_update(self, control, vehicle_speed, estimate):
        '''Predict the next state using the motion model'''
        
        predicted_state = particle_dynamics_update(estimate, control, vehicle_speed, self.dt, self.params.wheelbase) #next_states(x, y, theta)
        idx, closest_x, closest_y = self.scan_simulator.FrennetCon.closest_point_on_track(predicted_state[0], predicted_state[1])
        s, d, d_theta, closest_point = self.scan_simulator.FrennetCon.to_frenet(predicted_state[0], predicted_state[1],predicted_state[2])
        
        # Visualise closest point test
        # self.scan_simulator.FrennetCon.visualize(predicted_state[0], predicted_state[1], predicted_state[2])
        
        mu1 = self.scan_simulator.s_track_raw[idx]
        x = np.linspace(0, self.scan_simulator.s_track_raw[-1], len(self.scan_simulator.centreline_resampled))
        pdf1 = norm.pdf(x, mu1, MOTION_DISPERSION_X) # Compute the probability density functions (PDFs) o
        # pdf1_normalized = pdf1 / np.max(pdf1)
        # Integration = np.trapz(pdf1, x) # First few instances will show 0.5 until the full distribution is shown and the result is 1
        # Sum = sum(pdf1) # Will never summise to 1 since we are taking discrete points
        
        # plt.figure(figsize=(10, 6))
        # plt.plot(x, pdf1, label=f'Normal N({mu1}, {sigma1}²)', color='blue', linestyle='--')
        # plt.title('Belief of the car position')
        # plt.xlabel('x')
        # plt.ylabel('Probability Density')
        # plt.legend()
        # plt.grid(True)
        # plt.show()  # AT the moment the first closest point is the last point on the track, thus the peak is at the end of the graph
        
        return pdf1, pdf1 , x , s , d, d_theta,

    def particle_control_update(self, control, vehicle_speed):
        '''Add noise to motion model and update particles'''
        next_states = particle_dynamics_update(self.proposal_distribution, control, vehicle_speed, self.dt, self.params.wheelbase) #next_states(x, y, theta)
        next_states[:,0] += np.random.normal(loc=0.0,scale=MOTION_DISPERSION_X,size=self.NP)
        next_states[:,1] += np.random.normal(loc=0.0,scale=MOTION_DISPERSION_Y,size=self.NP)
        next_states[:,2] += np.random.normal(loc=0.0,scale=MOTION_DISPERSION_THETA,size=self.NP)
        
        # random_samples = np.random.multivariate_normal(np.zeros(3), self.Q, self.NP)s_track_raw
        # self.particles = next_states + random_samples
        self.particles = next_states

    def measurement_update(self, measurement):
        '''Update the weights of the particles based on the measurement
            The measurement in this case is the actual lidar scan from the car
        '''
        global_track = self.scan_simulator.centreline_resampled
        localTrack = self.scan_simulator.ExtractLocalTrack(measurement) # Get actual scan data from the car
        localTrack = localTrack[:, :2] # Extract only the x and y coordinates
        
        #Returns the normalised weights for each particle
        best_transformation, best_transformed_scan, costs, self.weights = self.scan_simulator.icp_scan_matching(global_track, localTrack, max_iterations=10, tolerance=1e-7, initial_guesses=self.particles)
        
        # # Resampling
        proposal_indices = np.random.choice(self.particle_indices, self.NP, p=self.weights)
        self.proposal_distribution = self.particles[proposal_indices,:]

    def curve_measurement_update(self, measurement):
        # Extract local curvature and heading angle
        localTrack = self.scan_simulator.ExtractLocalTrack(measurement) # Get actual scan data from the car
        localTrack = filter_and_add_point(localTrack[:,:2]) # This cuts of any points behind the car as it is needed for the porobability calculation
        
        kappafull = self.Full_map_curvature
        el_lengthsLocal = np.sqrt(np.sum(np.diff(localTrack, axis=0)**2, axis=1))
        psiLocal, kappaLocal = tph.calc_head_curv_num.calc_head_curv_num(np.column_stack((localTrack[:,1],localTrack[:,0])), el_lengthsLocal, False)
        psiLocal = -psiLocal #Issue for some reason the psi values are negative
        
        # Compare to the global curvature and get correlation porbablities
        x = np.linspace(0, self.scan_simulator.s_track_raw[-1], len(self.scan_simulator.centreline_resampled))
        correlation_scores = np.correlate(self.Full_map_curvature, kappaLocal, mode='same')
        
        #SHift all positivre
        min_score = np.min(correlation_scores)
        shifted_scores = correlation_scores - min_score # Shift the scores to ensure all are positive (if necessary) >=0
        normalized_scores = shifted_scores / np.max(shifted_scores) # Normalize the scores to a range between 0 and 1
        # shifted_scores /= np.trapz(shifted_scores, x)
        
        #only take positve correlation
        non_negative_scores = np.where(correlation_scores < 0, 0, correlation_scores)
        measurement_probabilities = non_negative_scores / np.sum(non_negative_scores) # Normalise the scores so the sum is 1
        # print(f'Sum of measurement probabilities: {np.sum(measurement_probabilities)}')
        # print(f'Lenght of full curvature: {len(self.Full_map_curvature)}')
        
        Integration = np.trapz(non_negative_scores, x)
        
        First_point_centreline = np.array([localTrack[0][0], localTrack[0][1], psiLocal[0]])
        Car_location = np.array([0,0,0])
        local_tranformation_matrix, delta_x, delta_y, delta_theta = compute_relative_transformation(First_point_centreline, Car_location)
        
        # plt.figure(figsize=(10, 6))
        # plt.scatter(localTrack[:, 0], localTrack[:, 1], color='blue', label='Local Track')
        # plt.arrow(First_point_centreline[0], First_point_centreline[1], 0.5 * np.cos(First_point_centreline[2]), 0.5 * np.sin(First_point_centreline[2]), head_width=0.2, head_length=0.3, fc='red', ec='red', label='Local Track Heading')
        # plt.arrow(Car_location[0], Car_location[1], 0.5 * np.cos(Car_location[2]), 0.5 * np.sin(Car_location[2]), head_width=0.2, head_length=0.3, fc='green', ec='green', label='Car Heading')
        # plt.title('Local Track and Car Heading')
        # plt.legend()
        # plt.grid(True)
        # plt.show()
        
        return non_negative_scores, localTrack, psiLocal ,local_tranformation_matrix, delta_x, delta_y, delta_theta
        # return measurement_probabilities
        
    def lap_complete(self):
        estimates = np.array(self.estimates)
        np.save(self.data_path + f"cf_estimates_{self.map_name}_{self.lap_number}.npy", estimates)
        times = np.array(self.Times)
        np.save(self.data_path + f"cf_steptimes_{self.map_name}_{self.lap_number}.npy", times)
        self.lap_number += 1
        
def plot_covariance_ellipse(mean, cov, n_std=2.0, ax=None, **kwargs):
    """
    Plots a covariance ellipse representing the covariance matrix.

    Parameters:
        mean (array-like): The mean of the Gaussian distribution, [x, y].
        cov (2x2 array-like): The covariance matrix of the Gaussian distribution.
        n_std (float): Number of standard deviations for the ellipse radius (default is 2).
        ax (matplotlib.axes.Axes, optional): The axes on which to plot the ellipse. 
                                            If not provided, a new plot will be created.
        **kwargs: Additional keyword arguments to pass to the Ellipse patch (e.g., color, alpha).

    Returns:
        matplotlib.patches.Ellipse: The ellipse patch representing the covariance.
    """
    if ax is None:
        ax = plt.gca()  # Get current axes if none provided

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort eigenvalues and eigenvectors by eigenvalue (largest first)
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

    # Compute the angle for the ellipse based on the first eigenvector
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

    # Compute the width and height of the ellipse based on eigenvalues
    width, height = 2 * n_std * np.sqrt(eigenvalues)

    # Create the ellipse and add it to the plot
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    # ax.add_patch(ellipse)
    plt.gca().add_patch(ellipse)

    # Plot the mean
    # ax.plot(*mean, 'ro', markersize=5)  # Mark the mean point
    plt.plot(*mean, 'ro', markersize=5)

    return ellipse

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


def particle_dynamics_update(states, actions, speed, dt, L):
    '''Motion model: Update the particles using the bicycle model'''
    # Convert states to numpy array if it's a list
    states = np.asarray(states)
    
    # Check if handling a single state or multiple states
    if states.ndim == 1:
        # Handle single state
        states[0] += speed * np.cos(states[2]) * dt
        states[1] += speed * np.sin(states[2]) * dt
        states[2] += speed * np.tan(actions[0]) / L * dt
    else:
        # Handle multiple states
        states[:, 0] += speed * np.cos(states[:, 2]) * dt
        states[:, 1] += speed * np.sin(states[:, 2]) * dt
        states[:, 2] += speed * np.tan(actions[0]) / L * dt

    return states

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
    
    return transformation_matrix, delta_x, delta_y, delta_theta

def reoreintate_pts(pts, position, theta):
    rotation_mtx = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts = np.matmul(pts, rotation_mtx.T) + position

    return pts

def reoreintate_pts_with_R(pts, position, rotation_mtx):
    pts = np.matmul(pts, rotation_mtx.T) + position

    return pts

class SensorModel:
    def __init__(self, map_name, test_id,  num_beams, fov, eps=0.01, theta_dis=2000, max_range=30.0):
        self.test_id = test_id
        self.num_beams = num_beams
        self.fov = fov
        self.eps = eps
        self.theta_dis = theta_dis
        self.max_range = max_range
        self.angle_increment = self.fov / (self.num_beams - 1)
        self.theta_index_increment = theta_dis * self.angle_increment / (2. * np.pi)
        self.orig_x = None
        self.orig_y = None
        self.map_img = None
        self.map_height = None
        self.map_width = None
        self.map_resolution = None
        self.dt = None
        
        self.map_data = None
        self.number_of_track_points = None
        self.centreline_raw = None
        self.centreline_resampled = None
        self.L1 = None
        self.L2 = None
        
        self.psi_raw = None
        self.kappa_raw = None
        self.psiSmooth = None
        self.kappaSmooth = None
        self.number_of_track_points = None
        self.s_track_raw = None
        
        self.FrennetCon = None
        
        theta_arr = np.linspace(0.0, 2*np.pi, num=theta_dis)
        self.sines = np.sin(theta_arr)
        self.cosines = np.cos(theta_arr)
        
        # Define weighting factors for distance, rotation, and translation
        self.w_dist = 1.0
        self.w_rot = 0.5
        self.w_trans = 0.5
        
        self.load_map(map_name)
        self.FeatureExtractor = LocalMapGenerator(map_name, 0, False) #False to avoid saving data
        
    def load_map(self, map_path):
        # load map image
       
        map_name = os.path.splitext(os.path.basename(map_path))[0]
        map_img_path = os.path.splitext(map_path)[0] + ".png"
        map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM))
        map_img = map_img.astype(np.float64)
       
        # grayscale -> binary
        map_img[map_img <= 128.] = 0.
        map_img[map_img > 128.] = 255.
        self.map_img = map_img

        self.map_height = map_img.shape[0]
        self.map_width = map_img.shape[1]

        with open(map_path + ".yaml", 'r') as yaml_stream:
            map_metadata = yaml.safe_load(yaml_stream)
            self.map_resolution = map_metadata['resolution']
            self.origin = map_metadata['origin']

        self.orig_x = self.origin[0]
        self.orig_y = self.origin[1]

        self.dt = self.map_resolution * edt(map_img)
        
        # Process track
        self.map_data = MapData(map_name)
        track = CentreLine(map_name)
        self.FrennetCon = FrenetConverter(track.path[:,0], track.path[:,1])
        
        #track.widths = track.widths / self.map_data.map_resolution
        self.L1 = track.path[:, :2] + track.nvecs * track.widths[:, 0][:, None]
        self.L2 = track.path[:, :2] - track.nvecs * track.widths[:, 1][:, None]
        
        scaled_L1_x = self.L1[:, 0]
        scaled_L1_y = self.L1[:, 1] 
        scaled_L2_x = self.L2[:, 0]
        scaled_L2_y = self.L2[:, 1] 
        
        # scaled_L1_x = (self.L1[:, 0] - self.orig_x) 
        # scaled_L1_y = (self.L1[:, 1] - self.orig_y) 
        # scaled_L2_x = (self.L2[:, 0] - self.orig_x) 
        # scaled_L2_y = (self.L2[:, 1] - self.orig_y)
        
        # scaled_L1_x = (self.L1[:, 0] - self.orig_x) / self.map_data.map_resolution
        # scaled_L1_y = (self.L1[:, 1] - self.orig_y) / self.map_data.map_resolution
        # scaled_L2_x = (self.L2[:, 0] - self.orig_x) / self.map_data.map_resolution
        # scaled_L2_y = (self.L2[:, 1] - self.orig_y) / self.map_data.map_resolution
        
        self.L1 = np.column_stack((scaled_L1_x, scaled_L1_y))
        self.L2 = np.column_stack((scaled_L2_x, scaled_L2_y))
        
        # print(f'Raw:{len(track.path)}') #1007
        # print(f'ellenghts:{len(track.el_lengths)}') #1006
        # print(f'distance:{len(track.s_path)}') #1007
        # print(f'distance {track.s_path}')
        
        # Map_origin = self.map_data.map_origin
        # self.map_data.plot_map_img_light()
        # plt.scatter(Map_origin[0], Map_origin[1], c='red', label='Map Origin')
        # plt.show()

        # # Plotting the track centerline
        # track.path[:, 0] = (track.path[:, 0] - self.map_data.map_origin[0]) / self.map_data.map_resolution
        # track.path[:, 1] = (track.path[:, 1] - self.map_data.map_origin[1]) / self.map_data.map_resolution
        # plt.plot(track.path[:, 0], track.path[:, 1], '--', linewidth=2, color='black')
        # plt.plot(scaled_L1_x, scaled_L1_y, color='green') # Plotting the track boundaries
        # plt.plot(scaled_L2_x, scaled_L2_y, color='green')
        # plt.show()
        
        # Get features from raw track 
        self.centreline_raw = track.path
        el_lengths = np.linalg.norm(np.diff(self.centreline_raw, axis=0), axis=1)
        self.psi_raw, self.kappa_raw = tph.calc_head_curv_num.calc_head_curv_num(np.column_stack((self.centreline_raw[:,1], self.centreline_raw[:,0])), el_lengths, False)
        self.psi_raw = -self.psi_raw #Issue for some reason the psi values are negative
        self.number_of_track_points = len(self.kappa_raw)
        self.s_track_raw = track.s_path
        
        
        # Resample the track points and get its features
        self.centreline_resampled, smooth_line = resample_track_points(self.centreline_raw, seperation_distance=0.2, smoothing=0.5)
        el_lengthsSmooth = np.linalg.norm(np.diff(self.centreline_resampled, axis=0), axis=1)
        smooth_s_path = np.insert(np.cumsum(el_lengthsSmooth), 0, 0)
        self.psiSmooth, self.kappaSmooth  = tph.calc_head_curv_num.calc_head_curv_num(np.column_stack((self.centreline_resampled[:,1], self.centreline_resampled[:,0])), el_lengthsSmooth, False)
        self.psiSmooth = -self.psiSmooth
        
        # print(f'kappa smooth:{len(self.kappaSmooth)}')
        
        # print(f'Raw:{len(self.centreline_raw)}') #1007
        # print(f'resampled:{len(self.centreline_resampled)}') #1009 Seems to add two extra points
        # print(f'Raw:{self.centreline_raw}')
        # print(f'resampled:{self.centreline_resampled}')
        
        # # Plot curvature in a separate figures for raw and smooth
        # # Use this to make sure the conventions and starting angles are correct
        # plt.figure()
        # plt.plot(range(len(self.kappa_raw)), self.kappa_raw)
        # plt.title('Raw Curvature of the track')
        # plt.xlabel('Sample Index')
        # plt.ylabel('Curvature')
        # plt.show()

        # plt.figure()
        # plt.plot(range(len(self.psi_raw)), self.psi_raw)
        # plt.title('Raw Heading angle of the track')
        # plt.xlabel('Sample Index')
        # plt.ylabel('Heading Angle (radians)')
        # plt.show()
        
        # plt.figure()
        # plt.plot( smooth_s_path, self.kappaSmooth)
        # plt.title('Smooth Curvature of the track')
        # plt.xlabel('Sample Index')
        # plt.ylabel('Curvature')
        # plt.show()

        # plt.figure()
        # plt.plot(range(len(self.psiSmooth)), self.psiSmooth)
        # plt.title('Smooth Heading angle of the track')
        # plt.xlabel('Sample Index')
        # plt.ylabel('Heading Angle (radians)')
        # plt.show()

    def scan(self, pose):
        scan = get_scan(pose, self.theta_dis, self.fov, self.num_beams, self.theta_index_increment, self.sines, self.cosines, self.eps, self.orig_x, self.orig_y, self.map_height, self.map_width, self.map_resolution, self.dt, self.max_range)
        # self.local_track = self.local_map_generator.generate_line_local_map(np.copy(obs['scan']))
        
        return scan
    
    def ExtractLocalTrack(self, scan):
        feature = self.FeatureExtractor.generate_line_local_map(scan)
        scan_xs, scan_ys = self.map_data.pts2rc(feature)
        # feature = np.vstack([scan_xs, scan_ys])
        
        return feature
    
    def transform_points(self, points, transformation):
        """ Apply a transformation to the points (2D rotation + translation) """
        theta, tx, ty = transformation
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
        return points @ rotation_matrix.T + np.array([tx, ty])
    
    def icp_scan_matching(self, global_map, local_scan, max_iterations=10, tolerance=1e-7, initial_guesses=None):
        def compute_cost(transformation, global_map, local_scan):
            """ Compute the cost based on distance after applying the particle transformation """
            transformed_scan = self.transform_points(local_scan, transformation)
            distances, _ = kdtree.query(transformed_scan)

            # Distance cost: sum of distances between transformed local scan points and the nearest global map points
            return np.sum(distances)

        # Create KDTree for fast nearest neighbor search
        kdtree = KDTree(global_map)

        # Best results
        best_transformation = None
        best_transformed_scan = None
        best_cost = np.inf
        particle_costs = []

        # plt.scatter(global_map[:, 0], global_map[:, 1], c='black', label='Global Map', alpha=0.5, s=0.5)
        # Iterate through each particle guess (x, y, orientation)
        for particle in initial_guesses:
            # Particle transformation: [theta, tx, ty] = [orientation, x, y]
            # plt.scatter(local_scan[:, 0], local_scan[:, 1], c='red', label='Local Scan', alpha=0.5, s=0.5)
            transformation = np.array([particle[2], particle[0], particle[1]])  # [theta, x, y]
            
            # Apply the particle transformation to the local scan
            transformed_scan = self.transform_points(local_scan, transformation)
            # plt.scatter(transformed_scan[:, 0], transformed_scan[:, 1], c='green', label='Initial Guess', alpha=0.5, s=0.5)
            
            # Compute the cost for this particle based on transformed scan and global map
            cost = compute_cost(transformation, global_map, local_scan)
            particle_costs.append(cost)

            # Track the best transformation with the lowest cost
            if cost < best_cost:
                best_cost = cost
                best_transformation = transformation
                best_transformed_scan = transformed_scan
                
        particle_costs = np.array(particle_costs)
        weights = 1 / (particle_costs + 1e-9)  # Add a small constant to avoid division by zero
        # Normalize the weights so they sum to 1
        weights /= np.sum(weights)
                
        # plt.show()
        # Return the best transformation, transformed scan, and the array of costs for all particles
        return best_transformation, best_transformed_scan, np.array(particle_costs), weights



def compute_weights_from_costs(self, costs):
    # Invert costs: smaller cost -> higher weight
    weights = 1 / (costs + 1e-9)  # Add a small constant to avoid division by zero
    
    # Normalize the weights so they sum to 1
    weights /= np.sum(weights)

    return weights


    def get_increment(self):
        return self.angle_increment

    def xy_2_rc(self, points):
        r, c = xy_2_rc_vec(points[:, 0], points[:, 1], self.orig_x, self.orig_y, self.map_resolution)
        return np.stack((c, r), axis=1)
    
 
# @njit(cache=True)
def interpolate_track_new(points, n_points=None, s=0):
    if len(points) <= 1:
        return points
    order_k = min(3, len(points) - 1)
    tck = interpolate.splprep([points[:, 0], points[:, 1]], k=order_k, s=s)[0]
    if n_points is None: n_points = len(points)
    track = np.array(interpolate.splev(np.linspace(0, 1, n_points), tck)).T
    
    return track

# @njit(cache=True)
def resample_track_points(points, seperation_distance=0.2, smoothing=0.2):
    # if points[0, 0] > points[-1, 0]:
    #     points = np.flip(points, axis=0)

    line_length = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    n_pts = max(int(line_length / seperation_distance), 2)
    smooth_line = interpolate_track_new(points, None, smoothing)
    resampled_points = interpolate_track_new(smooth_line, n_pts, 0)

    return resampled_points, smooth_line

@njit(cache=True)
def xy_2_rc(x, y, orig_x, orig_y, height, width, resolution):
    x_trans = x - orig_x
    y_trans = y - orig_y

    if x_trans < 0 or x_trans >= width * resolution or y_trans < 0 or y_trans >= height * resolution:
        c = -1
        r = -1
    else:
        c = int(x_trans/resolution)
        r = int(y_trans/resolution)


    return r, c

@njit(cache=True)
def xy_2_rc_vec(x, y, orig_x, orig_y, resolution):
    x_trans = x - orig_x
    y_trans = y - orig_y

    c = x_trans/resolution
    r = y_trans/resolution

    return r, c

@njit(cache=True)
def distance_transform(x, y, orig_x, orig_y, height, width, resolution, dt):
    r, c = xy_2_rc(x, y, orig_x, orig_y, height, width, resolution)
    distance = dt[r, c]
    return distance

@njit(cache=True)
def trace_ray(x, y, theta_index, sines, cosines, eps, orig_x, orig_y, height, width, resolution, dt, max_range):
    theta_index_ = int(theta_index)
    s = sines[theta_index_]
    c = cosines[theta_index_]

    dist_to_nearest = distance_transform(x, y, orig_x, orig_y, height, width, resolution, dt)
    total_dist = dist_to_nearest

    while dist_to_nearest > eps and total_dist <= max_range:
        x += dist_to_nearest * c
        y += dist_to_nearest * s

        dist_to_nearest = distance_transform(x, y, orig_x, orig_y, height, width, resolution, dt)
        total_dist += dist_to_nearest

    if total_dist > max_range:
        total_dist = max_range
    
    return total_dist

@njit(cache=True)
def get_scan(pose, theta_dis, fov, num_beams, theta_index_increment, sines, cosines, eps, orig_x, orig_y, height, width, resolution, dt, max_range):
    scan = np.empty((num_beams,))

    theta_index = theta_dis * (pose[2] - fov/2.)/(2. * np.pi)

    theta_index = np.fmod(theta_index, theta_dis)
    while (theta_index < 0):
        theta_index += theta_dis

    for i in range(0, num_beams):
        scan[i] = trace_ray(pose[0], pose[1], theta_index, sines, cosines, eps, orig_x, orig_y, height, width, resolution, dt, max_range)

        theta_index += theta_index_increment

        while theta_index >= theta_dis:
            theta_index -= theta_dis

    return scan