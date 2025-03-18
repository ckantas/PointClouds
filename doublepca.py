import multiprocessing
import os
import time
import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d
import util
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
import random

kdtree = None
radius = 1.3

def init_kdtree(tree):
    global kdtree
    kdtree = tree

def find_plane_directions(indices, points, radius=2):
    normals = []
    for idx in indices:
        neighbor_indices = kdtree.query_ball_point(points[idx], radius)
        neighbors = points[neighbor_indices]
        mean = np.mean(neighbors, axis=0)
        norm = neighbors - mean
        cov = np.cov(norm.T)
        eig_val, eig_vec = np.linalg.eigh(cov)
        sorted_idx = np.argsort(eig_val)[::-1]
        eig_vec = eig_vec[:, sorted_idx]
        plane_direction = eig_vec[:, 2]
        normals.append(plane_direction)
    return normals

def calculate_normal_standard_deviation(indices, points, normals, radius=2):
    standard_deviations = []

    for idx in indices:
        # **Step 1: Neighbor Query**
        neighbor_indices = kdtree.query_ball_point(points[idx], radius)

        # **Step 2: Retrieve Neighbor Normals (Optimized)**
        neighbor_normals = normals[neighbor_indices]  # Direct indexing (no conversion)

        if len(neighbor_normals) == 0:
            standard_deviations.append(0)  
            continue

        # **Step 3: Align Normals**
        reference_normal = neighbor_normals[0]
        dot_products = np.dot(neighbor_normals, reference_normal)
        aligned_normals = neighbor_normals * np.sign(dot_products)[:, np.newaxis]

        # **Step 4: Standard Deviation Calculation**
        std_dev = np.std(aligned_normals, axis=0)
        variation_measure = np.sum(std_dev)  

        standard_deviations.append(variation_measure)

    return standard_deviations  

if __name__ == "__main__": 
    dataname = "/home/chris/Code/PointClouds/data/ply/CircularVentilationGrateExtraCleanedFull.ply"
    pcd = o3d.io.read_point_cloud(dataname)
    pcd = util.preProcessCloud(pcd)
    myarray = np.asarray(pcd.points)
    indices = np.arange(len(myarray))
    kdtree = cKDTree(myarray)
    num_chunks = 16
    chunk_size = len(myarray) // num_chunks
    chunked_indices = [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]

    # First pass: Compute normals
    start_time_first_pca = time.time()
    with multiprocessing.Pool(processes=4, initializer=init_kdtree, initargs=(kdtree,)) as pool:
        normals_chunks = pool.starmap(find_plane_directions, [(chunk_indices, myarray, radius) for chunk_indices in chunked_indices])
    first_pca_duration = time.time() - start_time_first_pca
    print(f"First PCA time: {first_pca_duration:.2f} seconds")

    # Flatten normals to a single array in order
    all_normals = np.vstack(normals_chunks)

    # Second pass: Compute standard deviation-based variation
    start_time_standard_deviations = time.time()
    with multiprocessing.Pool(processes=4, initializer=init_kdtree, initargs=(kdtree,)) as pool:
        standard_deviation_chunks = pool.starmap(calculate_normal_standard_deviation, [(chunk_indices, myarray, all_normals, radius) for chunk_indices in chunked_indices])
    standard_deviations_duration = time.time() - start_time_standard_deviations
    print(f"Second PCA time: {standard_deviations_duration:.2f} seconds")

    # Flatten standard deviations to maintain order
    standard_deviations = np.hstack(standard_deviation_chunks)

    # Normalize variation values
    max_variation = np.max(standard_deviations) if len(standard_deviations) > 0 else 1
    standard_deviations /= max_variation

    # Print a few results to verify
    print("First 5 normals:")
    print(all_normals[:5])
    print("First 5 normalized standard deviations:")
    print(standard_deviations[:5])

class GaussMapVisualizer:
    def __init__(self, pcd, kdtree, all_normals, standard_deviations, radius):
        self.pcd = pcd
        self.points = np.asarray(pcd.points)
        self.pcd.paint_uniform_color([0.6, 0.6, 0.6])
        self.kdtree = kdtree
        self.plane_directions = all_normals
        self.radius = radius
        self.standard_deviations = standard_deviations
        low, high = np.percentile(self.standard_deviations, [95, 100])
        self.core_indices = np.where((self.standard_deviations > low) & (self.standard_deviations <= high))[0]
        print(f"Found {len(self.core_indices)} core points out of {len(self.pcd.points)} total points.")
        self.reference_normal = None

        self.current_index = 0
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window("GaussMapVisualizer")

        self.vis.register_key_callback(262, self.next_neighborhood)
        self.vis.register_key_callback(264, self.show_random_core_point)
        self.vis.add_geometry(self.pcd)
        self.apply_variation_heatmap()
        self._update_neighborhood()

    def get_nearest_neighbor_directions(self, point, kdtree, pcd, plane_directions, radius=2):
        """ Get the directions of the k nearest neighbors to a given point. """
        idx = kdtree.query_ball_point(point, radius)
        nearest_points = np.asarray(pcd.points)[idx]
        nearest_directions = np.asarray(plane_directions)[idx]
        return idx, nearest_points, nearest_directions
    
    def create_normal_lines(self, neighbor_points, neighbor_directions, scale=0.2):
        """ Create line segments for the normal directions at each point. """
        line_set = o3d.geometry.LineSet()

        start_points = np.array(neighbor_points)
        end_points = start_points + scale * np.array(neighbor_directions)
        lines = [[start_points[i], end_points[i]] for i in range(len(neighbor_points))]
        line_set.points = o3d.utility.Vector3dVector(np.concatenate(lines, axis=0))
        line_indices = [[i, i + 1] for i in range(0, len(lines) * 2, 2)]
        line_set.lines = o3d.utility.Vector2iVector(line_indices)
        line_set.colors = o3d.utility.Vector3dVector(np.tile((0, 0, 1), (len(lines), 1)))
        return line_set
    
    def align_normals(self, reference_normal, neighbor_directions):
        aligned_normals = np.array(neighbor_directions)
        
        # Check dot product: If negative, flip the normal
        for i in range(len(aligned_normals)):
            if np.dot(reference_normal, aligned_normals[i]) < 0:
                aligned_normals[i] = -aligned_normals[i]

        return aligned_normals
    
    def calculate_normal_variation(self, normals):
        mu = np.mean(normals, axis=0)
        norm = normals - mu
        cov = np.cov(norm.T)
        eig_val, _ = np.linalg.eig(cov)
        sorted_idx = np.argsort(eig_val)[::-1]
        eig_val = eig_val[sorted_idx]
        eig_val_norm = eig_val / np.sum(eig_val)
        
        return mu, eig_val_norm, cov
    
    def update_gauss_map(self, normals):
        """ Update the Gauss Map visualization with the current neighborhood's normals. """
        normals = np.array(normals)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)  # Normalize to unit sphere

        # Create figure
        plt.figure("Gauss Map", figsize=(6, 6))
        plt.clf()  # Clear previous plot
        ax = plt.subplot(111, projection="3d")

        # Plot unit sphere
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color="gray", alpha=0.3, edgecolor="none")  # Transparent sphere

        # Plot normal vectors
        for normal in normals:
            ax.quiver(0, 0, 0, normal[0], normal[1], normal[2], color="b", linewidth=1, arrow_length_ratio=0.1)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Gauss Map - Neighborhood {self.current_index}")

        plt.pause(0.1)  # Allow Matplotlib to update

    def show_random_core_point(self, vis):
        if len(self.core_indices) == 0:
            print("No core points found.")
            return

        # Pick a random core point
        self.current_index = random.choice(self.core_indices)

        # Visualize its neighborhood
        self._update_neighborhood()

        print(f"Showing core point {self.current_index}.")

    def _update_neighborhood(self):
        """ Update visualization for the current neighborhood. """
        # Get the currently selected point
        query_point = np.asarray(self.pcd.points)[self.current_index]

        # Get nearest neighbors
        idx, neighbor_points, neighbor_directions = self.get_nearest_neighbor_directions(query_point, self.kdtree, self.pcd, self.plane_directions, radius=self.radius)
        if self.current_index==0:
            self.reference_normal = neighbor_directions[0]

        #aligned_directions = self.align_normals(self.reference_normal, neighbor_directions)
        normal_mean, normal_variation, cov_after = self.calculate_normal_variation(neighbor_directions)

        if hasattr(self, "normal_lines"):
            self.vis.remove_geometry(self.normal_lines)
        self.normal_lines = self.create_normal_lines(neighbor_points, neighbor_directions, scale=2)
        self.vis.add_geometry(self.normal_lines)
        view_ctl = self.vis.get_view_control()
        lookat = query_point
        zoom = 0.080000000000000002
        front = [-0.024106890455448116,-0.57254772319971181,0.81951690799604338]
        up =  [0.014828165865396817,0.81946017828866602,0.57294427451208185]
        view_ctl.set_lookat(lookat)  # Set the point the camera is looking at
        view_ctl.set_up(up)      # Set the up direction of the camera
        view_ctl.set_front(front)  # Set the front direction of the camera
        view_ctl.set_zoom(zoom)          # Set the zoom factor of the camera

        #self.update_gauss_map(aligned_directions)

        self.vis.update_geometry(self.pcd)
        print(f"Neighborhood {self.current_index}/{len(self.pcd.points)} updated", flush=True)
        print(f'Aligned normals: {neighbor_directions}', flush=True)
        print(f"Normal mean: {normal_mean}, Normal variation: {normal_variation}", flush=True)
        print(f"Std Dev of Normals: {np.std(neighbor_directions, axis=0)}")
        print(f"Condition Number of Covariance: {np.linalg.cond(cov_after)}")
        print(15*"-", flush=True)

    def next_neighborhood(self, vis):
        """ Move to the next neighborhood when right arrow key is pressed. """
        self.current_index = (self.current_index + 500) % len(self.pcd.points)
        self._update_neighborhood()
    
    def normalize_variation_colors(self, variation_values):
        """ Normalize variation values to a colormap range. """
        min_val, max_val = np.percentile(variation_values, [2, 98])  # Robust normalization
        norm_variation = (variation_values - min_val) / (max_val - min_val + 1e-6)  # Normalize to [0,1]

        # Use a colormap (e.g., viridis) to visualize variation
        colors = cm.viridis(norm_variation)[:, :3]  # Extract RGB colors

        return colors

    def apply_variation_heatmap(self):
        """ Apply standard deviation-based variation as a heatmap to the point cloud. """
        colors = self.normalize_variation_colors(self.standard_deviations)

        self.pcd.colors = o3d.utility.Vector3dVector(colors)  # Assign colors to points

        # self.vis.update_geometry(self.pcd)  # Uncomment if using an Open3D visualizer
        print("Standard deviation heatmap applied!")


    def run(self):
        self.vis.run()  # Start the visualization loop (blocks until closed)
        self.vis.destroy_window()

viewer = GaussMapVisualizer(pcd, kdtree, all_normals, standard_deviations, radius)
viewer.run()

