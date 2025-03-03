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
    return normals  # Return normals instead of storing in shared array

def calculate_normal_variation(indices, points, normals, radius=2):
    eigenvalues = []
    for idx in indices:
        neighbor_indices = kdtree.query_ball_point(points[idx], radius)
        neighbor_normals = normals[neighbor_indices]
        mu = np.mean(neighbor_normals, axis=0)
        norm = neighbor_normals - mu
        cov = np.cov(norm.T)
        eig_val, _ = np.linalg.eigh(cov)
        sorted_idx = np.argsort(eig_val)[::-1]
        eig_val = eig_val[sorted_idx]
        eig_val_norm = eig_val / np.sum(eig_val)
        eigenvalues.append(eig_val_norm)
    return eigenvalues  # Return eigenvalues instead of storing in shared array

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

    # # Single-core execution check
    # results_single = find_plane_directions(chunked_indices[0], myarray)
    # print("First 5 normals (single-core):")
    # print(results_single[:5])

    # First pass: Compute normals
    start_time_first_pca = time.time()
    with multiprocessing.Pool(processes=4, initializer=init_kdtree, initargs=(kdtree,)) as pool:
        normals_chunks = pool.starmap(find_plane_directions, [(chunk_indices, myarray, radius) for chunk_indices in chunked_indices])
    first_pca_duration = time.time() - start_time_first_pca
    print(f"First PCA time: {first_pca_duration:.2f} seconds")

    # Flatten normals to a single array in order
    all_normals = np.vstack(normals_chunks)

    # # Single-core execution check
    # results_single = calculate_normal_variation(chunked_indices[0], myarray, all_normals)
    # print("First 5 eigenvalues (single-core):")
    # print(results_single[:5])

    # Second pass: Compute normal variation
    start_time_second_pca = time.time()
    with multiprocessing.Pool(processes=4, initializer=init_kdtree, initargs=(kdtree,)) as pool:
        eigenvalues_chunks = pool.starmap(calculate_normal_variation, [(chunk_indices, myarray, all_normals, radius) for chunk_indices in chunked_indices])
    second_pca_duration = time.time() - start_time_second_pca
    print(f"Second PCA time: {second_pca_duration:.2f} seconds")

    # Flatten eigenvalues to a single array in order
    all_eigenvalues = np.vstack(eigenvalues_chunks)
    lambda2_values = all_eigenvalues[:, 1]

    # Print a few results to verify
    print("First 5 normals:")
    print(all_normals[:5])
    print("First 5 eigenvalues:")
    print(all_eigenvalues[:5])

class GaussMapVisualizer:
    def __init__(self, pcd, kdtree, all_normals, lambda2_values, radius):
        self.pcd = pcd
        self.points = np.asarray(pcd.points)
        self.pcd.paint_uniform_color([0.6, 0.6, 0.6])
        self.kdtree = kdtree
        self.plane_directions = all_normals
        self.radius = radius
        self.lambda2_values = lambda2_values
        self.reference_normal = None

        self.current_index = 0
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window("GaussMapVisualizer")

        self.vis.register_key_callback(262, self.next_neighborhood)
        self.vis.add_geometry(self.pcd)
        self.apply_lambda2_heatmap()
        #self._update_neighborhood()

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

        return mu, eig_val_norm
    
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

    def _update_neighborhood(self):
        """ Update visualization for the current neighborhood. """
        # Get the currently selected point
        query_point = np.asarray(self.pcd.points)[self.current_index]

        # Get nearest neighbors
        idx, neighbor_points, neighbor_directions = self.get_nearest_neighbor_directions(query_point, self.kdtree, self.centers_pcd, self.plane_directions, radius=self.radius)
        if self.current_index==0:
            self.reference_normal = neighbor_directions[0]

        aligned_directions = self.align_normals(self.reference_normal, neighbor_directions)
        normal_mean, normal_variation = self.calculate_normal_variation(aligned_directions)
        # Extract neighbor points
        #neighbor_points = np.asarray(self.pcd.points)[neighbor_indices]

        # Create a point cloud for the neighbors (red color)
        self.pcd_colors = np.tile((0.6,0.6,0.6), (self.points.shape[0], 1))
        self.pcd_colors[idx] = (1, 0, 0)
        self.pcd.colors = o3d.utility.Vector3dVector(self.pcd_colors)

        if hasattr(self, "normal_lines"):
            self.vis.remove_geometry(self.normal_lines)
        self.normal_lines = self.create_normal_lines(neighbor_points, aligned_directions, scale=2)
        self.vis.add_geometry(self.normal_lines)
        view_ctl =self.vis.get_view_control()
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
        print(f"Neighborhood {self.current_index}/{len(self.pcd.points)} updated")
        print(f"Normal mean: {normal_mean}, Normal variation: {normal_variation}")
        print(15*"-")

    def next_neighborhood(self, vis):
        """ Move to the next neighborhood when right arrow key is pressed. """
        self.current_index = (self.current_index + 500) % len(self.pcd.points)
        self._update_neighborhood()
    
    def normalize_lambda2_colors(self, lambda2_values):
        """ Normalize λ2 values to a colormap range. """
        min_val, max_val = np.percentile(lambda2_values, [2, 98])
        norm_lambda2 = (lambda2_values - min_val) / (max_val - min_val + 1e-6)  # Normalize to [0,1]

        # Use a colormap (e.g., viridis)
        colors = cm.viridis(norm_lambda2)[:, :3]  # Extract RGB colors

        return colors

    def apply_lambda2_heatmap(self):
        colors = self.normalize_lambda2_colors(self.lambda2_values)

        self.pcd.colors = o3d.utility.Vector3dVector(colors)  # Assign colors to points

        #self.vis.update_geometry(self.pcd)
        print("Heatmap applied!")

    def run(self):
        """ Start the Open3D visualization loop. """
        #plt.ion() 
        self.vis.run()
        self.vis.destroy_window()
        #plt.ioff()
        #plt.show()

viewer = GaussMapVisualizer(pcd, kdtree, all_normals, lambda2_values, radius)
viewer.run()