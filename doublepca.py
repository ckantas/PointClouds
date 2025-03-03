import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import math
from functools import partial
from open3d.t.geometry import TriangleMesh
import util
import time

dataname = "C:/Users/chris/Desktop/Documents/NewData/CircularVentilationGrateExtraCleanedFull.ply"
pcd = o3d.io.read_point_cloud(dataname)
pcd = util.preProcessCloud(pcd)

octree_depth = 8
octree = o3d.geometry.Octree(max_depth=octree_depth)
octree.convert_from_point_cloud(pcd, size_expand=0.01)  # Expand slightly to ensure full coverage

def calculate_eigen_norm_and_plane_direction(neighbor_coordinates):
    if len(neighbor_coordinates) < 3:
        return np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0])
    
    mu = np.mean(neighbor_coordinates, axis=0)
    norm = neighbor_coordinates - mu
    cov = np.cov(norm.T)
    eig_val, eig_vec = np.linalg.eig(cov)
    sorted_idx = np.argsort(eig_val)[::-1]
    eig_val = eig_val[sorted_idx]
    eig_vec = eig_vec[:, sorted_idx]
    eig_val_norm = eig_val.copy()

    for z in range(len(eig_val)):
        eig_val_norm[z] = np.exp(eig_val[z])/np.sum(np.exp(eig_val))

    plane_direction = np.cross(eig_vec[:, 0], eig_vec[:, 1])

    return mu, eig_val_norm, plane_direction

def plane_direction_to_color(plane_direction, contrast_factor=0.5):
    """ Convert a plane normal (x, y, z) into a high-contrast RGB gradient mapping. """
    # Normalize direction vector
    plane_direction = plane_direction / np.linalg.norm(plane_direction)

    # Ensure flipped vectors get the same color by taking absolute values
    r = abs(plane_direction[0])
    g = abs(plane_direction[1])
    b = abs(plane_direction[2])

    # **Enhance contrast using exponential scaling**
    r = r**contrast_factor
    g = g**contrast_factor
    b = b**contrast_factor

    return [r, g, b]  # RGB values naturally in range [0,1]

def calculate_normal_variation(normals):
    mu = np.mean(normals, axis=0)
    norm = normals - mu
    cov = np.cov(norm.T)
    eig_val, _ = np.linalg.eig(cov)
    sorted_idx = np.argsort(eig_val)[::-1]
    eig_val = eig_val[sorted_idx]
    eig_val_norm = eig_val / np.sum(eig_val)

    return mu, eig_val_norm

def get_nearest_neighbor_directions(point, kdtree, centers_points, plane_directions, k=10):
    """ Get the directions of the k nearest neighbors to a given point. """
    _, idx, _ = kdtree.search_knn_vector_3d(point, k)
    nearest_points = centers_points[idx]
    nearest_directions = plane_directions[idx]
    return idx, nearest_points, nearest_directions

def align_normals(reference_normal, neighbor_directions):
    aligned_normals = np.array(neighbor_directions)
    
    # Check dot product: If negative, flip the normal
    for i in range(len(aligned_normals)):
        if np.dot(reference_normal, aligned_normals[i]) < 0:
            aligned_normals[i] = -aligned_normals[i]

    return aligned_normals

def get_neighbors(point, pcd, kdtree, radius=1.2):
    """ Get neighboring points around a given point using KD-tree search. """
    _, idx, _ = kdtree.search_radius_vector_3d(point, radius)
    return np.asarray(pcd.points)[idx]

def build_kdtree(centers):
    """ Create a KD-tree for the octree node centers. """
    centers_pcd = o3d.geometry.PointCloud()
    centers_pcd.points = o3d.utility.Vector3dVector(centers)
    return o3d.geometry.KDTreeFlann(centers_pcd), centers_pcd

def apply_double_pca(pcd, octree, depth=8, vector_scale=0.4, min_points=5, search_radius=1.2, k=20):
    """ Perform PCA on neighborhood points around each octree node's center. """
    lines = []
    means = []
    plane_directions = []
    all_points = np.asarray(pcd.points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    lambda2_values = []

    def apply_pca(node, node_info):
        if node_info.depth == depth and isinstance(node, o3d.geometry.OctreeLeafNode):
            if hasattr(node, "indices"):
                leaf_points = all_points[node.indices]

                if len(leaf_points) < min_points:
                    return
                
                # Compute mean position of leaf points
                mu = np.mean(leaf_points, axis=0)

                # Get neighboring points using KD-tree search
                neighbors = get_neighbors(mu, pcd, kdtree, radius=search_radius)

                if len(neighbors) < min_points:
                    return  # Skip if not enough neighbors for PCA
                
                # Compute PCA on the neighborhood points
                mu, _, plane_direction = calculate_eigen_norm_and_plane_direction(neighbors)

                # Compute start and end points of the line
                start_point = mu - (vector_scale / 2) * plane_direction
                end_point = mu + (vector_scale / 2) * plane_direction

                # Store the line
                lines.append([start_point, end_point])
                means.append(mu)
                plane_directions.append(plane_direction)

    octree.traverse(apply_pca)
    print("First PCA done")
    new_kdtree, centers_pcd = build_kdtree(means)
    plane_directions = np.asarray(plane_directions)
    centers_points = np.asarray(centers_pcd.points) 

    def apply_second_pca(node, node_info):
        if node_info.depth == depth and isinstance(node, o3d.geometry.OctreeLeafNode):
            if hasattr(node, "indices"):
                leaf_points = all_points[node.indices]
                if len(leaf_points) < min_points:
                    return

                mu = np.mean(leaf_points, axis=0)
                _, _, nearest_node_directions = get_nearest_neighbor_directions(mu, new_kdtree, centers_points, plane_directions, k=20)
                normal_mean, normal_variation = calculate_normal_variation(nearest_node_directions)
                lambda2_values.append((mu, normal_variation[1]))

    octree.traverse(apply_second_pca)
    print("Second PCA done")

    return lines, means, plane_directions, lambda2_values

lines, means, plane_directions, lambda2_values = apply_double_pca(pcd, octree, depth=octree_depth, vector_scale=0.4, min_points=5, search_radius=1.2, k=20)