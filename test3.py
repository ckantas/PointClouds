import multiprocessing
import os
import time
import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d
import util

kdtree = None

def init_kdtree(tree):
    global kdtree
    kdtree = tree

def find_plane_directions(indices, points, k=5):
    normals = []
    for idx in indices:
        _, neighbor_indices = kdtree.query(points[idx], k=k)
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

def calculate_normal_variation(indices, points, normals, k=5):
    eigenvalues = []
    for idx in indices:
        _, neighbor_indices = kdtree.query(points[idx], k=k)
        neighbor_normals = normals[neighbor_indices]
        mu = np.mean(neighbor_normals, axis=0)
        norm = neighbor_normals - mu
        cov = np.cov(norm.T)
        eig_val, _ = np.linalg.eig(cov)
        sorted_idx = np.argsort(eig_val)[::-1]
        eig_val = eig_val[sorted_idx]
        eig_val_norm = eig_val / np.sum(eig_val)
        eigenvalues.append(eig_val_norm)
    return eigenvalues  # Return eigenvalues instead of storing in shared array

if __name__ == "__main__": 
    dataname = "C:/Users/chris/Desktop/Documents/NewData/CircularVentilationGrateExtraCleanedFull.ply"
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
        normals_chunks = pool.starmap(find_plane_directions, [(chunk_indices, myarray) for chunk_indices in chunked_indices])
    first_pca_duration = time.time() - start_time_first_pca
    print(f"First PCA time: {first_pca_duration:.2f} seconds")

    # Flatten normals to a single array in order
    all_normals = np.vstack(normals_chunks)

    # Second pass: Compute normal variation
    start_time_second_pca = time.time()
    with multiprocessing.Pool(processes=4, initializer=init_kdtree, initargs=(kdtree,)) as pool:
        eigenvalues_chunks = pool.starmap(calculate_normal_variation, [(chunk_indices, myarray, all_normals) for chunk_indices in chunked_indices])
    second_pca_duration = time.time() - start_time_second_pca
    print(f"Second PCA time: {second_pca_duration:.2f} seconds")

    # Flatten eigenvalues to a single array in order
    all_eigenvalues = np.vstack(eigenvalues_chunks)

    # Print a few results to verify
    print("First 5 normals:")
    print(all_normals[:5])
    print("First 5 eigenvalues:")
    print(all_eigenvalues[:5])
