# Python program to understand 
# the concept of pool 
import multiprocessing 
import os 
import time
import numpy as np
from scipy.spatial import cKDTree

kdtree = None
def init_kdtree(tree):
    global kdtree
    kdtree = tree

def find_plane_directions(chunk, points, k=5):
    # Access the global KD-tree and query the nearest neighbors for a single point
    plane_directions = []
    for point in chunk:
        print("Worker process id for point {}: {}".format(point, os.getpid()))
        _, indices = kdtree.query(point, k=k)
        neighbors = points[indices]
        mean = np.mean(neighbors, axis=0)
        norm = neighbors - mean
        cov = np.cov(norm.T)
        eig_val, eig_vec = np.linalg.eigh(cov)
        sorted_idx = np.argsort(eig_val)[::-1]
        eig_val = eig_val[sorted_idx]
        eig_vec = eig_vec[:, sorted_idx]
        plane_direction = eig_vec[:, 2]
        print(f"Point {point}: Nearest neighbors' indices -> {indices}")
        plane_directions.append(plane_direction)
    
    return plane_directions

if __name__ == "__main__": 

    # input list 
    myarray = np.random.rand(5000, 3)
    kdtree = cKDTree(myarray)
    # Define number of chunks
    num_chunks = 10
    chunk_size = len(myarray) // num_chunks

    chunks = [myarray[i:i + chunk_size] for i in range(0, len(myarray), chunk_size)]
    #creating a pool object 
    p = multiprocessing.Pool(processes=4, initializer=init_kdtree, initargs=(kdtree,)) 
    results = p.starmap(find_plane_directions, [(chunk, myarray) for chunk in chunks])
    #map list to target function 
    #plane_directions = p.map(find_chunk_mean, chunks) 

    print(results) 
