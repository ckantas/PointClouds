import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import math
from functools import partial
from open3d.t.geometry import TriangleMesh
import multiprocessing
import time
from scipy.spatial import cKDTree

front =  [-0.47452876114542436, 0.57451207113849134, -0.66690204300328082]
lookat = [-6.3976792217838847, 20.927374714553928, 18.659758576873813]
up =  [-0.056918726368614558, -0.77607794684805009, -0.62806311705487861]
zoom = 0.69999999999999996

def preProcessCloud(pcd, voxel_size=0.05, visualize=False, verbose=False):
    #Outlier removal
    nn = 16
    std_multiplier = 10

    filtered_pcd = pcd.remove_statistical_outlier(nn,std_multiplier)
    filtered_pcd = filtered_pcd[0]

    #Downsampling
    voxel_size = 0.01

    pcd_downsampled = filtered_pcd.voxel_down_sample(voxel_size=voxel_size)

    #Extract normals
    nn_distance = np.mean([pcd.compute_nearest_neighbor_distance()])
    radius_normals = nn_distance*4

    pcd_downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normals, max_nn=16), fast_normal_computation=True)
    # pcd_downsampled.paint_uniform_color([0.6,0.6,0.6])
    
    return pcd_downsampled

    #o3d.visualization.draw_geometries([filtered_pcd, outliers])

def multiOrderRansac(pcd, pt_to_plane_dist, visualize=False, verbose=False):

    if verbose:
        print('Using planar patches to detect number of planes')
    oboxes = pcd.detect_planar_patches(
    normal_variance_threshold_deg=20,
    coplanarity_deg=75,
    outlier_ratio=0.2,
    min_plane_edge_length=0,
    min_num_points=0,
    search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
)

    max_plane_idx = len(oboxes)  # Use this as max_plane_idx
    if verbose:
        print(f'Found {max_plane_idx} planes')
    segment_models = {}
    segments = {}
    segment_indices = {}
    main_surface_idx = 0
    largest_surface_points = 0
    rest = pcd
    rest_indices = np.arange(len(pcd.points))
    
    if verbose:
        print('Running multi-order RANSAC')

    for i in range(max_plane_idx):
        #print(f'Run {i}/{max_plane_idx} started. ', end='')
        colors = plt.get_cmap("tab20")(i)
        segment_models[i], inliers = rest.segment_plane(distance_threshold=pt_to_plane_dist,ransac_n=3,num_iterations=50000)
        segments[i] = rest.select_by_index(inliers)
        global_inliers = rest_indices[inliers]
        segment_indices[i] = global_inliers
        if len(segments[i].points) > largest_surface_points:
            largest_surface_points = len(segments[i].points) 
            main_surface_idx = i
        segments[i].paint_uniform_color(list(colors[:3]))
        rest = rest.select_by_index(inliers, invert=True)
        rest_indices = np.delete(rest_indices, inliers)
        #print('Done')

    if visualize:
        o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)],zoom=zoom,front=front,lookat=lookat,up=up)

    return segment_models, segments, segment_indices, main_surface_idx


def planarPatches(pcd):
    oboxes = pcd.detect_planar_patches(normal_variance_threshold_deg=20,coplanarity_deg=75,
                                       outlier_ratio=0.2,min_plane_edge_length=0, min_num_points=0,
                                       search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

    return oboxes

def findAnglesBetweenPlanes(segment_models, main_surface_idx):
    angles_rad = {}
    angles_deg = {}

    for i in range(len(segment_models)):
        if i != main_surface_idx:
            dot_product = np.dot(segment_models[main_surface_idx][:3], segment_models[i][:3])
            angles_rad[i] = np.arccos(np.clip(dot_product, -1, 1))
            angles_deg[i] = angles_rad[i] * 180 / np.pi
            #print(angles_deg[i])

    return angles_rad

def findIntersectionLines(segment_models, main_surface_idx):

    intersection_lines = {}

    for i in range(len(segment_models)):
        if i != main_surface_idx:
            cross_product = np.cross(segment_models[main_surface_idx][:3], segment_models[i][:3])
            if abs(abs(cross_product[2]) < 0.0001):
                print(cross_product)
                common_point = np.linalg.solve([segment_models[main_surface_idx][1:3], segment_models[i][1:3]],
                                        [-segment_models[main_surface_idx][-1], -segment_models[i][-1]])
            else:
                common_point = np.linalg.solve([segment_models[main_surface_idx][:2], segment_models[i][:2]],
                                        [-segment_models[main_surface_idx][-1], -segment_models[i][-1]])

            common_point = np.asarray([common_point[0], common_point[1], 0])
                
            intersection_lines[i] = [cross_product,common_point]

    return intersection_lines

def findAnchorPoints(segment_models, segments, intersection_lines, main_surface_idx, visualize=False):
    anchor_points = {}
    max_plane_idx = len(segment_models)
    for i in range(len(segment_models)):
        if i != main_surface_idx:
            segment_center = segments[i].get_center()
            b_vec = [sc - il for sc, il in zip(segment_center, intersection_lines[i][1])]
            a_vec = intersection_lines[i][0]/np.linalg.norm(intersection_lines[i][0])
            p_vec = np.dot(a_vec,b_vec)/np.dot(a_vec,a_vec) * a_vec
            anchor_points[i] = intersection_lines[i][1] + p_vec
    
    if visualize:
        sphere_radius = 5  # Adjust the radius based on your scale
        spheres = []
        for point in anchor_points.values():
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
            sphere.translate(point)  # Move the sphere to the anchor point
            sphere.paint_uniform_color([1, 0, 0])  # Red color for visibility
            spheres.append(sphere)

        # Visualize
        o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)]+spheres, zoom=zoom, front=front, lookat=lookat, up=up)
    
    return anchor_points

def createLine(start_point, end_point, color=[1, 0, 0]):
    points = [start_point, end_point]
    lines = [[0, 1]]
    colors = [color]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

# Function to rotate a vector by a specified angle around a given axis
def rotateVector(vector, axis, angle_degrees):
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle_radians)
    rotated_vector = np.dot(rotation_matrix, vector)
    return rotated_vector

def generateAnchorLines(anchor_points, intersection_lines, segment_models, segments, main_surface_idx, visualize=False):
    # Parameters for line visualization
    line_length = 10  # Length of the lines

    # Lists to store direction vectors
    direction_vectors = []

    # Generate the red and rotated green lines based on the projection and surface normals
    lines = []
    for i in range(len(segment_models)):
        if i != main_surface_idx:
            # Get the intersection line direction
            intersection_direction = intersection_lines[i][0]
            intersection_direction /= np.linalg.norm(intersection_direction)  # Normalize
            
            # Find a vector that is perpendicular to the intersection line
            perpendicular_direction = np.cross(intersection_direction, [0, 0, 1])
            perpendicular_direction /= np.linalg.norm(perpendicular_direction)  # Normalize
            
            # Anchor point for the line
            anchor_point = anchor_points[i]
            
            # Get the center of the bend (center of the bent surface)
            bend_center = segments[i].get_center()
            
            # Project the bend center onto the perpendicular line
            b_vec = bend_center - anchor_point
            projection_length = np.dot(b_vec, perpendicular_direction)
            projected_point = anchor_point + projection_length * perpendicular_direction
            
            # Determine the direction towards the bend
            direction_to_bend = (projected_point - anchor_point) / np.linalg.norm(projected_point - anchor_point)
            
            # Determine the direction towards the main surface (opposite direction)
            direction_to_main_surface = -direction_to_bend
            
            # Save the direction vectors
            direction_vectors.append({
                'anchor_point': anchor_point,
                'direction_to_main_surface': direction_to_main_surface,
                'direction_to_bend': direction_to_bend
            })
            
            # Draw only the red line towards the main surface
            main_surface_end_point = anchor_point + line_length * direction_to_main_surface
            main_surface_line = createLine(anchor_point, main_surface_end_point, color=[1, 0, 0])  # Red for main surface direction
            lines.append(main_surface_line)
            
            # Now draw the reversed green line based on the normal of the bent surface
            bent_surface_normal = segment_models[i][:3]  # Get the normal of the bent surface
            bent_surface_normal /= np.linalg.norm(bent_surface_normal)  # Normalize the normal
            
            # Reverse the direction of the green line by flipping the normal vector
            bent_surface_normal *= -1
            
            # Find the axis of rotation (perpendicular to both the red and green lines)
            rotation_axis = np.cross(direction_to_main_surface, bent_surface_normal)
            rotation_axis /= np.linalg.norm(rotation_axis)  # Normalize the rotation axis
            
            # Rotate the green line by 90 degrees around the calculated axis
            rotated_green_direction = rotateVector(bent_surface_normal, rotation_axis, 90)
            
            # Create the rotated green line at the anchor point
            green_end_point = anchor_point + line_length * rotated_green_direction
            green_line = createLine(anchor_point, green_end_point, color=[0, 1, 0])  # Green for bent surface normal
            lines.append(green_line)
            
            # Save the green direction vector
            direction_vectors[-1]['rotated_green_direction'] = rotated_green_direction

    return lines, direction_vectors

def drawBendEdges(pcd, bend_edges):
    # Create line set
    line_points = []  # Stores points for line visualization
    line_indices = []  # Stores line indices

    for i, (start, end) in bend_edges.items():
        line_points.append(start)
        line_points.append(end)
        line_indices.append([len(line_points) - 2, len(line_points) - 1])  # Connect last two added points

    # Convert to Open3D format
    line_points = np.array(line_points)
    line_indices = np.array(line_indices)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(line_indices)
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(line_indices))  # Red color for visibility

    pcd.paint_uniform_color([0.6, 0.6, 0.6])
    # Visualize point cloud and lines
    o3d.visualization.draw_geometries([pcd, line_set])

kdtree = None  # Global k-d tree

def initKdtree(tree):
    """Initialize k-d tree in each worker process."""
    global kdtree
    kdtree = tree

def calculateNormals(indices, points, radius=2):
    """Estimate plane normals for given indices using PCA."""
    global kdtree
    if kdtree is None:  # Ensure k-d tree is initialized
        kdtree = cKDTree(points)
    normals = []
    for idx in indices:
        neighbor_indices = kdtree.query_ball_point(points[idx], radius)
        if not neighbor_indices:  
            normals.append(np.array([0, 0, 0]))  # Default normal if no neighbors
            continue
        
        neighbors = points[neighbor_indices]
        mean = np.mean(neighbors, axis=0)
        norm = neighbors - mean
        cov = np.cov(norm.T)
        eig_val, eig_vec = np.linalg.eigh(cov)
        sorted_idx = np.argsort(eig_val)[::-1]
        eig_vec = eig_vec[:, sorted_idx]
        plane_direction = eig_vec[:, 2]  # Smallest eigenvector is normal
        normals.append(plane_direction)
    return np.array(normals)

def alignNormalsGlobally(normals, points, radius=2):
    """Ensure all normals are consistently oriented across the entire point cloud."""
    kdtree = cKDTree(points)
    visited = np.zeros(len(normals), dtype=bool)

    queue = [0]  # Start with the first point as the reference
    visited[0] = True

    while queue:
        idx = queue.pop(0)
        reference_normal = normals[idx]

        # Find neighboring points
        neighbor_indices = kdtree.query_ball_point(points[idx], radius)

        for neighbor_idx in neighbor_indices:
            if visited[neighbor_idx]:
                continue  # Skip already visited neighbors

            # Flip neighbor normal if it points in the opposite direction
            if np.dot(normals[neighbor_idx], reference_normal) < 0:
                normals[neighbor_idx] *= -1

            visited[neighbor_idx] = True
            queue.append(neighbor_idx)  # Add to queue for further checking

    return normals

def calculateNormalStandardDeviation(indices, points, normals, radius=2):
    """Compute standard deviation of normal directions for given indices."""
    standard_deviations = []

    for idx in indices:
        neighbor_indices = kdtree.query_ball_point(points[idx], radius)
        if not neighbor_indices:
            standard_deviations.append(0)  
            continue

        neighbor_normals = normals[neighbor_indices]
        #reference_normal = neighbor_normals[0]
        #dot_products = np.dot(neighbor_normals, reference_normal)
        #aligned_normals = neighbor_normals * np.sign(dot_products)[:, np.newaxis]

        std_dev = np.std(neighbor_normals, axis=0)
        variation_measure = np.sum(std_dev)
        standard_deviations.append(variation_measure)

    return np.array(standard_deviations)

def calculatePointwiseNormalVariance(pcd, radius=2, num_chunks=16, num_workers=4, verbose=False):
    """
    Compute normal variation for a given point cloud.
    
    Args:
        pcd (open3d.geometry.PointCloud): Input point cloud.
        radius (float): Neighborhood search radius.
        num_chunks (int): Number of chunks for parallel processing.
        num_workers (int): Number of worker processes.

    Returns:
        np.ndarray: Normalized variation values for each point.
    """
    myarray = np.asarray(pcd.points)
    indices = np.arange(len(myarray))
    kdtree = cKDTree(myarray)  # Build k-d tree
    chunk_size = len(myarray) // num_chunks
    chunked_indices = [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]

    if verbose: 
        print(f"Calculating pointwise PCA using {num_workers} workers and {num_chunks} chunks. This may take a while... (approx. 1 minute per 1M points)")
    # First pass: Compute normals
    start_time_first_pca = time.time()
    with multiprocessing.Pool(processes=num_workers, initializer=initKdtree, initargs=(kdtree,)) as pool:
        normals_chunks = pool.starmap(calculateNormals, [(chunk, myarray, radius) for chunk in chunked_indices])
    first_pca_duration = time.time() - start_time_first_pca
    if verbose:
        print(f"PCA calculation time: {first_pca_duration:.2f} seconds")

    all_normals = np.vstack(normals_chunks)  # Flatten normal array
    all_normals = alignNormalsGlobally(all_normals, myarray, radius=2)
    # Second pass: Compute standard deviation-based variation
    if verbose:
        print("Calculating pointwise standard deviation. This may take a while... (approx. 30 sec per 1M points)")
    start_time_standard_deviations = time.time()
    with multiprocessing.Pool(processes=num_workers, initializer=initKdtree, initargs=(kdtree,)) as pool:
        standard_deviation_chunks = pool.starmap(calculateNormalStandardDeviation, [(chunk, myarray, all_normals, radius) for chunk in chunked_indices])
    standard_deviations_duration = time.time() - start_time_standard_deviations
    if verbose:
        print(f"Standard deviation calculation: {standard_deviations_duration:.2f} seconds")

    standard_deviations = np.hstack(standard_deviation_chunks)  # Flatten

    # Normalize variation values
    max_variation = np.max(standard_deviations) if len(standard_deviations) > 0 else 1
    normalized_variation = standard_deviations / max_variation

    return all_normals, normalized_variation

def getCorePoints(pointwise_variance, percentile=90):
    threshold = np.percentile(pointwise_variance, percentile)  # Compute 90th percentile
    core_indices = np.where(pointwise_variance >= threshold)[0]  # Indices of core points
    
    return core_indices

def is_within_bend_limits(point, bend_edge):
    """
    Checks if a point is within the bend limits by projecting it onto the bend vector.

    Args:
        point (np.ndarray): The 3D coordinates of the point.
        bend_edge (tuple): A tuple containing (start_point, end_point) for the current anchor.

    Returns:
        bool: True if the point is within the bend region, False otherwise.
    """
    point = np.array(point)  # Ensure NumPy array
    start_point, end_point = map(np.array, bend_edge)  # Convert both points to NumPy arrays

    # **Step 1: Compute the bend vector**
    bend_vector = end_point - start_point
    bend_length = np.linalg.norm(bend_vector)  # Total bend length
    bend_vector /= bend_length  # Normalize

    # **Step 2: Project the point onto the bend vector**
    point_vector = point - start_point
    projection_scalar = np.dot(point_vector, bend_vector)

    # **Step 3: Check if projection is within the segment**
    return 0 <= projection_scalar <= bend_length


def growRegionsAroundIntersections(anchor_points_dict, core_indices, pointwise_variance, points, bend_edges, search_radius=1.5, min_neighbors=20, variance_percentile=90):
    """
    Grows a region/cluster around anchor points, enforcing bend constraints.

    Args:
        anchor_points_dict (dict): Dictionary mapping each anchor index to its 3D coordinate.
        core_indices (np.ndarray): Indices of core points.
        pointwise_variance (np.ndarray): Normal variance values for each point.
        points (np.ndarray): 3D coordinates of all points (Px3).
        bend_edges (dict): Dictionary mapping bend regions to their start and end points.
        search_radius (float): Radius for neighbor search.
        min_neighbors (int): Minimum number of neighbors to consider a core point.
        variance_percentile (float): Minimum variance percentile for expansion.

    Returns:
        clusters (dictionary of sets): Dict of clusters, each containing point indices.
    """
    kdtree = cKDTree(points)  # KD-tree for fast neighbor search
    #core_kdtree = cKDTree(points[core_indices])  # KD-tree for core points only
    variance_threshold = np.percentile(pointwise_variance, variance_percentile)  # Compute variance cutoff
    clusters = {}  # Store clusters
    
    for anchor_idx, anchor_point in anchor_points_dict.items():
        anchor_point = np.array(anchor_point)
        # **Step 1: Find Closest Core Point**
        neighbor_distances, neighbor_indices = kdtree.query(anchor_point, k=20, distance_upper_bound=search_radius)
        
        core_neighbors = [idx for idx in neighbor_indices if idx in core_indices]
        
        if not core_neighbors:  # If no core points among the 20 closest neighbors, skip this anchor
            continue

        seed_idx = min(core_neighbors, key=lambda idx: neighbor_distances[list(neighbor_indices).index(idx)])
        cluster = set([seed_idx])  # Start cluster from this core point
        to_expand = [seed_idx]  # Queue for expansion

        # Get corresponding bend edge for this anchor
        bend_edge = bend_edges.get(anchor_idx, None)

        # **Step 2: Region Growing**
        while to_expand:
            current_idx = to_expand.pop()
            neighbors = kdtree.query_ball_point(points[current_idx], search_radius)  # Get neighbors
            
            for neighbor in neighbors:
                neighbor_point = points[neighbor]  # Get 3D coordinates

                # **Check if within variance threshold**
                if neighbor not in cluster and pointwise_variance[neighbor] >= variance_threshold:

                    # **Check if within bend limits**
                    if bend_edge and not is_within_bend_limits(neighbor_point, bend_edge):
                        continue  # Skip points outside the bend

                    cluster.add(neighbor)
                    to_expand.append(neighbor)  # Expand further

        clusters[anchor_idx] = cluster  # Save completed cluster

    return clusters


