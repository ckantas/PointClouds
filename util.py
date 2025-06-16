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

def preProcessSimple(pcd, voxel_size=0.05, visualize=False, verbose=False):
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

def multiOrderRansacAdvanced(pcd, pt_to_plane_dist, visualize=False, verbose=False):
    if verbose:
        print('Identifying main plane')
    
    plane_model, inliers = pcd.segment_plane(distance_threshold=pt_to_plane_dist, ransac_n=3, num_iterations=1000)
    main_plane = pcd.select_by_index(inliers)
    remaining = pcd.select_by_index(inliers, invert=True)
    remaining_indices = np.setdiff1d(np.arange(len(pcd.points)), inliers)

    if verbose:
        print('Clustering remaining points')
    labels = np.array(remaining.cluster_dbscan(eps=1.2, min_points=15))

    num_clusters = labels.max() + 1
    if verbose:
        print(f"Found {num_clusters} disconnected clusters")

    # Remove noise (label == -1)
    clean_indices = np.where(labels >= 0)[0]
    remaining_clean = remaining.select_by_index(clean_indices)
    clean_labels = labels[clean_indices]
    num_clean_clusters = clean_labels.max() + 1
    
    def angle_between_normals(n1, n2):
        cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
        return np.arccos(cos_angle) * 180.0 / np.pi

    # Get normal of main plane
    main_plane_normal = np.array(plane_model[:3])
    main_plane_normal = main_plane_normal / np.linalg.norm(main_plane_normal)

    angle_threshold = 10  # degrees

    if verbose:
        print('Fitting planes to remaining clusters and filtering')

    segment_models = {}
    segments = {}
    segment_indices = {}
    filtered_ids = []
    main_surface_idx = 0
    # Paint and add main surface
    main_plane.paint_uniform_color([0.3, 0.3, 1.0])
    segments[main_surface_idx] = main_plane
    segment_models[main_surface_idx] = plane_model
    segment_indices[main_surface_idx] = np.array(inliers)
    # Colormap
    cmap = plt.get_cmap("tab20")
    color_idx = 0
    index = 0

    for i in range(num_clean_clusters):
        indices = np.where(clean_labels == i)[0]
        cluster = remaining_clean.select_by_index(indices)

        if len(cluster.points) < 800:
            continue

        try:
            cluster_plane, inliers = cluster.segment_plane(distance_threshold=pt_to_plane_dist, ransac_n=3, num_iterations=1000)
        except:
            continue

        cluster_normal = np.array(cluster_plane[:3])
        cluster_normal /= np.linalg.norm(cluster_normal)
        angle = angle_between_normals(main_plane_normal, cluster_normal)

        if angle > angle_threshold:
            cluster_id = index + 1
            color = cmap(color_idx / 20)[:3]
            cluster.paint_uniform_color(color)
            segment_models[cluster_id] = cluster_plane
            segments[cluster_id] = cluster
            # Map local cluster inliers to global indices in pcd
            inliers_global = remaining_indices[clean_indices[indices[inliers]]]  # indices in full pcd

            segment_indices[cluster_id] = inliers_global
            filtered_ids.append(cluster_id)
            
            index += 1
            color_idx += 1  # advance for next unique color
        else:
            # Optional: keep red color for discarded ones, or skip saving them
            pass

    if visualize:
        o3d.visualization.draw_geometries([segments[i] for i in segments])

    return segment_models, segments, segment_indices, main_surface_idx

def project_point_onto_plane(point, plane_model):
    # plane_model = [a, b, c, d]
    normal = np.array(plane_model[:3])
    d = plane_model[3]
    normal = normal / np.linalg.norm(normal)
    distance = np.dot(normal, point) + d
    return point - distance * normal

def project_point_onto_line(point, line_point, line_dir):
    """
    Projects a point onto a 3D line defined by a point and direction.
    """
    line_dir = line_dir / np.linalg.norm(line_dir)
    vec_to_point = point - line_point
    projection_length = np.dot(vec_to_point, line_dir)
    projected = line_point + projection_length * line_dir
    return projected

def is_wrap_around_bend(center_point, angle, main_plane_model, intersection_line, pcd_tree, search_radius=0.5, offset_distance=1.5):
    """
    Determines if a bend wraps around the main surface using two strategies:
    - For near-90° bends: project center to main plane and offset perpendicularly to crease
    - Otherwise: use direct projection to main plane
    """
    n_main = np.array(main_plane_model[:3])
    n_main /= np.linalg.norm(n_main)

    if 75 < angle < 105:
        #print(f"Near-90° bend detected with angle {angle:.2f}°")

        # Project center onto main surface
        projected = project_point_onto_plane(center_point, main_plane_model)

        # Project that onto the intersection line
        line_dir = np.array(intersection_line[0])
        line_point = np.array(intersection_line[1])
        line_proj = project_point_onto_line(projected, line_point, line_dir)

        # Compute offset direction away from the line
        offset_dir = projected - line_proj
        offset_dir /= np.linalg.norm(offset_dir)

        # Move away from crease
        probe_point = projected + offset_distance * offset_dir
    else:
        # Use direct projection
        probe_point = project_point_onto_plane(center_point, main_plane_model)

    [k, _, _] = pcd_tree.search_radius_vector_3d(probe_point, search_radius)
    #if 75 < angle < 105:
        #print(k, "points found within search radius for near-90° bend")
    return k > 0

def flip_normals_by_bend_orientation(segment_models, intersection_lines, segments, main_surface_idx, pcd_tree, search_radius=0.8, verbose=False):
    """
    Uses geometry and context to determine correct normal orientation
    for each inclined plane based on whether it wraps around the main surface.
    """
    aligned_models = {}
    main_normal = np.array(segment_models[main_surface_idx][:3])
    main_normal /= np.linalg.norm(main_normal)

    if main_normal[2] < 0:
        if verbose:
            print("🔄 Flipping main surface normal to face upward")
        segment_models[main_surface_idx] = -1 * segment_models[main_surface_idx]
        main_normal = -1 * main_normal
        
    main_plane_model = segment_models[main_surface_idx]

    for idx, model in segment_models.items():
        if idx == main_surface_idx:
            aligned_models[idx] = model
            continue

        intersection_line = intersection_lines.get(idx)
        if intersection_line is None:
            aligned_models[idx] = model
            continue

        center = np.array(segments[idx].get_center())
        n_inclined = np.array(model[:3])
        n_inclined /= np.linalg.norm(n_inclined)

        dot = np.dot(main_normal, n_inclined)
        angle = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))

        if verbose:
            print(f"[Segment {idx}] dot = {dot:.3f} → angle = {angle:.2f}°")

        is_wrap = is_wrap_around_bend(
            center_point=center,
            angle=angle,
            main_plane_model=main_plane_model,
            intersection_line=intersection_line,
            pcd_tree=pcd_tree,
            search_radius=search_radius,
            offset_distance=1.5
        )

        if verbose:
            if 75 < angle < 105:
                print(f"→ near-90° bend → wrap = {is_wrap}")

        # Final flip logic — minimal and clear
        should_flip = (is_wrap and angle < 90) or (not is_wrap and angle > 90)

        if should_flip:
            aligned_models[idx] = -1 * model
            if verbose:
                print("↪️  Flipped")
        else:
            aligned_models[idx] = model
            if verbose:
                print("✅ Kept as-is")

    return aligned_models

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

def findIntersectionLinesLeastSquares(segment_models, main_surface_idx):
    """
    Computes intersection lines between the main plane and other segment planes.
    Each line is represented by (direction_vector, point_on_line).
    """
    intersection_lines = {}

    main_plane = segment_models[main_surface_idx]
    n1 = main_plane[:3]
    d1 = -main_plane[3]

    for i, plane in segment_models.items():
        if i == main_surface_idx:
            continue

        n2 = plane[:3]
        d2 = -plane[3]

        # Direction of intersection = cross product of normals
        direction = np.cross(n1, n2)
        if np.linalg.norm(direction) < 1e-6:
            continue  # Planes are parallel or identical

        # Build coefficient matrix and solve for point on line
        A = np.array([n1, n2, direction])
        b = np.array([d1, d2, 0])

        # Use least-squares to solve A x = b (more stable than direct solve)
        try:
            point_on_line = np.linalg.lstsq(A, b, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue  # Skip if system is badly conditioned

        intersection_lines[i] = [direction, point_on_line]

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

def drawBendEdgesWithCylinders(pcd, bend_edges, radius=0.3, color=[1, 0, 0]):
    geometries = [pcd.paint_uniform_color([0.6, 0.6, 0.6]) or pcd]
    for start, end in bend_edges.values():
        cylinder = create_cylinder_between_points(np.array(start), np.array(end), radius, color)
        geometries.append(cylinder)
    o3d.visualization.draw_geometries(geometries)

def create_cylinder_between_points(p1, p2, radius=0.3, color=[1, 0, 0]):
    height = np.linalg.norm(p2 - p1)
    direction = (p2 - p1) / height
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius, height)
    cylinder.paint_uniform_color(color)

    # Align z-axis to direction
    z = np.array([0, 0, 1])
    axis = np.cross(z, direction)
    angle = np.arccos(np.clip(np.dot(z, direction), -1.0, 1.0))
    if np.linalg.norm(axis) > 1e-6:
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis / np.linalg.norm(axis) * angle)
        cylinder.rotate(R, center=(0, 0, 0))

    # Move to midpoint
    midpoint = (p1 + p2) / 2
    cylinder.translate(midpoint)
    return cylinder

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

def create_normal_arrow(center, normal, length=6.0, color=[1, 0, 0]):
    start = center
    end = center + length * normal
    points = [start, end]
    lines = [[0, 1]]
    colors = [color]

    arrow = o3d.geometry.LineSet()
    arrow.points = o3d.utility.Vector3dVector(points)
    arrow.lines = o3d.utility.Vector2iVector(lines)
    arrow.colors = o3d.utility.Vector3dVector(colors)
    return arrow

def draw_normal_arrows(segment_models, segments, main_surface_idx):
    # Create arrows for each plane
    arrows = []
    for idx, model in segment_models.items():
        normal = np.array(model[:3])
        normal /= np.linalg.norm(normal)
        center = np.array(segments[idx].get_center())

        color = [0, 0, 1] if idx == main_surface_idx else [0, 1, 0]
        arrow = create_normal_arrow(center, normal, length=10.0, color=color)
        arrows.append(arrow)

    # Combine segments + arrows for visualization
    geometries = list(segments.values()) + arrows
    o3d.visualization.draw_geometries(geometries)

def create_normal_arrow_mesh(start_point, direction, length=10.0, cylinder_radius=0.2, cone_radius=0.4, color=[0.1, 0.7, 0.3]):
    direction = direction / np.linalg.norm(direction)
    end_point = start_point + direction * length

    # Cylinder
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=cylinder_radius, height=length * 0.8)
    cylinder.paint_uniform_color(color)

    z_axis = np.array([0, 0, 1])
    rot_axis = np.cross(z_axis, direction)
    if np.linalg.norm(rot_axis) > 1e-6:
        rot_axis /= np.linalg.norm(rot_axis)
        angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis * angle)
        cylinder.rotate(R, center=np.zeros(3))

    cylinder.translate(start_point + direction * (length * 0.4))

    # Cone
    cone_height = length * 0.2
    cone = o3d.geometry.TriangleMesh.create_cone(radius=cone_radius, height=cone_height)
    cone.paint_uniform_color(color)
    cone.rotate(R, center=np.zeros(3))
    cone.translate(end_point - direction * (cone_height * 1))

    return [cylinder, cone]

def draw_normal_arrows_with_geometry(segment_models, segments, main_surface_idx=None, random_flip=True):
    arrows = []

    for idx, model in segment_models.items():
        normal = np.array(model[:3])
        normal = normal / np.linalg.norm(normal)

        if random_flip and np.random.rand() > 0.5:
            normal *= -1  # Flip 180 degrees

        center = np.array(segments[idx].get_center())
        arrow_parts = create_normal_arrow_mesh(center, normal, length=8.0)
        arrows.extend(arrow_parts)

    o3d.visualization.draw_geometries(list(segments.values()) + arrows)

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

def calculatePointwiseNormalVariance(pcd, pca_radius=2, variance_radius = 2, num_chunks=16, num_workers=4, verbose=False):
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
        normals_chunks = pool.starmap(calculateNormals, [(chunk, myarray, pca_radius) for chunk in chunked_indices])
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
        standard_deviation_chunks = pool.starmap(calculateNormalStandardDeviation, [(chunk, myarray, all_normals, variance_radius) for chunk in chunked_indices])
    standard_deviations_duration = time.time() - start_time_standard_deviations
    if verbose:
        print(f"Standard deviation calculation: {standard_deviations_duration:.2f} seconds")

    standard_deviations = np.hstack(standard_deviation_chunks)  # Flatten

    # Normalize variation values
    max_variation = np.max(standard_deviations) if len(standard_deviations) > 0 else 1
    normalized_variation = standard_deviations / max_variation

    return all_normals, normalized_variation


def calculatePointwiseNormalVariance_open3d(
    pcd, 
    pca_radius=1.7, 
    variance_radius=0.8, 
    num_chunks=16, 
    num_workers=4, 
    verbose=False
):
    """
    Use Open3D to estimate normals, then compute standard deviation-based variation using multiprocessing.

    Args:
        pcd (open3d.geometry.PointCloud): Input point cloud.
        pca_radius (float): Radius for Open3D normal estimation.
        variance_radius (float): Radius for standard deviation computation.
        num_chunks (int): Number of chunks for parallel processing.
        num_workers (int): Number of worker processes.
        verbose (bool): Print timing information.

    Returns:
        tuple: (Nx3 normals, normalized variation per point)
    """
    points = np.asarray(pcd.points)
    indices = np.arange(len(points))
    chunk_size = len(points) // num_chunks
    chunked_indices = [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]
    global kdtree
    kdtree = cKDTree(points)

    if verbose:
        print(f"Estimating normals using Open3D with radius={pca_radius}...")

    start_time_normals = time.time()
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=pca_radius, max_nn=50)
    )
    pcd.orient_normals_consistent_tangent_plane(k=20)
    all_normals = np.asarray(pcd.normals).copy()
    normals_duration = time.time() - start_time_normals

    if verbose:
        print(f"Normal estimation (Open3D): {normals_duration:.2f} seconds")
        print("Calculating pointwise standard deviation with multiprocessing...")

    start_time_std = time.time()
    with multiprocessing.Pool(processes=num_workers, initializer=initKdtree, initargs=(kdtree,)) as pool:
        std_chunks = pool.starmap(
            calculateNormalStandardDeviation, 
            [(chunk, points, all_normals, variance_radius) for chunk in chunked_indices]
        )
    std_duration = time.time() - start_time_std

    if verbose:
        print(f"Standard deviation calculation: {std_duration:.2f} seconds")

    standard_deviations = np.hstack(std_chunks)
    max_variation = np.max(standard_deviations) if len(standard_deviations) > 0 else 1
    normalized_variation = standard_deviations / max_variation

    return all_normals, normalized_variation


def getCorePoints(pointwise_variance, percentile=65):
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

def compute_normal_derivative_pca(normals):
    """Compute dominant direction of normal variation using PCA."""
    cov = np.cov(normals.T)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    return eig_vecs[:, np.argmax(eig_vals)]  # Largest eigenvector = main change direction
    
def calculate_normal_derivatives(normals, points, radius, verbose=False):
    kdtree = cKDTree(points)
    derivatives = []
    for i, point in enumerate(points):
        # Get neighborhood
        if verbose:
            print(f"Computing normal derivative {i}/{len(points)}")
        neighbor_indices = kdtree.query_ball_point(point, radius)

        if len(neighbor_indices) < 3:
            continue  # Skip small neighborhoods

        neighbor_normals = normals[neighbor_indices]

        # Compute normal derivative
        normal_gradient = compute_normal_derivative_pca(neighbor_normals)  # or self.compute_normal_derivative

        derivatives.append(normal_gradient)
    return np.array(derivatives)