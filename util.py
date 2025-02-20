import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import math
from functools import partial
from open3d.t.geometry import TriangleMesh

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

def detectPlanes(pcd, method='planar', pt_to_plane_dist=0.4, visualize=False, verbose=False):
    if method == 'ransac':
        segment_model, inliers = pcd.segment_plane(distance_threshold=pt_to_plane_dist,ransac_n=3,num_iterations=50000)
        segment = pcd.select_by_index(inliers)
        segment.paint_uniform_color([1,0,0])
        rest = pcd.select_by_index(inliers, invert=True)
        if visualize:
            o3d.visualization.draw_geometries([segment, rest],zoom=zoom,front=front,lookat=lookat,up=up)
        return segment_model, segment, rest
    elif method == 'planar':
        oboxes = planarPatches(pcd)

        geometries = []
        for obox in oboxes:
            mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obox, scale=[1, 1, 0.0001])
            mesh.paint_uniform_color(obox.color)
            geometries.append(mesh)
            geometries.append(obox)
        if visualize:
            o3d.visualization.draw_geometries(geometries,zoom=zoom,front=front,lookat=lookat,up=up)
        return oboxes
    
def multiOrderRansac(pcd, max_plane_idx, pt_to_plane_dist, visualize=False, verbose=False):
    max_plane_idx = 7
    pt_to_plane_dist = 0.4

    segment_models = {}
    segments = {}
    main_surface_idx = 0
    largest_surface_points = 0
    rest = pcd

    for i in range(max_plane_idx):
        #print(f'Run {i}/{max_plane_idx} started. ', end='')
        colors = plt.get_cmap("tab20")(i)
        segment_models[i], inliers = rest.segment_plane(distance_threshold=pt_to_plane_dist,ransac_n=3,num_iterations=50000)
        segments[i] = rest.select_by_index(inliers)
        if len(segments[i].points) > largest_surface_points:
            largest_surface_points = len(segments[i].points) 
            main_surface_idx = i
        segments[i].paint_uniform_color(list(colors[:3]))
        rest = rest.select_by_index(inliers, invert=True)
        #print('Done')

    if visualize:
        o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)]+[rest],zoom=zoom,front=front,lookat=lookat,up=up)

    return segment_models, segments, main_surface_idx


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

def draw_bend_edges(pcd, bend_edges):
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

