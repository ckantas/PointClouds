import open3d as o3d
import numpy as np
import os

def densify_point_cloud(point_cloud, thickness, interval):
    """
    Densify a point cloud by adding copies along the Z-axis at regular intervals.

    Parameters:
    - point_cloud: Open3D PointCloud object
    - thickness: Desired thickness along the Z-axis
    - interval: Thickness interval between layers

    Returns:
    - densified_point_cloud: Densified Open3D PointCloud object
    """
    # Extract point coordinates from the original point cloud
    points = np.asarray(point_cloud.points)

    # Initialize an empty array for densified points
    densified_points = []

    # Add layers at regular intervals until reaching the desired thickness
    current_thickness = 0.0
    while current_thickness <= thickness:
        # Duplicate points along the Z-axis to achieve thickness
        duplicated_positive = points + [0, 0, current_thickness]
        duplicated_negative = points - [0, 0, current_thickness]

        # Append duplicated points to the densified array
        densified_points.extend(duplicated_positive)
        densified_points.extend(duplicated_negative)

        # Move to the next layer
        current_thickness += interval

    # Create a new PointCloud object with densified points
    densified_point_cloud = o3d.geometry.PointCloud()
    densified_point_cloud.points = o3d.utility.Vector3dVector(np.array(densified_points))

    return densified_point_cloud

# Example usage
if __name__ == "__main__":
    # Load your original point cloud using Open3D
    point_cloud_path = '/home/chris/Code/PointClouds/data/FLIPscans/GrateAndCover/Cover/CoverCleaned290k.ply'
    original_point_cloud = o3d.io.read_point_cloud(point_cloud_path)

    # Set the desired thickness and interval
    thickness = 4.0
    interval = 0.3

    # Densify the point cloud
    densified_point_cloud = densify_point_cloud(original_point_cloud, thickness, interval)

    # Visualize the original and densified point clouds
    o3d.visualization.draw_geometries([original_point_cloud, densified_point_cloud])
    
    output_path = os.path.splitext(point_cloud_path)[0] + '_extra_height.ply'
    o3d.io.write_point_cloud(output_path, densified_point_cloud)