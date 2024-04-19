import open3d as o3d
import numpy as np

# Load the point cloud
pcd = o3d.io.read_point_cloud("/home/chris/Code/PointClouds/data/ply/Cover/CoverCleaned.ply")

# Since the point cloud has extra scalar fields (intensity),
# we need to remove it to only keep x, y, z coordinates
xyz_only = np.asarray(pcd.points)  # Extract only x, y, z coordinates
pcd.points = o3d.utility.Vector3dVector(xyz_only)

# Estimate normals and orient them consistently
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
pcd.orient_normals_consistent_tangent_plane(50)

# Perform surface reconstruction
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)

# Remove vertex colors and normals
if mesh.has_vertex_colors():
    mesh.vertex_colors = o3d.utility.Vector3dVector([])  # Remove vertex colors
if mesh.has_vertex_normals():
    mesh.vertex_normals = o3d.utility.Vector3dVector([])  # Remove vertex normals

# Save the mesh as an OBJ file
o3d.io.write_triangle_mesh("/home/chris/Code/PointClouds/data/obj/CoverCleaned.obj", mesh, write_ascii=True)