import open3d as o3d

print('Loading and displaying ply file...')
ply_file_path = '/home/chris/Code/PointClouds/data/GrateAndCover/Cover/cleaned_txt/Scan7cleaned.ply'

point_cloud = o3d.io.read_point_cloud(ply_file_path)

num_points = len(point_cloud.points)
print(f"Number of points in the point cloud: {num_points}")

o3d.visualization.draw_geometries([point_cloud])

print('Done')
