import open3d as o3d

point_cloud_path = '/home/chris/Code/PointClouds/data/ply_files/VentilationGrate_2_ds.ply'
output_path = point_cloud_path.split('.')[0] + '_or.ply'

def display_inlier_outlier(cloud,ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    outlier_cloud.paint_uniform_color([1,0,0])
    inlier_cloud.paint_uniform_color([0.8,0.8,0.8])

    o3d.visualization.draw_geometries([inlier_cloud,outlier_cloud])

point_cloud = o3d.io.read_point_cloud(point_cloud_path)

new_point_cloud, ind = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

print('Visualizing outliers in red and inliers in gray:')
display_inlier_outlier(point_cloud, ind)

print(f'Saving new point cloud in {output_path}')
o3d.io.write_point_cloud(output_path, new_point_cloud)

print('Done')