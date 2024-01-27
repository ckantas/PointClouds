import open3d as o3d
import sys

def voxel_downsampling(point_cloud, voxel_size):
    voxel_down_pcd = point_cloud.voxel_down_sample(voxel_size=voxel_size)
    return voxel_down_pcd

def uniform_downsampling(point_cloud, k_points):
    uniform_down_pcd = point_cloud.uniform_down_sample(every_k_points=k_points)
    return uniform_down_pcd

point_cloud_path = '/home/chris/Code/PointClouds/data/point_cloud_files/VentilationGrate_2.ply'

pcd = o3d.io.read_point_cloud(point_cloud_path)
orig_num_points = len(pcd.points)

voxel_size = 2.5
k_points = 5

type = None
while type == None:
    type = input('Choose voxel downsampling, uniform downsampling or exit code [v/u/e]:')
    if type == 'v':
        print(f'Downsampling point cloud using voxel downsampling (voxel_size: {voxel_size})')
        ds_pcd = voxel_downsampling(point_cloud=pcd, voxel_size=voxel_size)
    elif type == 'u':
        print(f'Downsampling point cloud using uniform downsampling (every_k_points = {k_points})')
        ds_pcd = uniform_downsampling(point_cloud=pcd, k_points=k_points)
    elif type == 'e':
        print('Exiting program')
        sys.exit()
    else:
        print('Invalid input. Try again')
        type = None

down_num_points = len(ds_pcd.points)
print(f'Done!\n Original number of points: {orig_num_points}\n Number of points after downsampling: {down_num_points}\n Visualizing point cloud...')

o3d.visualization.draw_geometries([ds_pcd])

save = None
while save == None:
    save= input('Save downsampled point cloud? [y/n]:')
    if save == 'y':
        if type == 'v':
            output_path = point_cloud_path.split('.')[0]+'_voxelds.ply'
            print(print(f'Saving in {output_path}'))
            o3d.io.write_point_cloud(output_path, ds_pcd)
        elif type == 'u':
            output_path = point_cloud_path.split('.')[0]+'_uniformds.ply'
            print(print(f'Saving in {output_path}'))
            o3d.io.write_point_cloud(output_path, ds_pcd)
    elif save == 'n':
        print('Exiting program')
        sys.exit()
    else:
        print('Invalid input. Try again')
        save = None