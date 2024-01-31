import open3d as o3d
import numpy as np
import os 
import sys 
import copy 

input_folder_path = '/home/chris/Code/PointClouds/data/noisetest/NoisyPLY'
output_folder_path = '/home/chris/Code/PointClouds/data/noisetest/DsNoisyPLY'
file_list = sorted(os.listdir(input_folder_path))
voxel_list_initial = np.arange(0.5, 5.1, 0.1).tolist()
voxel_list = [round(num,1) for num in voxel_list_initial]

def voxel_downsampling(point_cloud, voxel_size):
    voxel_down_pcd = point_cloud.voxel_down_sample(voxel_size=voxel_size)
    return voxel_down_pcd

def uniform_downsampling(point_cloud, k_points):
    uniform_down_pcd = point_cloud.uniform_down_sample(every_k_points=k_points)
    return uniform_down_pcd

type_repeat = True
while type_repeat == True:
    type = input('Choose type of downsampling between uniform and voxel (or program exit). [u/v/e]: ')
    if type == 'e':
        print('Exiting program')
        sys.exit()
    elif type == 'u' or type =='v':
        points_repeat = True
        while points_repeat == True:
            target_num_points = input('Choose number of points to downsample to, range 10000-50000: ')
            points_repeat = False
            try:
                _ = int(target_num_points)
                if int(target_num_points) >=10000 and int(target_num_points) <=50000:
                    points_repeat = False
                else:
                    print('Input out of bounds. Try again.')
                    points_repeat = True
            except:
                print('Invalid input. Try again.')
                points_repeat = True
        type_repeat = False
    else:
        print('Invalid input. Please try again.')

for i, point_cloud_path in enumerate(file_list):
    full_input_path = os.path.join(input_folder_path,point_cloud_path)

    output_filename = os.path.splitext(point_cloud_path)[0] + '_ds.ply'
    full_output_path = os.path.join(output_folder_path,output_filename)

    pcd = o3d.io.read_point_cloud(full_input_path)
    original_num_points = len(pcd.points)

    print(70*'-')
    print(f'[{i+1}/{len(file_list)}] Processing {point_cloud_path}. Original number of points: {original_num_points}')
    if type == 'u':
        k_points = int(round(original_num_points/int(target_num_points)))
        ds_pcd = uniform_downsampling(point_cloud=pcd, k_points=k_points)
        down_num_points = len(ds_pcd.points)
        print(f'[{i+1}/{len(file_list)}] Downsampled {point_cloud_path} to {down_num_points} number of points through uniform downsampling.')
        o3d.io.write_point_cloud(full_output_path, ds_pcd)
    if type == 'v':
        best_diff = None
        best_voxel_size = None
        for j, voxel_size in enumerate(voxel_list):
            ds_pcd = voxel_downsampling(point_cloud=pcd, voxel_size=voxel_size)
            down_num_points = len(ds_pcd.points)
            diff = abs(down_num_points-int(target_num_points))
            if j == 0:
                best_diff = diff
                best_voxel_size = voxel_size
            elif diff < best_diff:
                best_diff = diff
                best_voxel_size = voxel_size
            if best_diff == 0:
                break
        ds_pcd = voxel_downsampling(point_cloud=pcd, voxel_size=best_voxel_size)
        down_num_points = len(ds_pcd.points)
        print(f'[{i+1}/{len(file_list)}] Downsampled {point_cloud_path} to {down_num_points} number of points through voxel downsampling.')
        o3d.io.write_point_cloud(full_output_path, ds_pcd)

