import open3d as o3d
import sys
import os

point_cloud_path = '/home/chris/Code/PointClouds/data/FLIPscans/GrateAndCover/Cover/CoverCleaned290k_extra_height.ply'
def_voxel_size = 2.5
def_k_points = 5

def voxel_downsampling(point_cloud, voxel_size):
    print(f'Downsampling point cloud using voxel downsampling (voxel_size: {voxel_size})')
    voxel_down_pcd = point_cloud.voxel_down_sample(voxel_size=voxel_size)
    return voxel_down_pcd

def uniform_downsampling(point_cloud, k_points):
    print(f'Downsampling point cloud using uniform downsampling (every_k_points = {k_points})')
    uniform_down_pcd = point_cloud.uniform_down_sample(every_k_points=k_points)
    return uniform_down_pcd

def downsample_pcd(pcd, type):
    voxel_size = 0
    k_points = 0
    redo = True
    orig_num_points = len(pcd.points)

    if type == 'v':
        while redo == True:
            voxel_size = input(f'Choose voxel_size (default = {def_voxel_size}, allowed range 0.5-5):')
            flag = True
            while flag == True:
                try: 
                    _ = float(voxel_size)
                except:
                    print('Value entered is not a number')
                    flag = False
                    continue
                if float(voxel_size) >= 0.5 and float(voxel_size) <= 5:
                    ds_pcd = voxel_downsampling(point_cloud=pcd, voxel_size=float(voxel_size))
                    down_num_points = len(ds_pcd.points)
                    print(f'Done!\n\nOriginal number of points: {orig_num_points}\nNumber of points after downsampling: {down_num_points}\nVisualizing point cloud...')
                    o3d.visualization.draw_geometries([ds_pcd])
                    redo_flag = True
                    while redo_flag == True:
                        redo_input = input('Choose different voxel_size? [y/n]:')
                        if redo_input == 'n':
                            redo = False
                            redo_flag = False
                        elif redo_input == 'y':
                            redo = True 
                            redo_flag = False
                        else:
                            print('Invalid input. Try again')
                        flag = False
                else:   
                    again = input('Invalid parameter values. Try again? [y/n]:')
                    if again == 'y':
                        flag = False
                    else:
                        print('Exiting program')
                        sys.exit()
    elif type == 'u':
        while redo == True:
            k_points = input(f'Choose every_k_points (default = {def_k_points}, allowed range 2-500):')
            flag = True
            while flag == True:
                try: 
                    _ = float(k_points)
                except:
                    print('Value entered is not a number')
                    flag = False
                    continue
                if float(k_points) >= 2 and float(k_points) <= 500:
                    ds_pcd = uniform_downsampling(point_cloud=pcd, k_points=int(k_points))
                    down_num_points = len(ds_pcd.points)
                    print(f'Done!\n\nOriginal number of points: {orig_num_points}\nNumber of points after downsampling: {down_num_points}\nVisualizing point cloud...')
                    o3d.visualization.draw_geometries([ds_pcd])
                    redo_flag = True
                    while redo_flag == True:
                        redo_input = input('Choose different value for every_k_points? [y/n]:')
                        if redo_input == 'n':
                            redo = False
                            redo_flag = False
                        elif redo_input == 'y':
                            redo = True 
                            redo_flag = False
                        else:
                            print('Invalid input. Try again')
                        flag = False
                else:   
                    again = input('Invalid parameter values. Try again? [y/n]:')
                    if again == 'y':
                        flag = False
                    else:
                        print('Exiting program')
                        sys.exit()
    elif type == 'e':
        print('Exiting program')
        sys.exit()
    else:
        print('Invalid input. Try again')
        type = None
    return ds_pcd, voxel_size, k_points, type

pcd = o3d.io.read_point_cloud(point_cloud_path)
type = None
while type == None:
    type = input('Choose between voxel downsampling, uniform downsampling or exit code [v/u/e]:')
    ds_pcd, voxel_size, k_points, type = downsample_pcd(pcd, type)

save = None
while save == None:
    save= input('Save downsampled point cloud? [y/n]:')
    if save == 'y':
        if type == 'v':
            output_path = os.path.splitext(point_cloud_path)[0] +'_voxeldown' + voxel_size + '.ply'
            print(f'Saving in {output_path}')
            o3d.io.write_point_cloud(output_path, ds_pcd)
        elif type == 'u':
            output_path = os.path.splitext(point_cloud_path)[0] +'_unidown' + k_points + '.ply'
            print(f'Saving in {output_path}')
            o3d.io.write_point_cloud(output_path, ds_pcd)
        sys.exit()
    elif save == 'n':
        print('Exiting program')
        sys.exit()
    else:
        print('Invalid input. Try again')
        save = None