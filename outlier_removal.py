import open3d as o3d
import os 
import sys
import copy

point_cloud_path = '/home/chris/Code/PointClouds/data/point_cloud_files/VentilationGrate_2_unidown80.ply'

output_path = os.path.splitext(point_cloud_path)[0] + '_or.ply'
def_nb_neighbors = 80
def_std_ratio=2.0

def display_inlier_outlier(cloud,ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    outlier_cloud.paint_uniform_color([1,0,0])
    inlier_cloud.paint_uniform_color([0.8,0.8,0.8])
    print('Visualizing outliers in red and inliers in gray:')
    o3d.visualization.draw_geometries([inlier_cloud,outlier_cloud])

def different_params(point_cloud, nb_neighbors, std_ratio):
    choose_new = True
    redo = True
    while redo == True:
        diff_params = input('Choose different parameters? [y/n]:')
        if diff_params == 'y':
            flag = True
            while flag == True:
                nb_neighbors = input(f'Choose nb_neighbors (default = {def_nb_neighbors}, allowed range 1-2000): ')
                std_ratio = input(f'Choose std_ratio (default = {def_std_ratio}, allowed range 1-5):')
                try:
                    _ = int(nb_neighbors)
                    _ = float(std_ratio)
                except:
                    print('Value entered is not a number')
                    continue
                if int(nb_neighbors) >= 1 and int(nb_neighbors) <= 2000 and float(std_ratio) >= 1 and float(std_ratio) <= 5:
                    print(f'Performing statistical outlier removal with parameters nb_neighbors = {nb_neighbors} and std_ratio = {std_ratio}')
                    new_point_cloud, ind = point_cloud.remove_statistical_outlier(nb_neighbors=int(nb_neighbors), std_ratio=float(std_ratio))
                    display_inlier_outlier(point_cloud, ind)
                    flag = False
                    redo = False
                    return new_point_cloud, choose_new, nb_neighbors, std_ratio
                else:
                    again = input('Invalid parameter values. Try again? [y/n]:')
                    if again == 'y':
                        flag = True
                    else:
                        print('Exiting program')
                        sys.exit()
        elif diff_params == 'n':
            choose_new = False
            return point_cloud, choose_new, nb_neighbors, std_ratio
        else:
            print('Invalid input. Try again')

point_cloud = o3d.io.read_point_cloud(point_cloud_path)

print(f'Performing statistical outlier removal with parameters nb_neighbors = {def_nb_neighbors} and std_ratio = {def_std_ratio}')
new_point_cloud, ind = point_cloud.remove_statistical_outlier(nb_neighbors=def_nb_neighbors, std_ratio=def_std_ratio)

display_inlier_outlier(point_cloud, ind)

repeat = True
nb = def_nb_neighbors
std = def_std_ratio
while repeat == True:
    apply = input('Remove points? [y/n]:')

    if apply == 'y':
        print('Removing points and visualizing point cloud...')
        point_cloud = new_point_cloud
        vis_point_cloud = copy.deepcopy(point_cloud)
        vis_point_cloud.paint_uniform_color([0.8,0.8,0.8])
        o3d.visualization.draw_geometries([point_cloud])
        repeat_input = input('Repeat process with same parameters? [y/n]:')
        if repeat_input == 'y':
            print(f'Performing statistical outlier removal with parameters nb_neighbors = {nb} and std_ratio = {std}')
            new_point_cloud, ind = point_cloud.remove_statistical_outlier(nb_neighbors=int(nb), std_ratio=float(std))
            display_inlier_outlier(point_cloud, ind)
            repeat = True
        else:
            new_point_cloud, choose_new, nb, std = different_params(point_cloud=point_cloud, nb_neighbors=nb,std_ratio=float(std))
            if choose_new == True:
                repeat = True
            else: 
                repeat = False

    elif apply == 'n':
        new_point_cloud, choose_new, nb, std = different_params(point_cloud=point_cloud, nb_neighbors=nb,std_ratio=float(std))
        if choose_new == True:
            repeat = True
        else:
            repeat = False

save = input('Save point cloud? [y/n]:')

if save == 'y': 
    print(f'Saving new point cloud in {output_path}')
    o3d.io.write_point_cloud(output_path, new_point_cloud)
else:   
    print('Exiting program')
    sys.exit()

print('Done')