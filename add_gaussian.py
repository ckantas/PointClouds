import open3d as o3d
import numpy as np
import os 
import sys 
import copy 

def add_gaussian_noise(points, mean=0, std_dev=0.02):
    noise = np.random.normal(mean, std_dev, points.shape)
    noisy_points = points + noise
    return noisy_points

if __name__ == "__main__":
    point_cloud_path= '/home/chris/Code/PointClouds/data/noisetest/PLY/Rectangle0.01cm.ply'
    point_cloud = o3d.io.read_point_cloud(point_cloud_path)

    # Extract points as NumPy array
    points = np.asarray(point_cloud.points)
    std = 0
    redo = True
    while redo == True:
        std = input('Choose standard deviation for noise (default: 0.02), allowed range 0.01-1.5: ')
        try:
            _ = float(std)
        except:
            print('Value entered is not a number')
            continue
        if float(std) >= 0.01 and float(std) <= 1.5:
            noisy_points = add_gaussian_noise(points, mean=0, std_dev=float(std))
            temp_point_cloud = copy.deepcopy(point_cloud)
            temp_point_cloud.points = o3d.utility.Vector3dVector(noisy_points)
            print(f'Adding noise with standard deviation {std} and drawing point cloud...')
            o3d.visualization.draw_geometries([temp_point_cloud])
            diff_params = input('Choose different parameters? [y/n]:')
            if diff_params == 'y':
                redo = True
                continue
            else:
                point_cloud.points = o3d.utility.Vector3dVector(noisy_points)
                redo = False

    save = input('Save point cloud? [y/n]:')

    if save == 'y':
        # Save the modified point cloud to a new PLY file
        output_path = os.path.splitext(point_cloud_path)[0] + '_noise' + std + '.ply'
        o3d.io.write_point_cloud(output_path, point_cloud)

        print(f'Noisy point cloud saved to {output_path}')
    else:
        print('Exiting program')
        sys.exit()