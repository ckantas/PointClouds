import open3d as o3d
import numpy as np

def add_gaussian_noise(points, mean=0, std_dev=0.01):
    noise = np.random.normal(mean, std_dev, points.shape)
    noisy_points = points + noise
    return noisy_points

def main():
    # Load PLY file
    input_file = "input_cloud.ply"
    point_cloud = o3d.io.read_point_cloud(input_file)

    # Extract points as NumPy array
    points = np.asarray(point_cloud.points)

    # Add Gaussian noise to point locations
    noisy_points = add_gaussian_noise(points)

    # Update point cloud with noisy points
    point_cloud.points = o3d.utility.Vector3dVector(noisy_points)

    # Save the modified point cloud to a new PLY file
    output_file = "noisy_cloud.ply"
    o3d.io.write_point_cloud(output_file, point_cloud)

    print(f"Noisy point cloud saved to {output_file}")

if __name__ == "__main__":
    main()