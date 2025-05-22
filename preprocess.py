import open3d as o3d
import numpy as np

def estimate_point_density(pcd, radius_mm=1.0, min_neighbors=30, verbose=False):
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    counts = []
    total_checked = 0
    for i in range(0, len(pcd.points), max(1, len(pcd.points)//1000)):  # sample ~1000 points
        [_, idx, _] = kdtree.search_radius_vector_3d(pcd.points[i], radius_mm)
        neighbor_count = len(idx) - 1  # exclude the point itself
        if neighbor_count >= min_neighbors:
            counts.append(neighbor_count)
        total_checked += 1

    if counts:
        avg_density = np.mean(counts)
        if verbose:
            print(f"[INFO] Average density (excluding sparse points): {avg_density:.2f} points per {radius_mm}mm sphere")
            print(f"[INFO] Used {len(counts)} / {total_checked} sampled points (≥ {min_neighbors} neighbors)")
    else:
        avg_density = 0
        print(f"[WARN] No sampled points had ≥ {min_neighbors} neighbors.")

    return avg_density

def raycast_topdown(mesh, x_rotation, z_rotation, spacing=0.1):

    x_rotation = 110
    z_rotation = 90
    Rx = mesh.get_rotation_matrix_from_axis_angle([np.deg2rad(x_rotation), 0, 0])
    Rz = mesh.get_rotation_matrix_from_axis_angle([0, 0, np.deg2rad(z_rotation)])
    mesh.rotate(Rx, center=mesh.get_center())
    mesh.rotate(Rz, center=mesh.get_center())

    scene = o3d.t.geometry.RaycastingScene()
    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    _ = scene.add_triangles(t_mesh)
    aabb = mesh.get_axis_aligned_bounding_box()
    min_x, min_y, _ = aabb.min_bound
    max_x, max_y, _ = aabb.max_bound
    x_vals = np.linspace(min_x, max_x, int((max_x - min_x)/spacing))
    y_vals = np.linspace(min_y, max_y, int((max_y - min_y)/spacing))
    xx, yy = np.meshgrid(x_vals, y_vals)
    origin_z = aabb.max_bound[2] + 10
    origins = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, origin_z)], axis=1)
    directions = np.tile([0, 0, -1], (len(origins), 1))
    rays = o3d.core.Tensor(np.hstack((origins, directions)), dtype=o3d.core.Dtype.Float32)
    hits = scene.cast_rays(rays)
    mask = hits['t_hit'].isfinite()
    hit_points = origins[mask.numpy()] + hits['t_hit'][mask].numpy().reshape(-1, 1) * directions[mask.numpy()]
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(hit_points))

def preprocess_for_fpfh(pcd, voxel_size):
    """Downsample and estimate FPFH features for a point cloud."""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )
    return pcd_down, fpfh

def run_global_icp_alignment(scan_pcd, cad_pcd, voxel_size=1.0, verbose=False):
    """Run global RANSAC + ICP refinement."""
    scan_down, scan_fpfh = preprocess_for_fpfh(scan_pcd, voxel_size)
    cad_down, cad_fpfh = preprocess_for_fpfh(cad_pcd, voxel_size)

    # RANSAC-based global registration
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        cad_down, scan_down, cad_fpfh, scan_fpfh, mutual_filter=True,
        max_correspondence_distance=voxel_size * 1.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )

    if verbose:
        print("[INFO] Global RANSAC transformation:")
        print(result_ransac.transformation)

    scan_pcd.estimate_normals()
    cad_pcd.estimate_normals()

    result_icp = o3d.pipelines.registration.registration_icp(
        cad_pcd, scan_pcd, 2.0, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    if verbose:
        print("[INFO] ICP refinement transformation:")
        print(result_icp.transformation)
        print(f"[INFO] ICP Fitness: {result_icp.fitness:.4f}, RMSE: {result_icp.inlier_rmse:.4f}")

    return result_icp.transformation

def crop_scan_near_mesh(scan_pcd, cad_mesh, max_distance=2.5):

    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(cad_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(t_mesh)

    scan_pts = np.asarray(scan_pcd.points)
    pcd_tensor = o3d.core.Tensor(scan_pts, dtype=o3d.core.Dtype.Float32)

    signed_dists = scene.compute_signed_distance(pcd_tensor)
    abs_dists = np.abs(signed_dists.numpy())
    mask = abs_dists < max_distance

    cropped = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scan_pts[mask]))
    return cropped

def remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0, radius=1.5, min_points=10):
    pcd_clean, ind_stat = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd_clean, ind_rad = pcd_clean.remove_radius_outlier(nb_points=min_points, radius=radius)
    
    return pcd_clean

def dbscan_cleanup(pcd, eps=0.8, min_points=10):
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    largest_cluster = labels == np.bincount(labels[labels >= 0]).argmax()
    final = pcd.select_by_index(np.where(largest_cluster)[0])
    return final

def preProcessData(scan_pcd, cad_mesh, x_rotation, z_rotation, verbose=False):
    avg_density = estimate_point_density(scan_pcd, radius_mm=1.0, verbose=verbose)
    cad_pcd = raycast_topdown(cad_mesh, x_rotation=x_rotation, z_rotation=z_rotation, spacing=0.1)
    T_final = run_global_icp_alignment(scan_pcd, cad_pcd, voxel_size=1.0, verbose=verbose)
    cad_pcd.transform(T_final)
    cad_mesh.transform(T_final)
    cropped = crop_scan_near_mesh(scan_pcd, cad_mesh, max_distance=1.5)
    cleaned = remove_outliers(cropped, nb_neighbors=20, std_ratio=4.0, radius=2.5, min_points=8)
    preprocessed_pcd = dbscan_cleanup(cleaned, eps=0.8, min_points=10)
    nn_distance = np.mean([preprocessed_pcd.compute_nearest_neighbor_distance()])
    radius_normals = nn_distance*4
    preprocessed_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normals, max_nn=16), fast_normal_computation=True)
    return preprocessed_pcd, avg_density, cad_pcd