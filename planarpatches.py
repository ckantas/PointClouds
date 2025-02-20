import numpy as np
import open3d as o3d
from functools import partial
import util
from BendLength import BendLengthCalculator

# point_cloud_location = "/home/chris/Code/PointClouds/data/ply/CircularVentilationGrateExtraCleanedFull.ply"
point_cloud_location = "/home/chris/Code/PointClouds/data/FLIPscans/MortenPlateTopExtraCleanedFull.ply"
pcd = o3d.io.read_point_cloud(point_cloud_location)

pcd = util.preProcessCloud(pcd)

# using all defaults
oboxes = pcd.detect_planar_patches(
    normal_variance_threshold_deg=20,
    coplanarity_deg=75,
    outlier_ratio=0.2,
    min_plane_edge_length=0,
    min_num_points=0,
    search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

geometries = []
for obox in oboxes:
    mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obox, scale=[1, 1, 0.0001])
    mesh.paint_uniform_color(obox.color)
    geometries.append(mesh)
    geometries.append(obox)
#geometries.append(pcd)

o3d.visualization.draw_geometries(geometries)