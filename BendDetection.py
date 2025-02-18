import numpy as np
import open3d as o3d
from functools import partial
import util
from BendLength import BendLengthCalculator

point_cloud_location = "/home/chris/Code/PointClouds/data/ply/CircularVentilationGrateExtraCleanedFull.ply"
pcd = o3d.io.read_point_cloud(point_cloud_location)

pcd = util.preProcessCloud(pcd)

expected_number_of_planes = 7
pt_to_plane_dist = 0.4

segment_models, segments, main_surface_idx = util.multiOrderRansac(pcd, expected_number_of_planes, pt_to_plane_dist)
angles_rad = util.findAnglesBetweenPlanes(segment_models, main_surface_idx)
intersection_lines = util.findIntersectionLines(segment_models, main_surface_idx)
anchor_points = util.findAnchorPoints(segment_models, segments, intersection_lines, main_surface_idx)

sample_dist = 0.3
aggregation_range = 15
eigen_threshold = 0.05
angle_threshold = 0.12
radius = 1.5
bend_length_calculator = BendLengthCalculator(pcd, anchor_points, intersection_lines, eigen_threshold, angle_threshold, aggregation_range, sample_dist, radius)
bend_edges = bend_length_calculator.compute_bend_lengths()

util.draw_bend_edges(pcd, bend_edges)

