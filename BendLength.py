import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from open3d.t.geometry import TriangleMesh

class BendLengthCalculator:
    def __init__(self, pcd, anchor_points, intersection_lines, eigen_threshold, angle_threshold, aggregation_range, sample_dist, radius):

        self.pcd = pcd
        self.pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        self.anchor_points=anchor_points
        self.eigen_threshold=eigen_threshold
        self.angle_threshold=angle_threshold
        self.aggregation_range=aggregation_range
        self.sample_dist=sample_dist
        self.radius=radius
        self.intersection_lines=intersection_lines
    
    def calculate_eigen_norm_and_plane_direction(self, neighbor_coordinates):
        mu = np.mean(neighbor_coordinates, axis=0)
        norm = neighbor_coordinates - mu
        cov = np.cov(norm.T)
        eig_val, eig_vec = np.linalg.eig(cov)
        sorted_idx = np.argsort(eig_val)[::-1]
        eig_val = eig_val[sorted_idx]
        eig_vec = eig_vec[:, sorted_idx]
        eig_val_norm = eig_val.copy()

        for z in range(len(eig_val)):
            eig_val_norm[z] = np.exp(eig_val[z])/np.sum(np.exp(eig_val))

        plane_direction = np.cross(eig_vec[:, 0], eig_vec[:, 1])

        return eig_val_norm, plane_direction
    
    def compute_bend_lengths(self):
        bend_edges_dict = {}
        for i in range(len(self.intersection_lines)):
            #print(f'Computing bend length for intersection {i+1}')
            it = 0
            forward = True
            eigen_value_dict = {}
            plane_direction_dict = {}

            while True:
                if forward:
                    sampled_anchor_point = self.anchor_points[i+1] + (it*self.sample_dist)*self.intersection_lines[i+1][0]/np.linalg.norm(self.intersection_lines[i+1][0])
                else:
                    sampled_anchor_point = self.anchor_points[i+1] - (it*self.sample_dist)*self.intersection_lines[i+1][0]/np.linalg.norm(self.intersection_lines[i+1][0])

                [k, idx, _] = self.pcd_tree.search_radius_vector_3d(sampled_anchor_point, self.radius)        

                #Exit condition 1: Point cloud boundary reached
                if k < 5:
                    if forward == False:
                        #print(f'Backward bend ended at {sampled_anchor_point}')
                        bend_edges_dict[i+1] = (forward_edge, sampled_anchor_point)
                        break
                    else:
                        eigen_value_dict.clear()
                        plane_direction_dict.clear()
                        forward = False
                        it = 0
                        forward_edge = sampled_anchor_point
                        continue
                
                neighbor_coordinates = np.asarray(self.pcd.points)[idx[1:], :]
                eigen_value_norm, plane_direction = self.calculate_eigen_norm_and_plane_direction(neighbor_coordinates)

                eigen_value_dict[it] = eigen_value_norm

                #Flip plane direction if necessary
                if it == 0:
                    plane_direction_dict[it] = plane_direction
                else:
                    previous_plane_direction = plane_direction_dict[it-1]
                    if np.dot(plane_direction,previous_plane_direction)<0:
                        plane_direction = -plane_direction
                    plane_direction_dict[it] = plane_direction

                eigen_agg = 0
                angle_agg = 0

                if it > self.aggregation_range:
                    #Calculate aggragate for the eigenvalues
                    for eigen_index in range(len(eigen_value_norm)):
                        for prev_iteration in range(self.aggregation_range):
                            eigen_diff = eigen_value_norm[eigen_index] - eigen_value_dict[it-(prev_iteration+1)][eigen_index]
                            eigen_agg += (eigen_diff/self.sample_dist)**2

                    #Calculate aggragate for the planes of the ellipsoids
                    for prev_iteration in range(self.aggregation_range):
                        dot_product = np.dot(plane_direction, plane_direction_dict[it-(prev_iteration+1)])
                        norm_1 = np.linalg.norm(plane_direction)
                        norm_2 = np.linalg.norm(plane_direction_dict[it-(prev_iteration+1)])
                        plane_angle = np.arccos(np.clip(dot_product/(norm_1*norm_2), -1, 1))
                        angle_agg += plane_angle

                    eigen_agg /= self.aggregation_range
                    angle_agg /= self.aggregation_range

                    # Exit condition 2: Bend ended/PCA structure changed
                    if eigen_agg > self.eigen_threshold or angle_agg > self.angle_threshold:
                        if forward == False:
                            #print(f'Backward bend ended at {sampled_anchor_point}')
                            bend_edges_dict[i+1] = (forward_edge, sampled_anchor_point)
                            break
                        else:
                            eigen_value_dict.clear()
                            plane_direction_dict.clear()
                            forward = False
                            it = 0
                            forward_edge = sampled_anchor_point
                            continue
                    
                it += 1

        #print(bend_edges_dict)
        return bend_edges_dict