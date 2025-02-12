import numpy as np
import open3d as o3d

class ConcaveHullClusterCallback:
    def __init__(self, cluster, k):
        cluster = np.unique(cluster, axis=0)
        self.cluster_points_3d = cluster
        self.cluster_pcd = o3d.geometry.PointCloud()
        self.cluster_pcd.points = o3d.utility.Vector3dVector(self.cluster_points_3d)
        self.cluster_colors = np.tile((0.1,0.1,0.1), (self.cluster_points_3d.shape[0], 1))
        self.cluster = np.delete(cluster, 2, 1)
        self.number_of_points = self.cluster.shape[0]
        self.section_cloud = False
        self.grid_size = 4
        if self.number_of_points > 10000:
            self.section_cloud = True
            self.create_sections()
        self.direction_vector = np.array([-1, 0])
        self.deactivated_point_indices = []
        self.deactivated_points = []
        self.grid_points = np.ones(self.cluster.shape[0], dtype=bool)
        self.k = k
        self.k_list = [self.k, self.k+2, self.k+4]
        self.hull_point_indices_global = []
        self.hull_point_indices_local = []
        self.it = 0
        self.breakloop_it = 10000
        self.hullclosed = False
        self.key_pressed = False
        self.collinearity = False
        self.current_section_index = None
        self.current_section_points = None
        self.current_section_points_global_indices = None

    def remove_duplicate_points(self, points):
        return np.unique(points, axis=0)
    
    def create_sections(self):
        print('Creating Sections', flush=True)
        x_min, y_min = self.cluster.min(axis=0)
        x_max, y_max = self.cluster.max(axis=0)
        
        print(f'x_max={x_max}, x_min={x_min}, y_max={y_max}, y_min={y_min}')
        print(f'x_diff = {(x_max - x_min)}, y_diff = {(y_max - y_min)}')
        M = round((x_max - x_min) / (3.5 * self.grid_size))
        N = round((y_max - y_min) / (3.5 * self.grid_size))
        print(f'M={M} and N={N}')
        if N == 0: 
            y_intervals = np.linspace(y_min, y_max, 2)
        else: 
            y_intervals = np.linspace(y_min, y_max, 2 * N + 1)

        if M == 0: 
            x_intervals = np.linspace(x_min, x_max, 2)
        else:
            x_intervals = np.linspace(x_min, x_max, 2 * M + 1)

        small_sections = []
        small_section_centers = []
        self.small_section_bboxes = []
        small_section_indices = []

        for i in range(max(2 * N, len(y_intervals) - 1)):
            for j in range(max(2 * M, len(x_intervals) - 1)):
                x_start, x_end = x_intervals[j], x_intervals[j + 1]
                y_start, y_end = y_intervals[i], y_intervals[i + 1]

                x_center = (x_start + x_end) / 2.0
                y_center = (y_start + y_end) / 2.0
                small_section_centers.append([x_center, y_center])
                
                if i == max(2 * N, len(y_intervals) - 1) - 1:
                    y_condition = (self.cluster[:, 1] >= y_start) & (self.cluster[:, 1] <= y_end)
                else:
                    y_condition = (self.cluster[:, 1] >= y_start) & (self.cluster[:, 1] < y_end)
                    
                if j == max(2 * M, len(x_intervals) - 1) - 1:
                    x_condition = (self.cluster[:, 0] >= x_start) & (self.cluster[:, 0] <= x_end)
                else:
                    x_condition = (self.cluster[:, 0] >= x_start) & (self.cluster[:, 0] < x_end)

                section_indices = np.where(x_condition & y_condition)[0]

                small_sections.append(self.cluster[section_indices])
                small_section_indices.append(section_indices)
            
                section_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=[x_start, y_start, -0.01],max_bound=[x_end,y_end,0.01])
                section_bbox.color = [0.42,0.53,0.39]
                self.small_section_bboxes.append(section_bbox)
        
        print(f'Created {len(small_sections)} small sections.')

        self.section_centers = []
        self.section_indices_arrays = []
        self.section_points_arrays = []

        if N >= 1 and M >= 1:
            for i in range(min(2 * N - 1, len(y_intervals) - 2)):
                for j in range(min(2 * M - 1, len(x_intervals) - 2)):
                    x_start = x_intervals[j]
                    x_end = x_intervals[j + 2]
                    y_start = y_intervals[i]
                    y_end = y_intervals[i + 2]

                    x_center = (x_start + x_end) / 2.0
                    y_center = (y_start + y_end) / 2.0
                    self.section_centers.append([x_center, y_center])
                    
                    #print(f'Section {i*(2*M-1)+j+1} contains small sections {j+2*M*i}, {j+1+2*M*i}, {j+2*M*(i+1)} and {j+1+2*M*(i+1)}')

                    section_indices = np.concatenate([
                        small_section_indices[j + 2 * M * i],
                        small_section_indices[j + 1 + 2 * M * i],
                        small_section_indices[j + 2 * M * (i + 1)],
                        small_section_indices[j + 1 + 2 * M * (i + 1)]
                    ])
                    
                    self.section_indices_arrays.append(section_indices)
                    self.section_points_arrays.append(self.cluster[section_indices])

                    #print(f'Created section {len(self.section_centers)} with indices: {section_indices}')
                    #print(f'Section points set size: {len(self.section_points_sets[-1])}')

        elif N==0 and M>=1:
            for j in range(2*M-1):
                x_start = x_intervals[j]
                x_end = x_intervals[j+2]
                y_start = y_intervals[0]
                y_end = y_intervals[1]

                x_center = (x_start + x_end) / 2.0
                y_center = (y_start + y_end) / 2.0
                self.section_centers.append([x_center, y_center])

                section_indices = np.concatenate([
                    small_section_indices[j],
                    small_section_indices[j+1],
                ])

                self.section_indices_arrays.append(section_indices)
                self.section_points_arrays.append(self.cluster[section_indices])

                #print(f'Created section {len(self.section_centers)} with indices: {section_indices}')
                #print(f'Section points set size: {len(self.section_points_sets[-1])}')


        elif N>=1 and M==0:
            for i in range(2*N-1):
                x_start = x_intervals[0]
                x_end = x_intervals[1]
                y_start = y_intervals[i]
                y_end = y_intervals[i+2]

                x_center = (x_start + x_end) / 2.0
                y_center = (y_start + y_end) / 2.0
                self.section_centers.append([x_center, y_center])

                section_indices = np.concatenate([
                    small_section_indices[i],
                    small_section_indices[i+1],
                ])

                self.section_indices_arrays.append(section_indices)
                self.section_points_arrays.append(self.cluster[section_indices])

                #print(f'Created section {len(self.section_centers)} with indices: {section_indices}')
                #print(f'Section points set size: {len(self.section_points_sets[-1])}')

        elif N==0 and M==0:
            self.section_cloud = False

        print(f'Total sections created: {len(self.section_centers)}', flush=True)

    def calculate_distances(self, hull_point, active_indices):
        if self.section_cloud:
            return np.sqrt(np.sum(np.square(self.current_section_points[active_indices] - hull_point), axis=1))
        else:
            return np.sqrt(np.sum(np.square(self.cluster[active_indices] - hull_point), axis=1))
    

    def find_closest_section(self, current_hull_point):
        distances = np.sqrt(np.sum(np.square(self.section_centers - current_hull_point), axis=1))
        return np.argsort(distances)[0]
    
    def get_knn(self, current_hull_point):
        if self.section_cloud:
            deactivated_mask = np.isin(self.current_section_points_global_indices, self.deactivated_point_indices)
            active_grid_points = np.logical_and(~deactivated_mask, self.grid_points)
            active_grid_points_indices_local = np.nonzero(active_grid_points)[0]

            distances = self.calculate_distances(current_hull_point, active_grid_points_indices_local)
            sorted_indices = np.argsort(distances)

            neighbor_indices = active_grid_points_indices_local[sorted_indices[:self.k]]
            neighbor_points = self.current_section_points[neighbor_indices]
            return neighbor_indices, neighbor_points
        else:
            inactive_indices = np.zeros(self.grid_points.shape, dtype=bool)
            inactive_indices[self.deactivated_point_indices] = True
            active_grid_points = np.logical_and(~inactive_indices, self.grid_points)
            active_grid_points_indices = np.nonzero(active_grid_points)[0]
            distances = self.calculate_distances(current_hull_point, active_grid_points_indices)
            sorted_indices = np.argsort(distances)
            neighbor_indices = active_grid_points_indices[sorted_indices[:self.k]]
            neighbor_points = self.cluster[neighbor_indices]
            #print(f'Neighbor indices: {neighbor_indices}', flush=True)
            return neighbor_indices, neighbor_points

    def get_points_inside_grid(self, current_hull_point):
        grid_size_2d = np.array([self.grid_size, self.grid_size])
        grid_min = current_hull_point - grid_size_2d / 2
        grid_max = current_hull_point + grid_size_2d / 2
        if self.section_cloud:
            #print(f'Current Section Points: {self.current_section_points}', flush=True)
            self.grid_points = np.all((self.current_section_points >= grid_min) & 
                                      (self.current_section_points <= grid_max), axis=1)
        else:
            self.grid_points = np.all((self.cluster >= grid_min) & (self.cluster <= grid_max), axis=1)

    def paint_pcd(self, vis, neighborhood_indices, current_hull_point_global_index, chosen_neighbor_index):
        colors = self.cluster_colors.copy()

        if self.section_cloud:
            colors[self.current_section_points_global_indices] = (0.42, 0.53, 0.39)
            deactivated_mask = np.isin(self.current_section_points_global_indices, self.deactivated_point_indices)
            active_grid_points = np.logical_and(~deactivated_mask, self.grid_points)
            active_grid_points_indices_local = np.nonzero(active_grid_points)[0]
            global_active_grid_points_indices = self.current_section_points_global_indices[active_grid_points_indices_local]
            #print(f'Global active grid points indices: {global_active_grid_points_indices}', flush=True)
            # Color the active grid points in the section
        else:
            inactive_indices = np.zeros(self.grid_points.shape, dtype=bool)
            inactive_indices[self.deactivated_point_indices] = True
            active_grid_points = np.logical_and(~inactive_indices, self.grid_points)
            global_active_grid_points_indices = np.nonzero(active_grid_points)[0]
            #print(f'Global active grid points indices: {global_active_grid_points_indices}', flush=True)

        # Color the active grid points
        colors[global_active_grid_points_indices] = (0.3, 0.41, 0.57)

        # Color other points
        past_hull_points_indices = np.asarray(self.hull_point_indices_global)
        colors[neighborhood_indices] = (0.9, 0.1, 0.1)
        colors[past_hull_points_indices] = (1, 0.6, 0.1)
        colors[current_hull_point_global_index] = (0.1, 0.1, 0.9)
        colors[chosen_neighbor_index] = (0.1, 0.6, 0.1)
        
        self.cluster_pcd.colors = o3d.utility.Vector3dVector(colors)
        vis.update_geometry(self.cluster_pcd)
        
        if hasattr(self, 'bbox'):
            vis.remove_geometry(self.bbox, reset_bounding_box=False)
        current_hull_point = np.append(self.cluster[current_hull_point_global_index], 0)
        self.bbox = self.create_grid(current_hull_point)
        vis.add_geometry(self.bbox, reset_bounding_box=False)
        
        if hasattr(self, 'bbox'):
            vis.remove_geometry(self.bbox, reset_bounding_box=False)
        current_hull_point = np.append(self.cluster[current_hull_point_global_index], 0)
        self.bbox = self.create_grid(current_hull_point)
        vis.add_geometry(self.bbox, reset_bounding_box=False)

    def create_grid(self, current_hull_point_3d):
        grid_size_3d = np.array([self.grid_size, self.grid_size, 0.01])
        min_bound = current_hull_point_3d - grid_size_3d/2
        max_bound = current_hull_point_3d + grid_size_3d/2
        grid = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
        grid.color = (1,0,0)
        return grid

    def find_lowest_point(self):
        lowest_y = np.amin(self.cluster[:,1])
        lowest_y_indices = np.where(self.cluster[:, 1] == lowest_y)[0]

        if lowest_y_indices.shape[0] == 1:
            lowest_point_index = lowest_y_indices[0]
            lowest_point = self.cluster[lowest_point_index]
        elif lowest_y_indices.shape[0] > 1:
            lowest_y_points = self.cluster[lowest_y_indices]
            lowest_x = np.amin(lowest_y_points[:, 0])
            lowest_point_index = np.where((self.cluster[:,0] == lowest_x) & (self.cluster[:,1] == lowest_y))[0][0]
            lowest_point = np.array([lowest_x, lowest_y])
        return lowest_point, lowest_point_index
    
    def unit_vector(self, vector):
        return vector/np.linalg.norm(vector)
    
    def ccw(self,A,B,C):
        return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

    def check_parallelism(self, vector_alpha, vector_beta):
        determinant = vector_alpha[0] * vector_beta[1] - vector_alpha[1] * vector_beta[0]
        return np.isclose(determinant, 0)
    
    def check_collinearity(self, A, B, C):
        area = A[0]*(B[1]-C[1]) + B[0]*(C[1]-A[1]) + C[0]*(A[1]-B[1])
        if area==0:
            return True
        return False

    def intersect_with_collinearity(self, A, B, C, D):
        if np.array_equal(A, C) or np.array_equal(A, D) or np.array_equal(B, C) or np.array_equal(B, D):
            return False
        if self.check_collinearity(A, C, D) or self.check_collinearity(B, C, D):
            return False
        return self.ccw(A, C, D) != self.ccw(B, C, D) and self.ccw(A, B, C) != self.ccw(A, B, D)
    
    def intersect_without_collinearity(self, A, B, C, D):
        if np.array_equal(A, C) or np.array_equal(A, D) or np.array_equal(B, C) or np.array_equal(B, D):
            return False
        return self.ccw(A, C, D) != self.ccw(B, C, D) and self.ccw(A, B, C) != self.ccw(A, B, D)

    def find_all_indices(self, target_list, query_value):
        return [i for i, x in enumerate(target_list) if x == query_value]
    
    def are_consecutive(self, index1, index2):
        return abs(index1 - index2) == 1
    
    def choose_neighbor(self, current_hull_point, neighbors):
        vectors = neighbors - current_hull_point
        dot_products = np.dot(vectors, self.direction_vector)
        cross_products = np.cross(self.direction_vector, vectors)
        norms = np.linalg.norm(vectors, axis=1)
        cos_angles = np.clip(dot_products / norms, -1.0, 1.0)
        angles = np.arccos(cos_angles)

        right_turn_indices = np.where(cross_products <= 0)[0]
        left_turn_indices = np.where(cross_products > 0)[0]

        if self.it > 2:
            # Separate the 180-degree turns
            threshold = np.deg2rad(175)  # Set a threshold for near 180-degree turns
            right_turn_angles = angles[right_turn_indices]
            left_turn_angles = angles[left_turn_indices]

            print(f'Right turn angles: {right_turn_angles}')
            print(f'Left turn angles: {left_turn_angles}')

            right_turn_180_indices = right_turn_indices[right_turn_angles >= threshold]
            right_turn_non_180_indices = right_turn_indices[right_turn_angles < threshold]
            left_turn_180_indices = left_turn_indices[left_turn_angles >= threshold]
            left_turn_non_180_indices = left_turn_indices[left_turn_angles < threshold]

            # Sort non-180-degree turns
            sorted_right_turn_non_180_indices = right_turn_non_180_indices[np.argsort(-right_turn_angles[right_turn_angles < threshold])]
            sorted_left_turn_non_180_indices = left_turn_non_180_indices[np.argsort(left_turn_angles[left_turn_angles < threshold])]

            # Combine all indices
            combined_indices = np.concatenate([sorted_right_turn_non_180_indices, sorted_left_turn_non_180_indices, right_turn_180_indices, left_turn_180_indices])

        else:
            right_turn_angles = angles[right_turn_indices]
            left_turn_angles = angles[left_turn_indices]
            sorted_right_turn_indices = right_turn_indices[np.argsort(-right_turn_angles)]
            sorted_left_turn_indices = left_turn_indices[np.argsort(left_turn_angles)]
            combined_indices = np.concatenate([sorted_right_turn_indices, sorted_left_turn_indices])
        
        if self.section_cloud:
            grid_hull_points_indices = [idx for idx in self.hull_point_indices_local if self.grid_points[idx]]
        else:
            grid_hull_points_indices = [idx for idx in self.hull_point_indices_global if self.grid_points[idx]]

        for index in combined_indices:
            chosen_neighbor = neighbors[index]
            C = current_hull_point
            D = chosen_neighbor
            intersects = False

            for i in range(1, len(grid_hull_points_indices)):
                A_index = grid_hull_points_indices[i - 1]
                B_index = grid_hull_points_indices[i]
                
                if self.section_cloud:
                    A_indices = self.find_all_indices(self.hull_point_indices_local, A_index)
                    B_indices = self.find_all_indices(self.hull_point_indices_local, B_index)
                else:
                    A_indices = self.find_all_indices(self.hull_point_indices_global, A_index)
                    B_indices = self.find_all_indices(self.hull_point_indices_global, B_index)

                is_consecutive = any(self.are_consecutive(A_idx, B_idx) for A_idx in A_indices for B_idx in B_indices)

                if is_consecutive:
                    if self.section_cloud:
                        A = self.current_section_points[A_index]
                        B = self.current_section_points[B_index]
                    else:
                        A = self.cluster[A_index]
                        B = self.cluster[B_index]

                    vector_alpha = B - A
                    vector_beta = D - C
                    if self.collinearity == False:
                        if self.intersect_without_collinearity(A, B, C, D) and not self.check_parallelism(vector_alpha, vector_beta):
                            intersects = True
                            break
                    else:
                        if self.intersect_with_collinearity(A, B, C, D) and not self.check_parallelism(vector_alpha, vector_beta):
                            intersects = True
                            break

            if not intersects:
                new_direction_vector = self.unit_vector(chosen_neighbor - current_hull_point)
                self.direction_vector = new_direction_vector
                return chosen_neighbor

        return None

    def increase_k(self, current_hull_point, neighbors):
        for z in range(1, len(self.k_list)):
            self.k = self.k_list[z]
            print(f'Increasing k to {self.k}', flush=True)
            _, neighbors = self.get_knn(current_hull_point)
            chosen_neighbor = self.choose_neighbor(current_hull_point, neighbors)
            if chosen_neighbor is None:
                continue
            else:
                return chosen_neighbor
            
    def enable_collinearity(self, current_hull_point, neighbors):
        print('Enabling collinearity')
        self.collinearity = True
        for z in range(len(self.k_list)):
            self.k = self.k_list[z]
            print(f'Trying with k: {self.k}', flush=True)
            _, neighbors = self.get_knn(current_hull_point)
            chosen_neighbor = self.choose_neighbor(current_hull_point, neighbors)
            if chosen_neighbor is None:
                continue
            else:
                self.collinearity = False
                return chosen_neighbor
        self.collinearity = False

    def concave_hull_iteration(self, vis):
        if self.key_pressed:
            return
        self.key_pressed = True
        self.k = self.k_list[0]
        if self.it == 0:
            self.lowest_point, self.lowest_point_index = self.find_lowest_point()
            #print(f'Lowest point = {self.lowest_point}', flush=True)
            #print(f'Deactivating point with index: {self.lowest_point_index}', flush=True)

            self.hull_point_indices_global.append(self.lowest_point_index)

            initial_colors = self.cluster_colors.copy()
            initial_colors[self.lowest_point_index] = (0.1,0.5,0.1)
            self.cluster_pcd.colors = o3d.utility.Vector3dVector(initial_colors)
            vis.add_geometry(self.cluster_pcd)
            if self.section_cloud:
                for section_bbox in self.small_section_bboxes:
                    vis.add_geometry(section_bbox)
                self.current_section_index = self.find_closest_section(self.lowest_point)
                #print(f'Closest section to lowest point is Section {self.current_section_index + 1}', flush=True)
                self.current_section_points_global_indices = self.section_indices_arrays[self.current_section_index]
                self.current_section_points = self.section_points_arrays[self.current_section_index]
                self.hull_point_indices_local = [np.where(self.current_section_points_global_indices == idx)[0][0]
                                                  for idx in self.hull_point_indices_global if idx in 
                                                  self.current_section_points_global_indices]

            self.it += 1
            self.key_pressed = False
        else:
            for j in range(1):
                #print(f'Iteration: {self.it}', flush=True, end='\r')
                #print(25*'-')
                if self.section_cloud and self.hull_point_indices_local:
                    #print(f'number of local hull_points = {len(self.hull_point_indices_local)}', flush=True)
                    #print(f'number of global hull_points = {len(self.hull_point_indices_global)}', flush=True)
                    current_hull_point_global_index = self.hull_point_indices_global[-1]
                    current_hull_point_local_index = self.hull_point_indices_local[-1]
                    current_hull_point = self.current_section_points[current_hull_point_local_index]
                else:
                    current_hull_point_global_index = self.hull_point_indices_global[-1]
                    current_hull_point = self.cluster[current_hull_point_global_index]

                self.deactivated_point_indices.append(current_hull_point_global_index)
                self.deactivated_points.append(current_hull_point)
                #print(f'Deactivating point with index: {current_hull_point_global_index}', flush=True)
                if self.section_cloud:
                    if self.it%4==0:
                        new_section_index = self.find_closest_section(current_hull_point)
                        if new_section_index != self.current_section_index:
                            #print('Changing sections!', flush=True)
                            self.current_section_index = new_section_index
                            self.current_section_points_global_indices = self.section_indices_arrays[self.current_section_index]
                            self.current_section_points = self.section_points_arrays[self.current_section_index]
                            self.hull_point_indices_local = [np.where(self.current_section_points_global_indices == idx)[0][0]
                                                    for idx in self.hull_point_indices_global if idx in self.current_section_points_global_indices]
                            #print(f'found this number of hull points in the new section: {len(self.hull_point_indices_local)}', flush=True)

                #print(f'Current hull point index: {current_hull_point_global_index}', flush=True)
                #print(f'All hull point indices so far: {self.hull_point_indices_global}', flush=True)
                #print(f'Direction vector: {self.direction_vector}', flush=True)

                if self.it == self.breakloop_it:
                    #print('Time to stop this madness', flush=True)
                    return
                
                self.get_points_inside_grid(current_hull_point)
                #print(f'grid points indices: {np.nonzero(self.grid_points)[0]}', flush=True)
                #print(f'Grid points mask size: {self.grid_points.shape}', flush=True)
                #print(f'Number of grid points: {np.sum(self.grid_points)}', flush=True)

                #local_active_grid_points_indices = np.nonzero(self.grid_points)[0]
                #print(f'Local active grid points indices: {local_active_grid_points_indices}', flush=True)
                #neighborhood_indices, neighbors = self.get_knn(current_hull_point)
                if self.section_cloud:
                    neighborhood_indices_local, neighbors = self.get_knn(current_hull_point)
                    neighborhood_indices_global = self.current_section_points_global_indices[neighborhood_indices_local]
                else:
                    neighborhood_indices_global, neighbors = self.get_knn(current_hull_point)

                #print('current_hull_point:', current_hull_point, flush=True)
                #print('neighbor indices:', neighborhood_indices_global, flush=True)
                #print('neighbors:', neighbors, flush=True)    
                chosen_neighbor = self.choose_neighbor(current_hull_point, neighbors)
                if chosen_neighbor is None:
                    chosen_neighbor = self.increase_k(current_hull_point, neighbors)
                if chosen_neighbor is None:
                    chosen_neighbor = self.enable_collinearity(current_hull_point, neighbors)
                if chosen_neighbor is None:
                    print('Couldnt find neighbor or find lowest point, closing loop.', flush=True)
                    return
                #print(f'Current hull point: {current_hull_point}', flush=True)
                #print(f'Chosen neighbor: {chosen_neighbor}', flush=True)
                if self.section_cloud:
                    chosen_neighbor_index_local = np.where((self.current_section_points[:, 0] == chosen_neighbor[0]) & (self.current_section_points[:, 1] == chosen_neighbor[1]))[0][0]
                    chosen_neighbor_index_global = self.current_section_points_global_indices[chosen_neighbor_index_local]
                    self.hull_point_indices_local.append(chosen_neighbor_index_local)
                else:
                    chosen_neighbor_index_global = np.where((self.cluster[:, 0] == chosen_neighbor[0]) & (self.cluster[:, 1] == chosen_neighbor[1]))[0][0]

                self.hull_point_indices_global.append(chosen_neighbor_index_global)

                #print(f'Chosen neighbor index: {chosen_neighbor_index_global}',flush=True)
                if chosen_neighbor_index_global == self.lowest_point_index:
                    print('\nSuccess!',flush=True)
                    return

                if self.it >= int(2*self.k):
                    del self.deactivated_point_indices[0]
                self.it +=1
            
            self.paint_pcd(vis, neighborhood_indices_global, current_hull_point_global_index, chosen_neighbor_index_global)
            self.key_pressed = False