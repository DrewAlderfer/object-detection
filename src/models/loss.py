from typing import List, Tuple, Union
import numpy as np
from numpy.typing import ArrayLike, NDArray
import tensorflow as tf
from tensorflow.keras.losses import Loss

class BoundingBox_Processor:
    def __init__(self, 
                 # label_vector:NDArray,
                 # pred_vector:NDArray,
                 debug:bool=False):

        self.debug = debug

        # self.label_box = label_vector
        # self.pred_box = pred_vector

        # self.intersection = self._intersection_points()
        # self.inter_center = np.mean(self.intersection, axis=0)

    def d_print(self, message, **kwargs):
        if self.debug:
            print(message, **kwargs)
    
    def construct_intersection(self, box1, box2, return_centered:bool=True):

        intersections, box1_edges, box2_edges = self.rolling_intersection(box1, box2)
        inner_points = tf.concat([self.find_inner_points(box1, box2_edges), self.find_inner_points(box2, box1_edges)], axis=-2)
        intersection_points = tf.concat([intersections, inner_points], axis=-2)
        
        non_zero = tf.cast(intersection_points > .1, tf.float32)
        mask = tf.cast(intersection_points > .1, dtype=tf.bool)

        denomenator = tf.reduce_sum(non_zero, axis=-2)
        center = tf.reduce_sum(intersection_points, axis=-2) / denomenator
        center_adj_points = tf.where(mask, 
                                     intersection_points - tf.expand_dims(center, axis=-2),
                                     non_zero)

        
        point_angle = tf.where(mask, tf.math.atan2(center_adj_points[..., :, 1:],
                                                   center_adj_points[..., :, 0:1]) + np.pi,
                               non_zero)

        point_order = tf.argsort(tf.transpose(point_angle, perm=[0, 1, 2, 4, 3]), direction="DESCENDING", axis=-1)

        points_T = None
        if return_centered:
            points_T = tf.transpose(center_adj_points, [0, 1, 2, 4, 3])
        if not return_centered:
            points_T = tf.transpose(intersection_points, [0, 1, 2, 4, 3])

        sorted_points = tf.transpose(tf.gather(points_T, point_order, batch_dims=-1), perm=[0, 1, 2, 4, 3])
        result = tf.concat([sorted_points[..., :8, :], tf.expand_dims(center, axis=-2)], axis=-2)

        return result

    def calculate_iou(self, box1, box2):
        
        intersection_points = self.construct_intersection(box1, box2, return_centered=True)
        points = intersection_points[..., :8, :]
        center = intersection_points[..., 8:, :]
        area = self.traingle_area(points)
        
    def traingle_area(self, points):

        ones = tf.ones(points.shape[:-1] + (1,), dtype=tf.float32)
        triangle_matrix = tf.concat([points, ones], axis=-1)
        print(triangle_matrix[0, 5, 8])
        mask = tf.cast(tf.abs(tf.reduce_sum(triangle_matrix, axis=-1)) > 1, dtype=tf.float32)
        triangle_matrix = tf.reshape(triangle_matrix, triangle_matrix.shape[:-2] + (4, 2, 3))
        print(triangle_matrix[0,5,8])
        print(f"triangle_matrix: {triangle_matrix.shape}")
        # for _ in range(8):
        #     # center = np.full(triangle_matrix[..., :2, :].shape[:-2] + (1, 3), [0, 0, 1], dtype=np.float32)
        #     # triangle = tf.concat([triangle_matrix[...,0:2, :], center], axis=-2)
        #     print(triangle_matrix[0, 5, 8, :2])
        #     # det = tf.abs(tf.linalg.det(triangle) * .5)
        #     # print(f"triangle: {triangle.shape}")
        #     # print(triangle[0, 5, 8].numpy())
        #     # print(f"\ndeterminant {det.shape}")
        #     # print(f"triangle area = {det[0, 5, 8].numpy()}")
        #     triangle_matrix = tf.roll(triangle_matrix, shift=-1, axis=-2)

        # get the points subract the center from them
        # something to consider is not returning the actual points. If you are just using them to
        # get the area and the shortest path to that is just taking the triangle areas to from (0, 0)

    def intersection_area(self):
        P = self.intersection
        C = self.inter_center
        result = 0
        for i in range(len(self.intersection)):
            j = i - 1
            triangle = np.array([[P[i - 1][0], P[i - 1][1], 1],
                                 [P[i][0],     P[i][1],     1],
                                 [C[0],        C[1],        1]], dtype=np.float32)
            result += np.abs(np.linalg.det(triangle) * .5)

        return result

    def rolling_intersection(self, box1, box2):
        box1_edges = self.get_edges(box1)
        box2_edges = self.get_edges(box2) 
        edge_intersections = None
        for i in range(4):
            if i != 0:
                edge_intersections = tf.concat([edge_intersections,
                                                self.get_intersections(box1_edges, box2_edges)], 
                                                axis=-2)
                box1_edges = tf.roll(box1_edges, shift=-1, axis=-3)
                continue
            edge_intersections = self.get_intersections(box1_edges, box2_edges)
            box1_edges = tf.roll(box1_edges, shift=-1, axis=-3)
        return edge_intersections, box1_edges, box2_edges

    def process_box_vector(self, vector_slices:list):
        assert(isinstance(vector_slices[0], tf.Tensor))
        result = []
        for slice in vector_slices:
            # Pull Box Probability from first slot in slice
            box_exists = slice[..., 0:1] 
            # slice the [rol, col, width, height, angle] columns off into the box_vector
            box_vector = slice[..., 1:]

            # Set the center of the box
            box_cx = tf.expand_dims(box_vector[..., 0], axis=-1)
            box_cy = tf.expand_dims(box_vector[..., 1], axis=-1)
            # Initialize a couple arrays to recieve the x / y coordinates
            x = np.zeros(box_vector.shape[:-1] + (4,), dtype=np.float32) 
            y = np.zeros(box_vector.shape[:-1] + (4,), dtype=np.float32) 
            # give width and height their own tensors to make it easier to write the math
            box_w = tf.expand_dims(box_vector[...,2], axis=-1)
            box_h = tf.expand_dims(box_vector[...,3], axis=-1)

            # Set x / y coordinates based on the center location and the width/height of the box
            x[..., 0:2] = tf.multiply(tf.add(box_cx, box_h/2), box_exists)
            x[..., 2:] = tf.multiply(tf.subtract(box_cx, box_h/2), box_exists)
            y[..., 0:1] = tf.multiply(tf.add(box_cy, box_w/2), box_exists)
            y[..., -1:] = tf.multiply(tf.add(box_cy, box_w/2), box_exists)
            y[..., 1:-1] = tf.multiply(tf.subtract(box_cy, box_w/2), box_exists)

            # Pull the angle out into its own tensor
            phi = -1 * (box_vector[..., -1:] - np.pi) * box_exists

            # Zip up (stack) the x / y coordinates so that we have (x, y) pairs
            box_points = tf.stack([x, y], axis=-1)
            box_center = tf.stack([box_cx, box_cy], axis=-1)

            if self.debug:
                print(f"box_exists shape: {box_exists.shape}")
                print(f"box_vector shape: {box_vector.shape}")
                print(f"box_vecotr:\n{box_vector[0, 5]}")


            # Rotate the box points to the correct orientation around their center
            box_processed = self.rotate_box_points(box_points, box_center, phi)
            result.append(box_processed) # pop that processed box in the list bad boy

        return result

    def get_corners(self, vectors):
        """
        Takes tensor batches, processes them into a format where we can find the GIoU, IoU
        and returns them to be tested.

        Returns:
            tensorflow.tensor(shape=(batch_size, x_cells, y_cells, 4, 2), dtype=float32)

            The returned value represents the 4 corners of the bounding box, rotated to align
            with the angle of the center axis.
        """ 
        # This first branch looks at whether the tensor is a prediction or a label
        # labels shape = (None, x_cells, y_cells, 19)
        # predictions  = (None, x_cells, y_cells, 31)
        if vectors.shape[-1] == 31:
            vector1 = tf.constant(vectors[...,13:19], dtype=tf.float32)
            vector2 = tf.constant(vectors[...,19:25], dtype=tf.float32)
            vector3 = tf.constant(vectors[...,25:], dtype=tf.float32)
            vector_slices = [vector1, vector2, vector3]
        else:
            vector_slices = [tf.constant(vectors[..., 13:19], dtype=tf.float32)]
        # send the tensors off in a list to be processed, then return the processed data
         
        return self.process_box_vector(vector_slices) 
        
    def rotate_box_points(self, box_points, box_center, angle):
        # Instatiate an array that will recieve the rotation vaules from the angle tensor
        R = np.zeros((angle.shape[:-1] + (2, 2)))

        # Insert Rotation Values
        R[..., 0:1, 0] = np.cos(angle)
        R[..., 0:1, 1] = np.sin(angle)*-1
        R[..., 1:, 0] = np.sin(angle)
        R[..., 1:, 1] = np.cos(angle)
        # Initialize values in tensors for matrix-multiplication
        A = tf.constant(tf.cast(R, dtype=tf.float32))
        B = tf.constant(tf.cast(box_points, dtype=np.float32))
        C = tf.subtract(B, box_center)
        # for the matmul you need to tranpose the x, y pairs into column vectors, then retranspose back
        # into their original form
        result = tf.linalg.matmul(A, tf.transpose(tf.subtract(B, box_center), perm=[0, 1, 2, 4, 3]))
        result = tf.add(tf.transpose(result, perm=[0, 1, 2, 4, 3]), box_center)

        return result

    def get_intersections(self, edge1, edge2):
        self.debug = True
        edge_a = edge1[..., 0:1, :, :]
        edge_b = edge2[..., 0:, :, :]

        x1 = edge_a[..., 0:1, 0:1]
        y1 = edge_a[..., 0:1, 1:]
        x2 = edge_a[..., 1:, 0:1]
        y2 = edge_a[..., 1:, 1:]

        x3 = edge_b[..., 0:1, 0:1]
        y3 = edge_b[..., 0:1, 1:]
        x4 = edge_b[..., 1:, 0:1]
        y4 = edge_b[..., 1:, 1:]
        
        denom =  (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)

        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
        zeros = tf.fill(tf.shape(ua), 0.0)
        ones = tf.fill(tf.shape(ua), 1.0)

        hi_pass_a = tf.math.less_equal(ua, ones)
        lo_pass_a = tf.math.greater_equal(ua, zeros)
        mask_a = tf.logical_and(hi_pass_a, lo_pass_a)

        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
        hi_pass_b = tf.math.less_equal(ub, ones)
        lo_pass_b = tf.math.greater_equal(ub, zeros)
        mask_b = tf.logical_and(hi_pass_b, lo_pass_b)

        mask = tf.logical_and(mask_a, mask_b)

        
        xnum = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
        ynum = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)

        x_i = (xnum / denom)
        y_i = (ynum / denom)

        mask = tf.cast(tf.squeeze(mask, axis=[-1]), dtype=tf.float32)
        intersections = tf.multiply(tf.squeeze(tf.stack([x_i, y_i], axis=-3), axis=[-2, -1]), mask)

        return intersections

    def get_edges(self, tensor):
        z = tf.roll(tensor, shift=-1, axis=-2)
        t = tf.stack([tensor, z], axis=-2)
        return t

    def find_inner_points(self, corners, edges):
        inner_points = None
        for _ in range(4):
            if not inner_points is None:
                inner_points = tf.concat([inner_points, self.is_inside(corners, edges)], axis=-2)
                corners = tf.roll(corners, shift=-1, axis=-2)
                continue
            inner_points = self.is_inside(corners, edges)
            corners = tf.roll(corners, shift=-1, axis=-2)
        return inner_points

    def is_inside(self, corners, box_edges):

        x = corners[..., 0:1, 0:1]
        y = corners[..., 0:1, 1:]

        result = False
        x_i = box_edges[..., 0:, 0:1, 0]
        y_i = box_edges[..., 0:, 0:1, 1]
        x_j = box_edges[..., 0:, 1:, 0]
        y_j = box_edges[..., 0:, 1:, 1]

        dir = (y - y_i) * (x_j - x_i) - (x - x_i) * (y_j - y_i)
        left = tf.greater(dir, 0)
        right = tf.less(dir, 0)
        check = tf.cast(tf.reduce_all(tf.logical_not(left, right), axis=-2), dtype=tf.float32)
        result = tf.squeeze(corners[..., 0:1, :], axis=-2) * check
        result = tf.reshape(result, [60, 12, 9, 1, 2])

        return result


    
    def clockwise_sort(self, point):
        x, y = point - self.inter_center
        # print(f"Point at ({x}, {y})\nhas arctan of {np.arctan(y/x):.4f}")
        return np.arctan2(y, x)
    
    def IoU(self):

        determinate_1 = np.ones((3,3), dtype=np.float32)
        determinate_2 = np.ones((3,3), dtype=np.float32)
        determinate_1[:,:2] = determinate_1[:,:2] * self.label_box[1:]
        determinate_2[:,:2] = determinate_2[:,:2] * self.pred_box[1:]
        
        box1_area = np.abs(np.linalg.det(determinate_1))
        box2_area = np.abs(np.linalg.det(determinate_2))
        # print(box1_area, box2_area) 
        intersection = self.intersection_area()
        union = (box1_area + box2_area) - intersection
        # print(f"{intersection}\n---------------\n{union}")

        return intersection / union

    def union(self):

        determinate_1 = np.ones((3,3), dtype=np.float32)
        determinate_2 = np.ones((3,3), dtype=np.float32)
        determinate_1[:,:2] = determinate_1[:,:2] * self.label_box[1:]
        determinate_2[:,:2] = determinate_2[:,:2] * self.pred_box[1:]
        
        box1_area = np.abs(np.linalg.det(determinate_1))
        box2_area = np.abs(np.linalg.det(determinate_2))
        intersection = self.intersection_area()
        union = (box1_area + box2_area) - intersection

        return union

    def IoU(self, y_true, pred):
        pass



    def _debug_get_intersections(self, edge1, edge2, level:int=1):
        self.debug = True
        edge_a = edge1[..., 0:1, :, :]
        edge_b = edge2[..., 0:, :, :]
        if self.debug:
            print(f"edge_a shape: {edge_a.shape}")
            print(f"edge_b shape: {edge_b.shape}")
            if level > 1:
                print(f"edge_a points:\n{tf.squeeze(edge_a[0,5, 4, 0:])}")
                print(f"edge_b points:\n{tf.squeeze(edge_b[0,5, 4, 0:])}")

        x1 = edge_a[..., 0:1, 0:1]
        y1 = edge_a[..., 0:1, 1:]
        x2 = edge_a[..., 1:, 0:1]
        y2 = edge_a[..., 1:, 1:]
        if self.debug:
            print(f"x1 shape: {x1.shape}")
            print(f"y1 shape: {y1.shape}")
            if level > 1:
                print(f"x1 value: {x1[0,5,4]}")
                print(f"y1 value: {y1[0,5,4]}")
                print(f"x2 value: {x2[0,5,4]}")
                print(f"y2 value: {y2[0,5,4]}")

        x3 = edge_b[..., 0:1, 0:1]
        y3 = edge_b[..., 0:1, 1:]
        x4 = edge_b[..., 1:, 0:1]
        y4 = edge_b[..., 1:, 1:]
        if self.debug:
            if level > 1:
                print(f"x3 value: {x3[0,5,4]}")
                print(f"y3 value: {y3[0,5,4]}")
                print(f"x4 value: {x4[0,5,4]}")
                print(f"y4 value: {y4[0,5,4]}")
        
        denom =  (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        
        if self.debug:
            print(f"denom shape: {denom.shape}")
            print(f"denom:\n{tf.squeeze(denom[0, 5, 4], axis=[-2, -1])}")

        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom

        zeros = tf.fill(tf.shape(ua), 0.0)
        ones = tf.fill(tf.shape(ua), 1.0)

        hi_pass_a = tf.math.less_equal(ua, ones)
        lo_pass_a = tf.math.greater_equal(ua, zeros)
        mask_a = tf.logical_and(hi_pass_a, lo_pass_a)

        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
        hi_pass_b = tf.math.less_equal(ub, ones)
        lo_pass_b = tf.math.greater_equal(ub, zeros)
        mask_b = tf.logical_and(hi_pass_b, lo_pass_b)

        mask = tf.logical_and(mask_a, mask_b)

        if self.debug:
            print(f"\nua value:       {tf.squeeze(ua[0, 5, 4], axis=[-2, -1])}")
            if level > 1:
                print(f"hi_pass value:  {tf.squeeze(hi_pass_a[0, 5, 4], axis=[-2, -1])}")
                print(f"lo_pass value:  {tf.squeeze(lo_pass_a[0, 5, 4], axis=[-2, -1])}")
            print(f"\nmask_a value:   {tf.squeeze(mask_a[0, 5, 4], axis=[-2, -1])}")
            print(f"\nub value:       {tf.squeeze(ub[0, 5, 4], axis=[-2, -1])}")
            if level > 1:
                print(f"hi_pass value:  {tf.squeeze(hi_pass_b[0, 5, 4], axis=[-2, -1])}")
                print(f"lo_pass value:  {tf.squeeze(lo_pass_b[0, 5, 4], axis=[-2, -1])}")
            print(f"\nmask_b value:   {tf.squeeze(mask_b[0, 5, 4], axis=[-2, -1])}")
            print(f"\nmask value:     \n{tf.squeeze(mask[0, 5, 4], axis=[-2])}")

        
        xnum = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
        ynum = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
        x_i = (xnum / denom)
        y_i = (ynum / denom)
        mask = tf.cast(tf.squeeze(mask, axis=[-1]), dtype=tf.float32)
        intersections = tf.multiply(tf.squeeze(tf.stack([x_i, y_i], axis=-3), axis=[-2, -1]), mask)
        if self.debug:
            print(f"mask_reshape: {mask.shape}")
            print(f"Intersection shape: {intersections.shape}")
            print(f"Intersection Points:\n{intersections[0, 5, 4]}")
        return intersections

