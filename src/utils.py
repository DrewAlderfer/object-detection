from typing import Union, Tuple, List
from pathlib import Path
import pprint as pp
from numpy.typing import ArrayLike, NDArray
from pandas.core.window import rolling
import tensorflow as tf
from tensorflow.types.experimental import TensorLike
from sklearn.cluster import KMeans

from pycocotools.coco import COCO
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tensorflow.keras.utils import load_img

def generate_anchors(labels:NDArray[np.float32],
                     boxes_per_cell:int=3,
                     xdivs:int=12,
                     ydivs:int=9, 
                     **kwargs):
    """
    Function:
        Generates anchors box initialization values from a set of training labels.
    Parameters:
        labels:
            An array or tensor of bounding box labels.
        boxes_per_cell:
            The number of boxes to be predicted for each cell in the YOLO grid. Effectively here this
            is the number of clusters to be returned from the KMeans algorithm.
    Returns:
        An array of initial values for the prediction bounding boxes.
    """
    rowcount = np.array(labels.sum(-1) != 0).sum(-1)
    max_obj = np.max(rowcount)
    # flatten labels
    input_shape = labels.shape
    batch, max_obj, _ = input_shape
    box_labels = tf.reshape(labels, [batch * max_obj, 19])[..., 14:].numpy()
    box_labels[:, 0:1] = 0 
    box_labels[:, 1:2] = 0 
    # mask no object
    mask_labels = tf.cast(tf.reduce_sum(box_labels, axis=-1) > .001, dtype=tf.bool)
    box_labels = tf.boolean_mask(box_labels, mask_labels, axis=0).numpy()
    print(f"box_labels: {box_labels.shape}")
    # find clusters
    clusters = KMeans(n_clusters=boxes_per_cell, max_iter=100, **kwargs)
    clusters.fit(box_labels)
    # retrieve the cluster space mapping
    cls = clusters.predict(box_labels)
    cls = np.expand_dims(cls, axis=-1)
    """ 
    The idea here is to pull the centroids out of the KMeans object by predicting on 
    the labels and then looping through the class predictions and taking the mean of 
    each point in the class.
    """
    centroid_locations = np.zeros((boxes_per_cell, 5), dtype=np.float32)
    for idx in range(boxes_per_cell):
        filter = tf.where(tf.equal(cls, idx),
                          box_labels,
                          np.zeros(box_labels.shape, dtype=np.float32))

        mask = tf.cast(tf.abs(tf.reduce_sum(filter, axis=-1)) > .001, dtype=tf.float32)
        # average = np.append(np.ones((1,), dtype=np.float32), tf.reduce_sum(filter, axis=0) / tf.reduce_sum(mask, axis=0))
        average = tf.reduce_sum(filter, axis=0) / tf.reduce_sum(mask, axis=0)
        centroid_locations[idx, :] = average 

    """ 
    idx_template: A template of the grid where the values are indices of cells in the grid.
    knudge_coords: A map of center-point coordinates for each cell in the grid.""" 
    idx_template = np.fromfunction(lambda x, y: [x, y], shape=(xdivs, ydivs), dtype=np.float32)   
    knudge_coords = (np.stack((idx_template[0], idx_template[1]), axis=-1) + .5)/ np.array([xdivs, ydivs])
    print(f"knudge_coords: {knudge_coords.shape}")
    """
    Then we're creating the return value by filling an array with the shape of our grid with the
    anchor box values and then assigning the x, y point values for each set of anchor boxes in the
    array to the centerpoints we created in the knudge_coords array."""
    knudge_coords = tf.reshape(knudge_coords, shape=(xdivs, ydivs, knudge_coords.shape[-1]))
    anchor_box_template = np.full((1, xdivs, ydivs, boxes_per_cell, 5), centroid_locations)
    print(f"anchor_box_template: {anchor_box_template.shape}")
    anchor_box_template[..., 0] = anchor_box_template[..., 0] + knudge_coords[..., 0:1]
    print(f"anchor_box_template: {anchor_box_template.shape}")
    anchor_box_template[..., 1] = anchor_box_template[..., 1] + knudge_coords[..., 1:]
    print(f"anchor_box_template: {anchor_box_template.shape}")
    return anchor_box_template

def stack_anchors(anchors:NDArray[np.float32]) -> NDArray:
    """
    Function stacks anchor box vectors:
        Parameters:
            anchors: An array or tensor representing bounding box predictions.
                    [P(class_1), ..., P(class_n), P(Obj), X1, Y1, W1, H1, A1, ..., Xn, ..., An]
        Returns:
            NDArray of shape (input.shape[0], 5): 
                    Represents a list of anchor boxes for each image
                    in the input tensor.

                    [[X1, Y1, W1, H1, A1],
                     ...,
                     [Xn, Yn, Wn, Hn, An]]
    """ 
    if len(anchors.shape) > 4:
        batch, xdivs, ydivs, num_boxes, units = anchors.shape
        anchors = anchors.reshape((batch, xdivs * ydivs, num_boxes, units))
    else:
        batch = 1
        xdivs, ydivs, units = anchors.shape
        anchors = anchors.reshape((xdivs * ydivs,) + (units,))
    print(f"units spec1: {units}")
    # size = xdivs * ydivs
    # num_boxes = int((anchors.shape[-1] - 13) / 6)
    print(f"anchors spec1: {anchors.shape}")
    # result = np.reshape(anchors[..., 13:], (batch, size, num_boxes * 6))
    # print(f"result spec2: {result.shape}")
    # result = np.expand_dims(result.reshape(batch, size, num_boxes, 6), axis=-2)
    # print(f"result spec3: {result.shape}")
    # result = np.reshape(np.transpose(np.full(result.shape[:-2] + (3, result.shape[-1]), result), (0, 2, 1, 3, 4))[..., 0, 1:], (batch, size * num_boxes, 5))
    # print(f"result spec4: {result.shape}")

    return anchors

def get_corners(box_vectors:NDArray[np.float32],
                img_width:int=512,
                img_height:int=384,
                debug:bool=False) -> NDArray[np.float32]:
    """
    Function that takes a tensor of box label vectors and returns the corners of the bounding box
    described by the vector [x, y, w, h, phi].
    """
    if debug:
        print("-" * 15)
        print("Get Corners Function Start:")
        print(f"inputs:\nbox_vectors: {box_vectors.shape}, {box_vectors.dtype}")

    rank = len(box_vectors.shape)
    if box_vectors.shape[-1] > 5:
        box_vectors = box_vectors[..., 14:19]
    # if rank > 3:
    #     print(f"get_corners error: {box_vectors.shape[0:2] + (np.cumprod(box_vectors.shape[2:-1])[-1], box_vectors.shape[-1])}")
    #     box_vectors = tf.reshape(box_vectors, box_vectors.shape[:2] + (np.cumprod(box_vectors.shape[2:-1])[-1], box_vectors.shape[-1]))
    #     print(f"box_vectors: {box_vectors.shape}")
    #     x = np.zeros(box_vectors.shape[0:-1] + (4,), dtype=np.float32) 
    #     y = np.zeros(box_vectors.shape[0:-1] + (4,), dtype=np.float32) 
    # else:
    #     x = np.zeros(box_vectors.shape[0:-1] + (4,), dtype=np.float32) 
    #     y = np.zeros(box_vectors.shape[0:-1] + (4,), dtype=np.float32) 

    # ---- Test
    x = tf.zeros(box_vectors.shape[0:-1] + (4,), dtype=tf.float32) 
    y = tf.zeros(box_vectors.shape[0:-1] + (4,), dtype=tf.float32) 
    # ---- Test

    cx, cy, = box_vectors[..., 0:1], box_vectors[..., 1:2]
    phi = box_vectors[..., -1:] * (2 * np.pi) - np.pi

    # give width and height their own tensors to make it easier to write the math
    w = tf.expand_dims(box_vectors[...,2], axis=-1)
    h = tf.expand_dims(box_vectors[...,3], axis=-1)
    # Set x / y coordinates based on the center location and the width/height of the box
    x1 = x[..., 0:2] + tf.add(cx, w/2)
    x2 = x[..., 2:] + tf.subtract(cx, w/2)
    y1 = y[..., 0:1] + tf.add(cy, h/2)
    y2 = y[..., 1:-1] + tf.subtract(cy, h/2)
    y3 = y[..., -1:] + tf.add(cy, h/2)
    x = tf.concat([x1, x2], axis=-1)
    y = tf.concat([y1, y2, y3], axis=-1)
    # Subtract the center point from each point in the corners. This centers the points around zero
    # so that we can rotate them around their local center point.
    scale = tf.constant([img_width, img_height], dtype=tf.float32)
    corner_points = tf.stack([x, y], axis=-1) * scale
    center_point = tf.stack([cx, cy], axis=-1) * scale
    centered = tf.subtract(corner_points, center_point) 

    # Set up the rotation matrix tensor
    # print(f"phi: {phi.shape}")
    cos, sin = tf.cos(phi), tf.sin(phi)
    R = tf.stack([tf.concat([cos, -sin], axis=-1),
                  tf.concat([sin, cos], axis=-1)], axis=-2)
    # Take a shape like: (a, a, b, c) or (a, a, b)
    # transpose it into: (b, c, a, a) or (b, a, a)
    # r_idx = tuple(range(len(R.shape)))
    # R = tf.transpose(R, r_idx[2:] + r_idx[0:2])
    # -------------
    # Debug Section
    # -------------
    if debug:
        print("Get Corners Function End:")
        print(f"output: {tf.add(tf.matmul(centered, R), center_point).shape}, {tf.add(tf.matmul(centered, R), center_point).dtype}")
        print("-" * 15)
    # -------------
    # Return the points with the rotation applied and then the center point added back in.
    return tf.add(tf.matmul(centered, R), center_point)

def get_edges(bbox_tensor:Union[tf.Tensor, NDArray], debug:bool=False) -> TensorLike:
    """
    Takes a set of box corners as a tensor like object and returns the edges of that shape.
    """
    # -------------
    # Debug Section
    # -------------
    if debug:
        print("-" * 15)
        print("Get Edges Function Start:")
        print(f"input:\nbbox_tensor: {bbox_tensor.shape}, {bbox_tensor.dtype}\n")
    # ------------------------
    # Function Body
    # ------------------------
    z = tf.roll(bbox_tensor, shift=-1, axis=-2)

    # -------------
    # Debug Section
    # -------------
    if debug:
        print("Get Edges End:")
        print(f"output: {tf.stack([bbox_tensor, z], axis=-2).shape}, {tf.stack([bbox_tensor, z], axis=-2).dtype}")
    # -------------

    return tf.stack([bbox_tensor, z], axis=-2)

def find_intersection_points(box1_edges:TensorLike,
                             box2_edges:TensorLike,
                             debug:bool=False,
                             **kwargs) -> TensorLike:
    """
    Parameters:
        Takes two tensors with the edges of label and anchor bounding boxes.

    Returns: 
        a tensor containing the intersection points of each label bounding box with each anchor
        box.
    """
    # -------------
    # Debug Section
    # -------------
    if debug:
        print("-" * 30)
        print("Find Intersection Points Function Start:")
        print(f"inputs:\nbox1: {box1_edges.shape}, {box1_edges.dtype}\nbox2_edges: {box2_edges.shape}, {box2_edges.dtype}")

    # --------------------------------------------
    # Find Intersection Points Function
    # --------------------------------------------
    if box1_edges.shape != box2_edges.shape:
        box1_edges = pump_tensor(box1_edges, **kwargs)
        box2_edges = stretch_tensor(box2_edges)
    box_shape = box1_edges.shape[:-3]
    # box1_edges = tf.constant(box1_edges.shape[:4] + (4,) + box1_edges.shape[-2:], tf.expand_dims(box1_edges, axis=-3))
    box1_edges = tf.broadcast_to(tf.expand_dims(box1_edges, axis=-3), shape=box1_edges.shape[:-3] + (4,) + box1_edges.shape[-3:])
    box2_edges = tf.expand_dims(box2_edges, axis=-4)

    # -------------
    # Debug Section
    # -------------
    if debug:
        result = tf.reshape(get_intersections(box1_edges, box2_edges, **kwargs), shape=box_shape + (16, 2))
        print("Find Intersection Points Function End:")
        print(f"output: {result.shape}")
        print("-" * 30)
        return result
    # -------

    return tf.reshape(get_intersections(box1_edges, box2_edges, **kwargs), shape=box_shape + (16, 2))

def get_intersections(box1_edge, box2_edge, debug:bool=False, **kwargs) -> TensorLike:
    """
    calculates the edge intersections between two sets of bounding boxes. 

    Returns:
        tf.Tensor with a list of intersections (or zeros for no intersection) between each edge, of
        the two boxes.
    """
    # -------------
    # Debug Section
    # -------------
    if debug:
        print("-" * 15)
        print("Get Intersections Function Start:")
        print(f"inputs:\nbox1_edge: {box1_edge.shape}, {box1_edge.dtype}\nbox2_edge: {box2_edge.shape}, {box2_edge.dtype}")
    # ------------------------------
    # Get Intersections Function
    # ------------------------------
    edge_a = box1_edge[..., 0:1, :, :]
    edge_b = box2_edge[..., :, :, :]
    x1 = edge_a[..., 0:1, 0:1]
    y1 = edge_a[..., 0:1, 1:]
    x2 = edge_a[..., 1:, 0:1]
    y2 = edge_a[..., 1:, 1:]

    x3 = edge_b[..., 0:1, 0:1]
    y3 = edge_b[..., 0:1, 1:]
    x4 = edge_b[..., 1:, 0:1]
    y4 = edge_b[..., 1:, 1:]

    # this is kind of like taking the area of a plane created between the two edges and then
    # subtracting them. If it equals zero then the edges are colinear.
    denom =  (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)

    # These check to see if the point of intersection falls along the line segments of the edge
    # ua and ub have a domain of 0 to 1 if the point is a valid intersection along the segments.
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
    mask_a = tf.cast(tf.math.logical_and(tf.greater(ua, 0), tf.less(ua, 1)), dtype=tf.bool)
    mask_b = tf.cast(tf.math.logical_and(tf.greater(ub, 0), tf.less(ub, 1)), dtype=tf.bool)
    
    # Combine the masks and change them to floats for removing non-intersections from the result.
    mask = tf.cast(tf.squeeze(tf.logical_and(mask_a, mask_b), axis=[-1]), dtype=tf.float32)

    # This actually just says where the intersection is 
    xnum = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
    ynum = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
    x_i = tf.math.divide_no_nan(xnum, denom)
    y_i = tf.math.divide_no_nan(ynum, denom)
    # -------------
    # Debug Section
    # -------------
    if debug:
        print("Get Intersections Function End:")
        print(f"output: {tf.multiply(tf.squeeze(tf.stack([x_i, y_i], axis=-3), axis=[-2, -1]), mask).shape}")
        print("-" * 15)

    # Finish by using the mask to multiply any intersections that didn't fall along the segments
    # by zero and all the ones that did by one.
    return tf.multiply(tf.squeeze(tf.stack([x_i, y_i], axis=-3), axis=[-2, -1]), mask)

def construct_intersection_vertices(labels,
                                    anchors,
                                    xdivs:int=12,
                                    ydivs:int=9, 
                                    debug:bool=False,
                                    **kwargs) -> TensorLike:
    """
    This function organizes the process of finding all points of the intersection between bounding
    boxes from the label set and all boxes from the anchor box set.

    Returns:
        A tensor with shape (B, M, M * 3, 24, 2)

    TO DO:
        Generalize all of these functions a little more. Instead of using for loops you can probably
        just "pump" the tensors up like you are when comparing a set of labels vs. anchors.

        So you would just concat the labels from (x, 108, 108, 4, 2) into (x, 108, 324, 4, 2). Just
        'drop' it on itself for each anchor box.
    """
    # -------------
    # Debug Section
    # -------------
    if debug:
        print("-" * 30)
        print("Construct_Intersection Points Function Start:")
        print(f"inputs:\nlabels: {labels.shape}, {labels.dtype}\nanchors: {anchors.shape}, {anchors.dtype}")

    pumps = anchors.shape[1] // (xdivs * ydivs)
    label_corners, anchor_corners = get_corners(labels), get_corners(anchors)
    label_edges, anchor_edges = get_edges(label_corners), get_edges(anchor_corners)
    inner_points = find_inner_points([label_corners, label_edges], [anchor_corners, anchor_edges], num_pumps=pumps)
    # print(f"inner_points: {inner_points.shape}, {len(inner_points.shape)} ")
    # print(f"labels: {labels.shape} {len(labels.shape)}")
    if len(inner_points.shape) == len(labels.shape):
        box_exists = tf.reshape(labels[..., 13:14], labels.shape[:2] + (labels.shape[2] * labels.shape[3], 1))
    else:
        box_exists = labels[..., 13:14]
    dif = len(inner_points.shape) - len(box_exists.shape)
    dif = tf.TensorShape([1 for x in range(dif)])

    if debug:
        print(f"inner_points: {inner_points.shape} | {box_exists.shape} < box_exists")
        print(f"dif: {dif}")
    box_exists = tf.reshape(box_exists, box_exists.shape[:2] + box_exists.shape[2:] + dif)

    if debug:
        print(f"inner_points: {inner_points.shape} | {box_exists.shape} < box_exists")

    inner_points = inner_points * box_exists
    intersection_points = find_intersection_points(label_edges, anchor_edges, num_pumps=pumps, **kwargs)
    if debug:
        result = tf.concat([intersection_points, inner_points], axis=-2), label_corners, anchor_corners
        print("Construct_Intersection Points Function End:")
        print(f"output: {result[0].shape}")
        print("-" * 30)
        return result

    return  tf.concat([intersection_points, inner_points], axis=-2), label_corners, anchor_corners

def find_inner_points(label:list, anchors:list, debug:bool=False, **kwargs):
    """
    Takes a set of bounding box corners and the edges from a rectangle and returns the corner point
    if it is inside the egdes of the shape.
    Parameters:
        corners: 
            Bound Box Corners points Tensor with shape: (B, M) + (4, 2)
        edges:
            Bounding Box edges Tensor with shape:       (B, M) + (4, 2, 2)

    Returns:
        tf.Tensor object of shape (B, M, M) + (4, 2)
    """
    # -------------
    # Debug Section
    # -------------
    if debug:
        print("-" * 15)
        print("Find Inner Points Function Start")
        print(f"inputs:\nlabels:\ncorners: {label[0].shape}, edges: {label[1].shape}")
        print(f"anchors:\ncorners: {anchors[0].shape}, edges: {anchors[1].shape}")

    label_corners, label_edges = label[0], label[1]
    anchor_corners, anchor_edges = anchors[0], anchors[1]
    # ----------
    # Prepare Corner Tensors by repeating each set of corners 4 times.
    # ---------
    if label_corners.shape != anchor_corners.shape:
        label_corners = pump_tensor(label_corners, **kwargs)
        anchor_corners = tf.expand_dims(anchor_corners, axis=1)

    label_corners = tf.broadcast_to(tf.expand_dims(label_corners, axis=-2), 
                                    shape=label_corners.shape[:-1] + (4, 2))

    anchor_corners = tf.broadcast_to(tf.expand_dims(anchor_corners, axis=-2),
                                     shape=anchor_corners.shape[:-1] + (4, 2))
    # ---------
    # Open up a dimension of edge tensors so that the operation can be broadcast into it.
    # ---------
    if label_edges.shape != anchor_edges.shape:
        label_edges = pump_tensor(label_edges, **kwargs)
        anchor_edges = tf.expand_dims(anchor_edges, axis=1)

    label_edges = tf.expand_dims(label_edges, axis=-4)
    anchor_edges = tf.expand_dims(anchor_edges, axis=-4)
    # -------------
    # Debug Section
    # -------------
    if debug:
        result = tf.concat([is_inside(label_corners, anchor_edges), 
                            is_inside(anchor_corners, label_edges)], axis=-2)
        print("Find Inner Points Function End:")
        print(f"outputs: {result.shape}, {result.dtype}")
        print("-" * 15)
        return result
    # ------
    return tf.concat([is_inside(label_corners, anchor_edges), 
                      is_inside(anchor_corners, label_edges)], axis=-2)

def is_inside(corners:TensorLike, box_edges:TensorLike, debug:bool=False) -> TensorLike:
    """
    Determines whether a point falls inside the edges of a convex shape by determining which side
    of each edge the point falls along moving clockwise along the edges.
    
    Returns:
        tf.Tensor object containing points that are fall inside the shape, and zeros out all points
        that do not.
    """
    # -------------
    # Debug Section
    # -------------
    if debug:
        print("-" * 15)
        print("Is Inside Start:")
        print(f"inputs:\ncorners: {corners.shape}, {corners.dtype}\nbox_edges: {box_edges.shape}, {box_edges.dtype}")
    # Gets the x, y coordinates for the first corner point in the each corner set of the tensor.
    x = corners[..., 0:1, 0:1]
    y = corners[..., 0:1, 1:]
    # Gets the start and stop coordinates for each edge of the shape.
    x_i = box_edges[..., 0:, 0:1, 0]
    y_i = box_edges[..., 0:, 0:1, 1]
    x_j = box_edges[..., 0:, 1:, 0]
    y_j = box_edges[..., 0:, 1:, 1]
    # TO DO:
    # A lot of these point comparison equations seem to be find the difference in the area between
    # two sections of a shape. You should draw this out step by step to understand better what this
    # is actually calculating.
    #
    # This determines which side of the line the point is on moving along the edges clockwise.
    dir = (y - y_i) * (x_j - x_i) - (x - x_i) * (y_j - y_i)
    left = tf.cast(tf.greater(dir, 0), dtype=tf.bool)
    right = tf.cast(tf.less(dir, 0), dtype=tf.bool)

    # This returns 1 if the point is on the same side of all the edges; 0 if it is not.
    check = tf.cast(tf.reduce_all(tf.not_equal(left, right), axis=-2), dtype=tf.float32)
    # check = tf.cast(tf.reduce_all(tf.logical_not(left, right), axis=-2), dtype=tf.float32)
    # Zero out the points not inside the shape.
    corners = tf.squeeze(corners[..., 0:1, :], axis=-2)
    # -------------
    # Debug Section
    # -------------
    if debug:
        result = tf.multiply(tf.cast(corners, dtype=tf.float32), check)
        print("Is Inside End:")
        print(f"output: {result}")
        print("-" * 15)
        return result

    return tf.multiply(tf.cast(corners, dtype=tf.float32), check)

def intersection_area(intersection_points:TensorLike, debug:bool=False, **kwargs) -> TensorLike:
    """
    Function that takes a set of intersection points and returns a set of the areas.
    """
    # -------------
    # Debug Section
    # -------------
    if debug:
        print("-" * 15)
        print("Intersection Area Function Start:")
        print(f"input:\nintersection_points: {intersection_points.shape}, {intersection_points.dtype}")

    nonzero = tf.cast(intersection_points > 0, dtype=tf.float32)
    mask = tf.cast(intersection_points > 0, dtype=tf.bool)

    denomenator = tf.reduce_sum(nonzero, axis=-2)
    center = tf.expand_dims(tf.reduce_sum(intersection_points, axis=-2) / denomenator, axis=-2)
    center_adj_points = tf.where(mask, 
                                 intersection_points - center,
                                 nonzero)
    # Get the angles of each point in the intersection to the center point of the intersection shape.
    angles = tf.math.atan2(center_adj_points[..., :, 1:], center_adj_points[..., :, 0:1]) 
    point_angle = tf.where(mask, 
                           tf.add(angles, tf.constant(np.pi, dtype=tf.float32, shape=(1,))),
                           nonzero)
    # order the points by their angle. The transpose here was necessary reordering the points. So
    # for a few steps the x and y values are not paired in the last axis of the tensor but remain
    # linked by their shared index in the ordering.
    shape_dim_flip = tf.constant([0], shape=(len(point_angle.shape)-2,), dtype=tf.int32)
    shape_dim_flip = tf.concat([shape_dim_flip, tf.constant([1, -1], dtype=tf.int32)], axis=0)
    # print(f"shape_dim_flip: {shape_dim_flip}")
    # print(f"point_angle: {point_angle.shape}")
    shape_dim_flip = tf.range(len(point_angle.shape), dtype=tf.int32) + shape_dim_flip
    # print(f"shape_dim_flip: {shape_dim_flip}")
    point_indices = tf.argsort(tf.transpose(point_angle, perm=shape_dim_flip), direction="DESCENDING", axis=-1)
    # just transposing the points to match the ordered indexes
    points_T = tf.transpose(intersection_points, shape_dim_flip)
    """
    Create a template to double up the point_order indices. This is so that when we create gather
    the intersection points into a tensor sorted by the angle of the points to the center point
    it will insert a copy of each point right below it's ordered spot. This is neccessary because
    for creating the full set of points in the edges of our triangles.
    """
    idx_template = tf.broadcast_to(tf.range(point_indices.shape[-1]), shape=point_indices.shape)
    # this creates the doubled set of ordered indexes
    point_order = tf.gather(point_indices, tf.sort(tf.concat([idx_template, idx_template], axis=-1), axis=-1), batch_dims=-1)
    # this gathers the point into a tensor ordered by their angle to the center point.
    edge_points = tf.transpose(tf.gather(points_T, point_order, batch_dims=-1)[..., :16], perm=shape_dim_flip)
    # this rolls the non-zero values one step. This sets up the tensor to be easily split into the
    # outer edge of the triangle making up the intersection shape.
    edge_points = tf.where(edge_points > 0, tf.roll(edge_points, shift=-1, axis=-2), 0)
    """
    The next few steps function as follows:
        1. Find the index of the first [0, 0] point in each set of intersection points.
            - after rolling the point axis of the tensor this index is the last point of our edges
        2. create a tensor that just represents the indexes for each x, y value in our tensor
        3. create a tensor that is just the first point of every x, y set.
        4. compare the last point index with the full index set and where they match insert the first
          point from each set.

    Now we have a full ordered set of edges for each intersection. These will be combined with the
    center point from before to create a set of traingles from which we can calculate the area of
    the intersection.
    """
    # 1
    last_point = tf.reduce_sum(edge_points, axis=-1)
    last_point = tf.math.count_nonzero(last_point, keepdims=True, axis=-1, dtype=tf.int32)
    last_point = tf.expand_dims(last_point, axis=-1)
    # 2
    idx = tf.transpose(tf.broadcast_to(np.arange(16, dtype=np.int32),
                                       last_point.shape[:-1] + (16,)), perm=shape_dim_flip)
    # 3
    first_point = tf.squeeze(tf.gather(edge_points, last_point - last_point,
                                       axis=-2, batch_dims=-2), axis=-2)
    # 4
    outer_edges = tf.reshape(tf.where(idx != last_point, edge_points, first_point), 
                             edge_points.shape[:-2] + (8, 2, 2))
    # Create Triangles and take their area via the determinant
    # Reshape the center point of each intersection to match the outer edges 
    center = tf.broadcast_to(tf.expand_dims(center, axis=-2),
                             outer_edges.shape[:-2] + (1, 2))
    # Insert the center point into each edge to form the triangles
    triangle_points = tf.concat([outer_edges, center],
                                axis=-2, name="triangle_outer_edges")
    # create a tensor filled with ones that will be concatenated to the last axis of the triangles.
    ones = tf.ones(triangle_points.shape[:-1] + (1,), dtype=tf.float32)
    # find the area of each triangle using the determinant and then sum them to find the area of
    # intersection.
    triangle_areas = tf.abs(tf.divide(tf.linalg.det(tf.concat([triangle_points, ones], axis=-1)), 2))
    # -------------
    # Debug Section
    # -------------
    if debug:
        result = tf.reduce_sum(triangle_areas, axis=-1, name="intersection_areas")
        print("Intersection Area Function End:")
        print(f'output: {tf.reduce_sum(triangle_areas, axis=-1, name="intersection_areas").shape}')
        print("-" * 15)
        return result

    return tf.reduce_sum(triangle_areas, axis=-1, name="intersection_areas")

def union_area(label_corners:TensorLike,
               anchor_corners:TensorLike,
               intersection:TensorLike,
               debug:bool=False,
               **kwargs):
    """
    Function that takes two tensors of bounding boxes + the intersections between them and returns
    the union between them.
    """
    # -------------
    # Debug Section
    # -------------
    if debug:
        print("-" * 15)
        print("Union Area Start:")
        print(f"input:\nlabel_corners: {label_corners.shape}, {label_corners.dtype}")
        print(f"input:\nanchor_corners: {anchor_corners.shape}, {anchor_corners.dtype}")
        print(f"input:\nintersection: {intersection.shape}, {intersection.dtype}")
    # -----------------------------------
    # Union Area Function
    # -----------------------------------
    if label_corners.shape != anchor_corners.shape:
        label_corners = pump_tensor(label_corners, **kwargs)
        anchor_corners = stretch_tensor(anchor_corners)
    box1_points, box2_points = label_corners[..., :3, :], anchor_corners[..., :3, :]
    ones1 = tf.ones(box1_points.shape[:-1] + (1,), dtype=tf.float32)
    ones2 = tf.ones(box2_points.shape[:-1] + (1,), dtype=tf.float32)

    area1 = tf.abs(tf.linalg.det(tf.concat([box1_points, ones1], axis=-1)))
    area2 = tf.abs(tf.linalg.det(tf.concat([box2_points, ones2], axis=-1)))
    # -------------
    # Debug Section
    # -------------
    if debug:
        result = (area1 + area2) - intersection
        print("Union Area End:")
        print(f"output: {result.shape}, {result.dtype}")
        print("-" * 15)
        return result

    return (area1 + area2) - intersection

def calculate_giou(label_corners:TensorLike,
                   anchor_corners:TensorLike,
                   union:TensorLike,
                   intersection:TensorLike,
                   debug:bool=False) -> TensorLike:
    # -------------
    # Debug Section
    # -------------
    if debug:
        print("-" * 15)
        print("GIoU Function Start:")
        print(f"input:\nlabel_corners: {label_corners.shape}, {label_corners.dtype}")
        print(f"input:\nanchor_corners: {anchor_corners.shape}, {anchor_corners.dtype}")
        print(f"input:\nintersection: {intersection.shape}, {intersection.dtype}")
        print(f"input:\nunion: {union.shape}, {union.dtype}")
    # End Debug
    # ----- 
    if label_corners.shape != anchor_corners.shape:
        label_corners = pump_tensor(label_corners, num_pumps=9)
        anchor_corners = tf.broadcast_to(stretch_tensor(anchor_corners), shape=label_corners.shape)

    all_points = tf.concat([label_corners, anchor_corners], axis=-2)
    axes = tuple(range(len(all_points.shape)))
    all_points = tf.sort(tf.transpose(all_points, axes[:-2] + (axes[-1], axes[-2])), axis=-1)
    all_points = tf.transpose(all_points, axes[:-2] + (axes[-1], axes[-2]))

    gMax = tf.reduce_max(all_points, axis=-2)
    gMin = tf.reduce_min(all_points, axis=-2)
    wh = gMax - gMin

    C = tf.squeeze(wh[..., 0:1] * wh[..., 1:])

    # -------------
    # Debug Section
    # -------------
    if debug:
        result = (intersection / union) - tf.abs(tf.divide(C, union)) / tf.abs(C)
        print("GIoU Function End:")
        print(f"output: {result.shape}, {result.dtype}")
        print("-" * 15)
        return result
    # End Debug
    # ----- 
    return (intersection / union) - tf.abs(tf.divide(C, union)) / tf.abs(C)
     


def stretch_tensor(box_edges):
    # -------------
    # Debug Section
    # -------------
    if debug:
        print("-" * 15)
        print("Stretch Tensor Start:")
        print(f"input: {box_edges.shape}")
        print(f"output: {tf.expand_dims(box_edges, axis=1).shape}")
        print("-" * 15)
    return tf.expand_dims(box_edges, axis=1)

def pump_tensor(box_edges, num_cells:int=108, num_pumps:int=3, debug:bool=False):
    """
    This function takes a bounding box edge tensor (h, grid_x * grid_y, 4, 2) and returns a 
    'pumped-up' version of it to be 'rolled' against another tensor for calculating intersections.
    
    Returns:
        NDArray with shape (B, M, 4, 2, 2)
    """
    # -------------
    # Debug Section
    # -------------
    if debug:
        print("-" * 15)
        print("Pump Tensor Function Start:")
        print(f"inputs:\nbox_edges: {box_edges.shape}, {box_edges.dtype}")
    batch, num_boxes = box_edges.shape[0:2]
    end = box_edges.shape[2:]
    step1 = tf.expand_dims(box_edges, axis=2)
    step2 = tf.fill(dims=(batch, num_boxes, num_cells * num_pumps) + end, value=step1)
    # -------------
    # Debug Section
    # -------------
    if debug:
        print("Pump Tensor Function End:")
        print(f"output: {step2.shape}, {step2.dtype}")
        print("-" * 15)

    return step2

def nonmaxsuppresion(labels:TensorLike, anchors:TensorLike) -> TensorLike:
    """
    Nonmax Suppression of Anchor boxes based on GIoU calculation.

    Params:
        labels:     tf.types.experimental.TensorLike object with shape (batch_size, 18, 19)
        anchors:    tf.types.experimental.TensorLike object with shape (batch_size, 972, 5)

    Returns:
        A TensorLike with shape (batch_size, 18, 5)
    """
    # -------------
    # Debug Section
    # -------------
    if debug:
        print("-" * 60)
        print(f"NonMaxSuppression Function Start:")
        print(f"input shapes:\nlabels: {labels.shape}, {labels.dtype}\nanchors: {anchors.shape}, {anchors.dtype}")
    # Step through process of calculating the GIoU for each box in the anchors against each box in
    # the labels.
    intersection_points, label_corners, anchor_corners = construct_intersection_vertices(labels, anchors) 
    intersection = intersection_area(intersection_points)
    union = union_area(label_corners, anchor_corners, intersection, num_pumps=9)
    giou = calculate_giou(label_corners, anchor_corners, union, intersection)
 
    # Sort the indices to find the maximum GIoU for each anchor box label pair and the second highest
    # in case of a duplicate, i.e. a single anchor box is returned for more than one of the label
    # bounding boxes.
    sorted = tf.argsort(giou, axis=-1, direction='DESCENDING')
    max_boxes = sorted[..., 0:1]
    alt_boxes = sorted[..., 1:2]
    # -------------
    # Debug Section
    # -------------
    if debug:
        print(f"giou: {giou.shape}")
        print(f"sorted: {sorted.shape}")

  
    """
    This next block locates images where a single anchor box was returned as the maximum match for
    more than one object. It does this by checking each indices from sorted against each indices in
    the image and returning 1 if any of them match. 
    
    This ends with a tensor with a shape of (batch_size, 18, 1) where the final dimension is a 1 or 0
    indicating a conflicting anchor box selection. For each image the cumulative sum of these marked
    values is taken and then the first instance in each image removed by subtracting 1 from the
    cumulative sum tensor.

    This allows the final tensor to be gathered using the duplicate mask as a filter, returning
    the maximum value for the first duplicate anchor and the alternate (second highest) anchor match
    for subsequent matches. 

    TO DO:
        This approach is pretty hacky. If there are two pairs of duplicate anchor boxes in a single
        image, it will return the anchor with the maximum GIoU for the first but the second highest
        for all subsequent anchors.

        This could be non-optimal later. It would be better to compare GIoU scores for any double
        anchor assigments and return the lowest loss set of anchors for each conflicting pair or
        set.

        This seems to work practically with this dataset, but I guess we'll see.
    """
   
    # -------------
    # Debug Section
    # -------------
    if debug:
        # sorted = tf.reshape(sorted, labels.shape[:-1] + (1,))
        unique_boxes = tf.broadcast_to(tf.expand_dims(sorted, axis=-3), [anchors.shape[0], 18, 18, 972])[..., 0:1]
        print(f"unique_boxes: {unique_boxes.shape}")
        unique_boxes = tf.reduce_sum(tf.where(tf.equal(unique_boxes, tf.expand_dims(max_boxes, axis=-2)), 1, 0), axis=-2)
        print(f"unique_boxes: {unique_boxes.shape}")
        print(f"labels:       {labels.shape}")
        unique_boxes = tf.cast(unique_boxes > 1, dtype=tf.float32) * labels[..., 13:14]
        duplicate_mask = (tf.cumsum(unique_boxes, axis=-2) - unique_boxes) * unique_boxes

        pred_boxes = tf.where(duplicate_mask > 0,
                              tf.gather(anchors, tf.squeeze(max_boxes), axis=-2, batch_dims=-1),
                              tf.gather(anchors, tf.squeeze(alt_boxes), axis=-2, batch_dims=-1))
        result = pred_boxes * labels[..., 13:14]
        print(f"NonMaxSuppression Function End:")
        print(f"output: {result.shape}")
        print("-" * 60)
        return result

    # sorted = tf.reshape(sorted, labels.shape[:-1] + (1,))
    unique_boxes = tf.broadcast_to(tf.expand_dims(sorted, axis=-3), [anchors.shape[0], 18, 18, 972])[..., 0:1]
    unique_boxes = tf.reduce_sum(tf.where(tf.equal(unique_boxes, tf.expand_dims(max_boxes, axis=-2)), 1, 0), axis=-2)
    unique_boxes = tf.cast(unique_boxes > 1, dtype=tf.float32) * labels[..., 13:14]
    duplicate_mask = (tf.cumsum(unique_boxes, axis=-2) - unique_boxes) * unique_boxes

    pred_boxes = tf.where(duplicate_mask > 0,
                          tf.gather(anchors, tf.squeeze(max_boxes), axis=-2, batch_dims=-1),
                          tf.gather(anchors, tf.squeeze(alt_boxes), axis=-2, batch_dims=-1))

    return pred_boxes * labels[..., 13:14]
    
def calc_best_anchors(y_true:TensorLike, 
                      pred_bboxes:TensorLike) -> Tuple[TensorLike, TensorLike]:
    """
    Function that takes the ground truth boxes, calculates their cell numbers and assigns an
    anchor/detector to those objects.

    Returns:
        TensorLike object representing the detectors responsible for each labeled object and a 
        tensor representing the [cell, detector] addresses for each in the model output.
    """
    # ------------------------------
    # Find Indexes Cells with Objects
    # ------------------------------
    # print(f"y_true: {y_true.shape}")
    # print(f"y_pred: {pred_bboxes.shape}")
    true_xy = y_true[..., 14:16]
    grid_dims = tf.constant([[12, 9]], dtype=tf.float32)
    cell_num = tf.cast(tf.math.floor(true_xy * grid_dims), dtype=tf.int32)
    # ------------------------------
    # Populate a Tensor with the Anchors from those Cells
    # ------------------------------
    pred_idx = cell_num[..., 0:1] * 9 + cell_num[..., 1:]
    pred_anchors = tf.gather_nd(pred_bboxes, pred_idx, batch_dims=1)
    # print(f"pred_anchors: {pred_anchors.shape}")
    # print(f"y_true target: {tf.expand_dims(y_true, axis=-2).shape}")
    true_labels = tf.broadcast_to(tf.expand_dims(y_true, axis=-2), shape=pred_anchors.shape[:-1] + (19,))
    # ------------------------------
    # Calculate GIoU of the Ground Truth boxes against the Anchor Boxes
    # ------------------------------
    intersection_points, label_corners, anchor_corners = construct_intersection_vertices(true_labels, pred_anchors) 
    intersection = intersection_area(intersection_points)
    union = union_area(label_corners, anchor_corners, intersection, num_pumps=9)
    giou = calculate_giou(label_corners, anchor_corners, union, intersection)
    # ------------------------------
    # Populate a Tensor with only the Detectors that are Assigned to the Objects
    # ------------------------------
    best_idx = tf.argsort(giou, direction="DESCENDING", axis=-1)[..., 0:1]
    best_idx = tf.concat([pred_idx, best_idx], axis=-1)
    best_boxes = tf.gather_nd(pred_bboxes, best_idx, batch_dims=1)
    # ------------------------------
    # End
    # ------------------------------
    return best_boxes, best_idx
