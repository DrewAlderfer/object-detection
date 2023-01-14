from typing import Union, Tuple, List
from pathlib import Path
import pprint as pp
from numpy.typing import ArrayLike, NDArray
import tensorflow as tf
from sklearn.cluster import KMeans

from pycocotools.coco import COCO
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tensorflow.keras.utils import load_img


def generate_anchors(labels:NDArray[np.float32], boxes_per_cell:int=3, **kwargs):
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
    # flatten labels
    target_size = np.asarray((512, 384), dtype=np.float32)
    input_shape = labels.shape
    batch, xdivs, ydivs, _ = input_shape
    box_labels = tf.reshape(labels, [batch * xdivs * ydivs, 19])[..., 14:].numpy()
    box_labels[:, 0:1] = 0 
    box_labels[:, 1:2] = 0 
    # mask no object
    mask_labels = tf.cast(tf.reduce_sum(box_labels, axis=-1) > .001, dtype=tf.bool)
    box_labels = tf.boolean_mask(box_labels, mask_labels, axis=0).numpy()
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
    centroid_locations = np.ones((13,), dtype=np.float32)
    for idx in range(boxes_per_cell):
        filter = tf.where(tf.equal(cls, idx),
                          box_labels,
                          np.zeros(box_labels.shape, dtype=np.float32))

        mask = tf.cast(tf.abs(tf.reduce_sum(filter, axis=-1)) > .001, dtype=tf.float32)
        average = np.append(np.ones((1,), dtype=np.float32), tf.reduce_sum(filter, axis=0) / tf.reduce_sum(mask, axis=0))
        centroid_locations = np.append(centroid_locations, average)

    """ 
    idx_template: A template of the grid where the values are indices of cells in the grid.
    knudge_coords: A map of center-point coordinates for each cell in the grid.""" 
    idx_template = np.fromfunction(lambda x, y: [x, y], shape=(xdivs, ydivs), dtype=np.float32)   
    knudge_coords = (np.stack((idx_template[0], idx_template[1]), axis=-1) + .5)/ np.array([xdivs, ydivs])
    """
    Then we're creating the return value by filling an array with the shape of our grid with the
    anchor box values and then assigning the x, y point values for each set of anchor boxes in the
    array to the centerpoints we created in the knudge_coords array."""
    anchor_box_template = np.full((batch, xdivs, ydivs, 13 + boxes_per_cell * 6), centroid_locations)
    anchor_box_template[..., 14::6] = anchor_box_template[..., 14::6] + knudge_coords[..., 0:1]
    anchor_box_template[..., 15::6] = anchor_box_template[..., 15::6] + knudge_coords[..., 1:]

    return anchor_box_template

def rotate(input, phi, center:Union[tuple,np.ndarray]=(0, 0)):
    """
    Give it a vector, a center point and an angle in radians and this little guy will rotate that
    vector for you.
    """
    vector = np.subtract(input, center)
    id = np.asarray([[np.cos(phi), -np.sin(phi)],
                     [np.sin(phi), np.cos(phi)]], 
                    dtype=np.float32)
    vector = np.matmul(id, vector)
    return np.add(vector, center)


def make_masks(
    data: dict,
    input_size: tuple = (None, None),
    target_size: tuple = (None, None),
    draw_bbox: bool = True,
    draw_center: bool = False,
    get_crops:bool=False,
    save_mask: bool = False,
    file_name: Union[str, None] = None,
    path_to_dir: Union[str, None] = None,
    debug=False) -> Union[List[np.ndarray], np.ndarray]:
    """
    Function takes an image shape and a dictionary that defines a bounding box and
    returns a numpy array version of the image mask.
    """
    colors = ["red", "blue", "green", "purple"]
    pick = np.random.choice
    results = np.empty(target_size + (0,), dtype=np.float32)
    crops = []
    debug = {}
    for cat in range(1, 14, 1):
        mask = Image.new("L", (target_size[1], target_size[0]), color="black")
        first = True
        for entry in data:
            img_crops = np.empty((0, 2, 2), dtype=np.float32)
            if entry["category_id"] == cat:
                if first and debug:
                    debug.update({f"{cat}": {}})
                # pp.pprint(entry)
                draw = ImageDraw.Draw(mask, "L")

                bbox, crop_box, center_line = process_img_annotations(entry['bbox'])

                if first and debug:
                    debug[f"{cat}"]["bbox_01"] = bbox.copy()

                # draw mask shapes
                if draw_bbox:
                    for i, (x, y) in enumerate(bbox):
                        x = x * target_size[1] / input_size[1]
                        y = y * target_size[0] / input_size[0]
                        bbox[i] = (x, y)
                    draw.line(bbox, fill="white")

                if first and debug:
                    debug[f"{cat}"]["bbox_02"] = bbox.copy()

                if draw_center:
                    for i, (x, y) in enumerate(center_line):
                        x = x * target_size[1] / input_size[1]
                        y = y * target_size[0] / input_size[0]
                        center_line[i] = (x, y)
                    draw.line(center_line, fill="white", width=3)
                first = False
                img_crops = np.append(img_crops, crop_box.reshape((1, 2, 2)), axis=0)

            crops.append(img_crops)

        if save_mask:
            mask.save(f"{path_to_dir}/{cat:02d}_{file_name}", format="png")

        mask_data = np.asarray(mask, dtype=np.float32)
        results = np.append(results, mask_data.reshape(target_size + (1,)), axis=2)

    if debug:
        pp.pprint(debug)
    if get_crops:
        return crops
    return results

def convert_mask_to_class(stack:np.ndarray, debug=False):
    if debug:
        print(f"Input Shape: {stack.shape}")
    result = np.empty((0,) + stack.shape[1:-1] + (1,), dtype=np.float32)
    first = True
    for mask in stack:
        if debug:
            print(f"mask off stack shape: {mask.shape}")
        flat_mask = np.empty(mask.shape[:-1] + (0,), dtype=np.float32)
        for i in range(mask.shape[-1]):
            channel = mask[:,:,i]
            channel = channel / 255

            if first and debug:
                print(f"Channel Shape: {channel.shape}")
                print(f"Channel Max: {channel.max()}")
                print(f"Channel Min: {channel.min()}")

            channel = channel * (i + 1)
            flat_mask = np.append(flat_mask, channel.reshape(channel.shape + (1,)), axis=2)
            first = False

        if debug:
            print(f"Class Mask Shape before flattening:\n{flat_mask.shape}")
        flat_mask = np.amax(flat_mask, axis=2)
        if debug:
            print(f"Class Mask Shape after flattening:\n{flat_mask.shape}")
        result = np.append(result, flat_mask.reshape((1,) + flat_mask.shape + (1,)), axis=0)
        if debug:
            print(f"Result Shape: {result.shape}")

    return result

def init_COCO(json_path:str, divs:List[str]):

    result = {}
    for target in divs:
        file = Path(f"{json_path}/mvtec_screws_{target}.json")
        db = COCO(file)

        ids = db.getImgIds()
        imgs = db.loadImgs(ids)
        annIds = db.getAnnIds(ids)
        print(f"Found {len(imgs)} {target} images")
        result.update({target: {"coco": db,
                                "ids": ids,
                                "img_data": imgs,
                                "annotations": db.loadAnns(annIds)}
                       })
    return result

def process_img_annotations(entry:list):
    """
    process individual annotations from the MVTec Screws Dataset. Takes the 'bbox' list and
    returns the adjusted coordinates of the bounding box as a tuple with three lists of points.

    returns:
        boundingbox_corners, crop_corners(top right, and left points), center_line([x1, y1], [x2, y2])
    """
    # grab bbox info
    col, row, height, width, phi = entry
    # define center point of bbox
    center = np.array([col, row])
    # -pi to pi -> 0 to 2*pi
    phi = -1 * (phi - np.pi)
    # initial bounds
    y0, y1 = row - height / 2, row + height / 2
    x0, x1 = col - width / 2, col + width / 2

    # corner points
    corners = [(x0, y0), (x0, y1), (x1, y1), (x1, y0)] # Corners
    ref_points = [(x0, row), (x1, row)]  # Center Line
    # rotate_box([p0, p1, p2, p3], center, phi)
    bbox = [rotate(point, phi, center) for point in corners]
    crop_box = np.array([bbox[0], bbox[2]])
    center_line = [rotate(point, phi, center) for point in ref_points]

    return bbox, crop_box, center_line


def format_layers_for_display(activations:np.ndarray, layer_names:list, images_per_row:int=16):

    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros(((size + 1) * n_cols - 1,
                                images_per_row * (size + 1) - 1))
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_index = col * images_per_row + row
                channel_image = layer_activation[0, :, :, channel_index].copy() 

                if channel_image.sum() != 0:
                    channel_image -= channel_image.mean()
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype("uint8")
                display_grid[col * (size + 1): (col + 1) * size + col,
                             row * (size + 1) : (row + 1) * size + row] = channel_image
        scale = 1. / size
        yield display_grid, scale, layer_name

# def display_convolution_layers(activations:Union[ArrayLike, list], layer_names:list, **kwargs):
#     
#     for display_grid, scale, layer_name in format_layers_for_display(activations, layer_names):
#         
#         fig, ax = (figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
#
#         plt.title(layer_name)
#         plt.grid(False)
#         plt.axis("off")

#         plt.imshow(display_grid, aspect="auto", cmap="viridis")

def make_predictions(labels, divs:tuple=(9, 12)):
    ydivs, xdivs = divs
    # Initialize an array with zeros    
    # The shape is (batch_size, x_division, y_division, 13(classes) + 3(boxes) * 6(x, y, width, height, angle))
    preds = np.zeros(labels.shape[:-1] + (13 + 3 * 6,), dtype=np.float32)

    # add our label tensor to it; the model predicts three anchor boxes for each grid cell so we create three predictions
    preds[...,:19] = preds[...,:19] + labels
    preds[..., 19:25] = labels[..., 13:19] 
    preds[..., 25:] = labels[..., 13:19]
    # b_e means "box exists" it is the probability score that a box is present in the cell.
    # we can also use this value (here is is always equal to the true value) to reduce rows we that do not contain values to zero
    b_e = preds[..., 13]
    b_e = b_e.reshape((preds.shape[:-1] + (1,)))
    box_exists = np.concatenate((b_e, b_e, b_e, b_e, b_e), axis=-1)
    # establishing a known loss for the first anchor box prediction so that I can follow the calculations for a sanity check
    # if it becomes necessary
    known_loss = np.asarray([10, 10, 5, 20, np.pi * .05])
    known_loss = np.full(preds.shape[:-1] + (5,), known_loss, dtype=np.float32)

    # generating the loss values and adding them to the prediction tensor
    preds[..., 14:19] = known_loss + preds[..., 14:19]
    preds[..., 20:25] = box_exists * (preds[..., 20:25] + np.random.normal(0, 2, preds.shape[:-1] + (5,)))
    preds[..., 26:] = box_exists * (preds[..., 26:] + np.random.normal(0, 2, preds.shape[:-1] + (5,)))

    return preds

def translate_points(entry:list, input_size:Tuple[int, int], target_size:Tuple[int, int]):
    """
    Functions: 
        Process individual annotations from the MVTec Screws Dataset. Takes the 'bbox' list and
        returns the adjusted coordinates of the bounding box as a tuple with three lists of points.

    Returns:
        Bounding box vector translated into the target space.
    """
    # grab bbox info
    row, col, width, height, phi = entry
    # initial bounds
    col = col * target_size[1] / input_size[1]
    row = row * target_size[0] / input_size[0]
    width = width * target_size[1] / input_size[1]
    height = height * target_size[0] / input_size[0]
    phi = phi
    return row, col, width, height, phi

def display_label(label, ax, color, input_size=(1440, 1920), target_size=(384, 512)):
    """
    Function:
        Takes a Label [x, y, w, h, angle] and an Axis object, with the image (h, w) and the plot (h, w).

    Returns:
    
        A tuple of mpl.axis objects. One contains the bounding box and the other contains the an 
        arrow showing the orientation and directoin of the object.
    """
    from matplotlib.patches import Arrow, Polygon
    x, y, w, h, phi = label
    phi = phi + np.pi / 2
    x, y = x * target_size[1], y * target_size[0]
    w, h = w * target_size[1], h * target_size[1]
    # vec_x_mag, vec_y_mag = ((h / 2) * np.cos(phi)), ((h / 2) * np.sin(phi))
    vec_x_mag, vec_y_mag = ((w / 2) * np.sin(phi)), ((w / 2) * np.cos(phi))

    label_tensor = np.ones((1, 19), dtype=np.float32)
    t_label = label # translate_points(label.tolist(), input_size, target_size)
    label_tensor[..., 14:] = label_tensor[..., 14:] * t_label
    test_box_corners = np.squeeze(get_corners(label_tensor), axis=0)

    bbox = Polygon(test_box_corners, fill=None, edgecolor=color, lw=1, zorder=20)
    arrow = Arrow(x, y, vec_x_mag, vec_y_mag, width=15, color=color, zorder=100)

    return bbox, arrow

def get_corners(box_vectors:NDArray[np.float32],
                img_width:int=512,
                img_height:int=384) -> NDArray[np.float32]:
    """
    Function that takes a tensor of box label vectors and returns the corners of the bounding box
    described by the vector [x, y, w, h, phi].
    """
    rank = len(box_vectors.shape)
    if box_vectors.shape[-1] > 5:
        box_vectors = box_vectors[..., 14:19]
    if rank > 2:
        box_vectors = box_vectors.reshape(box_vectors.shape[0], np.cumprod(box_vectors.shape[1:-1])[-1], box_vectors.shape[-1])
        x = np.zeros(box_vectors.shape[0:2] + (4,), dtype=np.float32) 
        y = np.zeros(box_vectors.shape[0:2] + (4,), dtype=np.float32) 
    else:
        x = np.zeros((box_vectors.shape[0],) + (4,), dtype=np.float32) 
        y = np.zeros((box_vectors.shape[0],) + (4,), dtype=np.float32) 
    cx, cy, = box_vectors[..., 0:1], box_vectors[..., 1:2]
    phi = box_vectors[..., -1:]
     
    # give width and height their own tensors to make it easier to write the math
    w = tf.expand_dims(box_vectors[...,2], axis=-1)
    h = tf.expand_dims(box_vectors[...,3], axis=-1)
    # Set x / y coordinates based on the center location and the width/height of the box
    x[..., 0:2] = tf.add(cx, w/2)
    x[..., 2:] = tf.subtract(cx, w/2)
    y[..., 0::3] = tf.add(cy, h/2)
    y[..., 1:-1] = tf.subtract(cy, h/2)
    # Subtract the center point from each point in the corners. This centers the points around zero
    # so that we can rotate them around their local center point.
    scale = np.array([img_width, img_height])
    corner_points = np.stack([x, y], axis=-1) * scale
    center_point = np.stack([cx, cy], axis=-1) * scale
    centered = np.subtract(corner_points, center_point) 

    # Set up the rotation matrix tensor
    cos, sin = np.cos(phi), np.sin(phi)
    R = np.squeeze(np.row_stack(np.array([[[cos, -sin],
                                           [sin, cos]]])), axis=-1)
    # Take a shape like: (a, a, b, c) or (a, a, b)
    # transpose it into: (b, c, a, a) or (b, a, a)
    r_idx = tuple(range(len(R.shape)))
    R = np.transpose(R, r_idx[2:] + r_idx[0:2])
    # Return the points with the rotation applied and then the center point added back in.
    return np.add(np.matmul(centered, R), center_point)

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
    if len(anchors.shape) > 3:
        batch, xdivs, ydivs, units = anchors.shape
        anchors = anchors.reshape((batch, xdivs * ydivs,) + (units,))
    else:
        xdivs, ydivs, units = anchors.shape
        anchors = anchors.reshape((xdivs * ydivs,) + (units,))
    anchor_box1 = anchors[..., 14:19]
    anchor_box2 = anchors[..., 20:25]
    anchor_box3 = anchors[..., 26:]

    return np.concatenate([anchor_box1, anchor_box2, anchor_box3], axis=-2)

def calulate_grid_cell(label_vectors, grid_dim):
    x, y = label_vectors[..., 14], label_vectors[..., 15]
    x_cell = np.floor(x * grid_dim[0])
    y_cell = np.floor(y * grid_dim[1])

    return x_cell, y_cell
    
def get_edges(bbox_tensor:Union[tf.Tensor, NDArray]):
    z = tf.roll(bbox_tensor, shift=-1, axis=-2)
    t = tf.stack([bbox_tensor, z], axis=-2)
    return t

def rolling_intersection(box1, box2):
    """
    Calculates the points of edge intersection between two parallelograms by roling one tensor of
    corner points over the other to create edges.
    """
    # box1_edges = get_edges(box1)
    # box2_edges = get_edges(box2) 
    box1_edges = box1
    box2_edges = box2
    edge_intersections = None
    for i in range(4):
        if i != 0:
            edge_intersections = tf.concat([edge_intersections,
                                            get_intersections(box1_edges, box2_edges)], 
                                            axis=-2)
            box1_edges = tf.roll(box1_edges, shift=-1, axis=-3)
            continue
        edge_intersections = get_intersections(box1_edges, box2_edges)
        box1_edges = tf.roll(box1_edges, shift=-1, axis=-3)
    return edge_intersections, box1_edges, box2_edges

def get_intersections(box1_edge, box2_edge):
    """
    Takes two edges and returns the point of intersection between them if one exists.
    """
    # print(f"input1: {box1_edge.shape}")
    # print(f"input2: {box2_edge.shape}")
    edge_a = box1_edge[..., 0:1, :, :]
    edge_b = box2_edge[..., :108, :, :, :]
    # print(f"edge_a: {edge_a.shape}")
    # print(f"edge_b: {edge_b.shape}")

    x1 = edge_a[..., 0:1, 0:1]
    y1 = edge_a[..., 0:1, 1:]
    x2 = edge_a[..., 1:, 0:1]
    y2 = edge_a[..., 1:, 1:]

    x3 = edge_b[..., 0:1, 0:1]
    y3 = edge_b[..., 0:1, 1:]
    x4 = edge_b[..., 1:, 0:1]
    y4 = edge_b[..., 1:, 1:]

    # print(f"x1, y1: {x1[0, 0, 0, 0, 0, 0]}, {y1[0, 0, 0, 0, 0, 0]}")
    # print(f"x2, y2: {x2[0, 0, 0, 0, 0, 0]}, {y2[0, 0, 0, 0, 0, 0]}")
    # print(f"box:\n{edge_b[0, 0, 0]}")
    # print(f"y1: {y1.shape}")
    # print(f"x2: {x2.shape}")
    # print(f"y2: {y2.shape}")
    # print(f"x3: {x3[0, 0, 0]}")
    # print(f"y3: {y3.shape}")
    # print(f"x4: {x4.shape}")
    # print(f"y4: {y4.shape}")
   
    # this is kind of like taking the area of a plane created between the two edges and then
    # subtracting them. If it equals zero then the edges are colinear.
    denom =  (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)

    # These check to see if the point of intersection falls along the line segments of the edge
    # ua and ub have a domain of 0 to 1 if the point is a valid intersection along the segments.
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
    mask_a = tf.cast(tf.math.logical_and(tf.greater(ua, 0), tf.less(ua, 1)), dtype=tf.bool)
    mask_b = tf.cast(tf.math.logical_and(tf.greater(ub, 0), tf.less(ub, 1)), dtype=tf.bool)
    mask = tf.logical_and(mask_a, mask_b)
    print(f"mask:\n{mask[0, 2, 3::9]}")
    # zeros = tf.fill(tf.shape(ua), 0.0)
    # ones = tf.fill(tf.shape(ua), 1.0)
    print(f"ua:\n {ua[0, 2, 3::9]}")
    print(f"ub:\n {ub[0, 2, 3::9]}")

    # hi_pass_a = tf.math.less_equal(ua, ones)
    # lo_pass_a = tf.math.greater_equal(ua, zeros)
    # mask_a = tf.logical_and(hi_pass_a, lo_pass_a)

    # ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
    # hi_pass_b = tf.math.less_equal(ub, ones)
    # lo_pass_b = tf.math.greater_equal(ub, zeros)
    # mask_b = tf.logical_and(hi_pass_b, lo_pass_b)

    # mask = tf.logical_and(mask_a, mask_b)

    # This actually just says where the intersection is 
    xnum = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
    ynum = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)

    x_i = np.asarray((xnum / denom), dtype=np.float32)
    y_i = np.asarray((ynum / denom), dtype=np.float32)
    print(f"x_i:{type(x_i)} {x_i[0,0,0]}") 
    print(f"y_i:{type(y_i)} {y_i[0,0,0]}") 
    # here I am using the mask to multiply any intersections that didn't fall along the segments
    # by zero and all the ones that did by one.
    mask = tf.cast(tf.squeeze(mask, axis=[-1]), dtype=tf.float32)
    # print(f"mask: {type(mask)} {mask.shape}")
    # print(f"squeeze: {tf.squeeze(tf.stack([x_i, y_i], axis=-3), axis=[-2, -1]).shape}")
    intersections = tf.multiply(tf.squeeze(tf.stack([x_i, y_i], axis=-3), axis=[-2, -1]), mask)

    return intersections
