from typing import Union, Tuple, List
from pathlib import Path
import pprint as pp
from numpy.typing import ArrayLike, NDArray
from pandas.core.window import rolling
import tensorflow as tf
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
    target_size = np.asarray((512, 384), dtype=np.float32)
    input_shape = labels.shape
    batch, max_obj, _ = input_shape
    box_labels = tf.reshape(labels, [batch * max_obj, 19])[..., 14:].numpy()
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
    corner_points = tf.stack([x, y], axis=-1) * scale
    center_point = tf.stack([cx, cy], axis=-1) * scale
    centered = tf.subtract(corner_points, center_point) 

    # Set up the rotation matrix tensor
    cos, sin = np.cos(phi), np.sin(phi)
    R = np.squeeze(np.row_stack(np.array([[[cos, -sin],
                                           [sin, cos]]])), axis=-1)
    # Take a shape like: (a, a, b, c) or (a, a, b)
    # transpose it into: (b, c, a, a) or (b, a, a)
    r_idx = tuple(range(len(R.shape)))
    R = np.transpose(R, r_idx[2:] + r_idx[0:2])
    # Return the points with the rotation applied and then the center point added back in.
    return tf.add(tf.matmul(centered, R), center_point)

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
        batch = 1
        xdivs, ydivs, units = anchors.shape
        anchors = anchors.reshape((xdivs * ydivs,) + (units,))
    size = xdivs * ydivs
    num_boxes = int((anchors.shape[-1] - 13) / 6)
    result = np.reshape(anchors[..., 13:], (batch, size, num_boxes * 6))
    result = np.expand_dims(result.reshape(batch, size, num_boxes, 6), axis=-2)
    result = np.reshape(np.transpose(np.full(result.shape[:-2] + (3, result.shape[-1]), result), (0, 2, 1, 3, 4))[..., 0, 1:], (batch, size * num_boxes, 5))

    return result

def calulate_grid_cell(label_vectors, grid_dim):
    x, y = label_vectors[..., 14], label_vectors[..., 15]
    x_cell = np.floor(x * grid_dim[0])
    y_cell = np.floor(y * grid_dim[1])

    return x_cell, y_cell
    
def get_edges(bbox_tensor:Union[tf.Tensor, NDArray]):
    z = tf.roll(bbox_tensor, shift=-1, axis=-2)
    return tf.stack([bbox_tensor, z], axis=-2)

def stretch_tensor(box_edges):
    return tf.expand_dims(box_edges, axis=1)

def pump_tensor(box_edges, num_cells:int=108, num_pumps:int=3):
    """
    This function takes a bounding box edge tensor (h, grid_x * grid_y, 4, 2) and returns a 
    'pumped-up' version of it to be 'rolled' against another tensor for calculating intersections.
    
    Returns:
        NDArray with shape (B, M, 4, 2, 2)
    """
    batch, num_boxes = box_edges.shape[0:2]
    end = box_edges.shape[2:]
    input = box_edges
    # for _ in range(num_pumps - 1):
    #     box_edges = tf.concat([box_edges, input], axis=1)
    step1 = np.expand_dims(box_edges, axis=2)
    step2 = np.full((batch, num_boxes, num_cells * num_pumps) + end, step1)
    return step2

def rolling_intersection(box1, box2):
    """
    DEPRECATED:
    Calculates the points of edge intersection between two parallelograms by roling one tensor of
    corner points over the other to create edges.
    """
    box1_edges = pump_tensor(box1, **kwargs)
    box2_edges = stretch_tensor(box2)
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
    return edge_intersections

def find_intersection_points(box1, box2, **kwargs):
    """
    Parameters:
        Takes two tensors with the edges of label and anchor bounding boxes.

    Returns: 
        a tensor containing the intersection points of each label bounding box with each anchor
        box.
    """
    box1_edges = pump_tensor(box1, **kwargs)
    box2_edges = stretch_tensor(box2)
    batch, box_num, anchor_num = box1_edges.shape[:3]
    # box1_edges = tf.constant(box1_edges.shape[:4] + (4,) + box1_edges.shape[-2:], tf.expand_dims(box1_edges, axis=-3))
    box1_edges = tf.broadcast_to(tf.expand_dims(box1_edges, axis=-3), shape=box1_edges.shape[:4] + (4,) + box1_edges.shape[-2:])
    box2_edges = tf.expand_dims(box2_edges, axis=-4)

    return tf.reshape(get_intersections(box1_edges, box2_edges), [batch, box_num, anchor_num, 16, 2])

def get_intersections(box1_edge, box2_edge):
    """
    calculates the edge intersections between two sets of bounding boxes. 

    Returns:
        tf.Tensor with a list of intersections (or zeros for no intersection) between each edge, of
        the two boxes.
    """
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
    # Finish by using the mask to multiply any intersections that didn't fall along the segments
    # by zero and all the ones that did by one.
    return tf.multiply(tf.squeeze(tf.stack([x_i, y_i], axis=-3), axis=[-2, -1]), mask)

def construct_intersection_vertices(labels, anchors, xdivs:int=12, ydivs:int=9, **kwargs):
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
    pumps = anchors.shape[1] // (xdivs * ydivs)
    label_corners, anchor_corners = get_corners(labels), get_corners(anchors)
    label_edges, anchor_edges = get_edges(label_corners), get_edges(anchor_corners)
    inner_points = find_inner_points([label_corners, label_edges], [anchor_corners, anchor_edges], num_pumps=pumps)
    box_exists = tf.reshape(labels[..., 13:14], labels.shape[:-1] + (1, 1, 1))
    print(f"box_exists:   {box_exists.shape}")
    print(f"inner_points: {inner_points.shape}")
    inner_points = inner_points * box_exists
    print(f"inner_points: {inner_points[0, 16, 0]}")

    intersection_points = find_intersection_points(label_edges, anchor_edges, **kwargs)
    return  tf.concat([intersection_points, inner_points], axis=-2)

def find_inner_points(label:list, anchors:list, **kwargs):
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
    label_corners, label_edges = label[0], label[1]
    label_corners = pump_tensor(label_corners, **kwargs)
    label_corners = tf.broadcast_to(tf.expand_dims(label_corners, axis=-2), 
                                    shape=label_corners.shape[:-1] + (4, 2))
    label_edges = pump_tensor(label_edges, **kwargs)
    label_edges = tf.expand_dims(label_edges, axis=-4)

    anchor_corners, anchor_edges = anchors[0], anchors[1]
    anchor_edges = tf.expand_dims(anchor_edges, axis=1)
    anchor_edges = tf.expand_dims(anchor_edges, axis=-4)
    anchor_corners = tf.expand_dims(anchor_corners, axis=1)
    anchor_corners = tf.broadcast_to(tf.expand_dims(anchor_corners, axis=-2),
                                     shape=anchor_corners.shape[:-1] + (4, 2))

    return tf.concat([is_inside(label_corners, anchor_edges), 
                      is_inside(anchor_corners, label_edges)], axis=-2)

def is_inside(corners, box_edges):
    """
    Determines whether a point falls inside the edges of a convex shape by determining which side
    of each edge the point falls along moving clockwise along the edges.
    
    Returns:
        tf.Tensor object containing points that are fall inside the shape, and zeros out all points
        that do not.
    """
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
    left = tf.greater(dir, 0)
    right = tf.less(dir, 0)

    # This returns 1 if the point is on the same side of all the edges; 0 if it is not.
    check = tf.cast(tf.reduce_all(tf.logical_not(left, right), axis=-2), dtype=tf.float32)
    # Zero out the points not inside the shape.
    return tf.cast(tf.squeeze(corners[..., 0:1, :], axis=-2), dtype=tf.float32) * check

def intersection_area(intersection_points):
    """
    Function that takes a set of intersection points and returns a set of the areas.
    """
    nonzero = tf.cast(intersection_points > 0, dtype=tf.float32)
    mask = tf.cast(intersection_points > 0, dtype=tf.bool)

    denomenator = tf.reduce_sum(nonzero, axis=-2)
    center = tf.expand_dims(tf.reduce_sum(intersection_points, axis=-2) / denomenator, axis=-2)
    center_adj_points = tf.where(mask, 
                                 intersection_points - center,
                                 nonzero)
    # Get the angles of each point in the intersection to the center point of the intersection shape.
    point_angle = tf.where(mask, tf.math.atan2(center_adj_points[..., :, 1:],
                                               center_adj_points[..., :, 0:1]) + np.pi,
                           nonzero)
    # order the points by their angle. The transpose here was necessary reordering the points. So
    # for a few steps the x and y values are not paired in the last axis of the tensor but remain
    # linked by their shared index in the ordering.
    point_indices = tf.argsort(tf.transpose(point_angle, perm=[0, 1, 2, 4, 3]), direction="DESCENDING", axis=-1)
    # just transposing the points to match the ordered indexes
    points_T = tf.transpose(intersection_points, [0, 1, 2, 4, 3])
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
    edge_points = tf.transpose(tf.gather(points_T, point_order, batch_dims=-1)[..., :16], perm=[0, 1, 2, 4, 3])
    # this rolls the non-zero values one step. This sets up the tensor to be easily split into the
    # outer edge of the triangle making up the intersection shape.
    edge_points = tf.where(edge_points > 0, tf.roll(edge_points, shift=-1, axis=-2), 0)
    """
    The next few steps function as follows:
        - Find the index of the first [0, 0] point in each set of intersection points.
            - after rolling the point axis of the tensor this index is the last point of our edges
        - create a tensor that just represents the indexes for each x, y value in our tensor
        - create a tensor that is just the first point of every x, y set.
        - compare the last point index with the full index set and where they match insert the first
          point from each set.

    Now we have a full ordered set of edges for each intersection. These will be combined with the
    center point from before to create a set of traingles from which we can calculate the area of
    the intersection.
    """
    last_point = tf.expand_dims(tf.math.count_nonzero(tf.reduce_sum(edge_points, axis=-1), keepdims=True, axis=-1, dtype=tf.int32), axis=-1)
    idx = tf.transpose(tf.broadcast_to(np.arange(16, dtype=np.int32), last_point.shape[:-1] + (16,)), perm=[0, 1, 2, 4, 3])
    first_point = tf.squeeze(tf.gather(edge_points, last_point - last_point, axis=-2, batch_dims=-2), axis=-2)
    outer_edges = tf.reshape(tf.where(idx != last_point, edge_points, first_point), edge_points.shape[:-2] + (8, 2, 2))
    # Reshape the center point of each intersection to match the outer edges 
    center = tf.broadcast_to(tf.expand_dims(center, axis=-2), outer_edges.shape[:-2] + (1, 2))
    # Insert the center point into each edge to form the triangles
    triangle_points = tf.concat([outer_edges, center], axis=-2)
    # create a tensor filled with ones that will be concatenated to the last axis of the triangles.
    ones = tf.ones(triangle_points.shape[:-1] + (1,), dtype=tf.float32)
    # find the area of each triangle using the determinant and then sum them to find the area of
    # intersection.
    return tf.reduce_sum(tf.abs(tf.linalg.det(tf.concat([triangle_points, ones], axis=-1))), axis=-1) / 2

def union_area(box1, box2, intersection, **kwargs):
    box1 = pump_tensor(box1, **kwargs)
    box2 = stretch_tensor(box2)
    box1_points, box2_points = box1[..., :3, :], box2[..., :3, :]
    ones1 = tf.ones(box1_points.shape[:-1] + (1,), dtype=tf.float32)
    ones2 = tf.ones(box2_points.shape[:-1] + (1,), dtype=tf.float32)
    
    area1 = tf.abs(tf.linalg.det(tf.concat([box1_points, ones1], axis=-1)))
    area2 = tf.abs(tf.linalg.det(tf.concat([box2_points, ones2], axis=-1)))

    return (area1 + area2) - intersection
