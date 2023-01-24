import numpy as np
from PIL import Image, ImageDraw 
from typing import Union, List, Tuple

import tensorflow as tf

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

def calulate_grid_cell(label_vectors, grid_dim):
    x, y = label_vectors[..., 14], label_vectors[..., 15]
    x_cell = np.floor(x * grid_dim[0])
    y_cell = np.floor(y * grid_dim[1])

    return x_cell, y_cell

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
