from typing import Union, Tuple, List
from pathlib import Path
import pprint as pp
from numpy.typing import ArrayLike

from pycocotools.coco import COCO
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tensorflow.keras.utils import load_img

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
                print(crop_box.shape)
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
