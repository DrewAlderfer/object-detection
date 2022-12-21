from typing import Union, Tuple, List
from pathlib import Path

from pycocotools.coco import COCO
import matplotlib as plt
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
    save_mask: bool = False,
    file_name: Union[str, None] = None,
    path_to_dir: Union[str, None] = None) -> np.ndarray:
    """
    Function takes an image shape and a dictionary that defines a bounding box and
    returns a numpy array version of the image mask.
    """
    colors = ["red", "blue", "green", "purple"]
    pick = np.random.choice
    results = np.empty(target_size + (0,), dtype=np.float32)
    for cat in range(1, 14, 1):
        mask = Image.new("L", target_size, color="black")
        for entry in data:
            if entry["category_id"] == cat:
                draw = ImageDraw.Draw(mask, "L")
                # grab bbox info
                row, col, width, height, phi = entry["bbox"]
                # define center point of bbox
                center = np.array([col, row])
                # -pi to pi -> 0 to 2*pi
                phi = phi + np.pi
                # initial bounds
                y0, y1 = row - height / 2, row + height / 2
                x0, x1 = col - width / 2, col + width / 2

                # corner points
                # corners = [(x0, y0), (x0, y1), (x1, y1), (x1, y0)] # Corners
                ref_points = [(x0, y0), (x1, y1)]  # Center Line
                # rotate_box([p0, p1, p2, p3], center, phi)
                bbox = [rotate(point, center, phi) for point in ref_points]
                input_size = input_size
                target_size = target_size
                for i, (x, y) in enumerate(bbox):
                    x = x * target_size[1] / input_size[1]
                    y = y * target_size[0] / input_size[0]
                    bbox[i] = (x, y)

                # draw mask shapes
                if draw_bbox:
                    draw.polygon(ref_points, fill="white")
                if draw_center:
                    draw.line(ref_points, fill="white")

        if save_mask:
            mask.save(f"{path_to_dir}/{cat:02d}_{file_name}", format="png")
        mask_data = np.asarray(mask, dtype=np.float32)
        results = np.append(results, mask_data.reshape(target_size + (1,)), axis=2)

    return results

def init_COCO(divs:List[str]):
    result = {}
    for target in divs:
        file = Path(f"/content/data/mvtec_screws_{target}.json")
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
