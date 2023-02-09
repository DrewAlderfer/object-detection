import os
from pathlib import Path
from typing import List, Union, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pycocotools.coco import COCO
import tensorflow as tf
from tensorflow.keras.utils import load_img

from .old.old import make_masks, process_img_annotations, rotate


class LabelWorker:
    def __init__(self, 
                 data_name:str,
                 coco_obj:dict,
                 image_path:str,
                 input_size:Union[Tuple[int, int], NDArray, ArrayLike],
                 target_size:Union[Tuple[int, int], NDArray, ArrayLike]):

        self.name = data_name
        self.lookup = coco_obj[self.name]
        self.X_path = image_path 
        self.input_size = input_size
        self.target_size = target_size

    def get_masks(self, sample_size:Union[int, None]=None, **kwargs):
        if sample_size:
            try:
                img_data = self.lookup['img_data'][:sample_size]
            except IndexError:
                print(f"That index is out of bounds for your data! Returning the entire set instead.")
                img_data = self.lookup['img_data']
        else:
            img_data = self.lookup['img_data']

        img_ids = [(x['id'], x['file_name']) for x in img_data]
        coco = self.lookup['coco']
        images = np.empty((0,), dtype=np.float32)
        masks = np.empty((0,), dtype=np.float32)

        for id, file_name in img_ids:
            annot = coco.getAnnIds(id)
            annot = coco.loadAnns(annot)
            img = np.asarray(load_img(f"{self.X_path}/{self.name}/{file_name}", target_size=self.target_size), dtype=np.float32)
            mask = make_masks(annot, input_size=self.input_size, target_size=self.target_size, **kwargs)
            if images.shape == (0,):
                images = np.empty(((0,) + img.shape), dtype=np.float32)
                masks = np.empty(((0,) + mask.shape), dtype=np.float32)
            images = np.append(images, img.reshape((1,) + img.shape), axis=0)
            masks = np.append(masks, mask.reshape((1,) + mask.shape), axis=0)

        return images, masks

    def annot_to_tensor(self, 
                        xdivs:int=12,
                        ydivs:int=9,
                        num_classes:int=13,
                        sample_size:Union[int, None]=None):

        if sample_size:
            try:
                img_data = self.lookup['img_data'][:sample_size]
            except IndexError:
                print(f"That index is out of bounds for your data! Returning the entire set instead.")
                img_data = self.lookup['img_data']
        else:
            img_data = self.lookup['img_data']

        img_ids = [(x['id'], x['file_name']) for x in img_data]
        coco = self.lookup['coco']
        labels = np.empty((0, 18, num_classes + 6), dtype=np.float32)
        for id, file_name in img_ids:
            # print(f"searching image: {file_name}")
            annot = coco.getAnnIds(id)
            annot = coco.loadAnns(annot)
            # [1, 2, 3, ... 13, Pc, x1, y1, w, h, phi]
            img_labels = np.zeros((0, num_classes + 6), dtype=np.float32) 
            for entry in annot:
                y1, x1, w, h, phi = self.translate_points(entry['bbox'])

                cat = entry['category_id']
                label_vec = np.zeros((1, num_classes + 6), dtype=np.float32)
                label_vec[:, 13:] = 1, x1, y1, w, h, phi
                label_vec[:, cat] = 1 
                img_labels = np.concatenate((img_labels, label_vec), axis=0)

            fill = np.zeros((labels.shape[1] - img_labels.shape[0],) + (num_classes + 6,), dtype=np.float32)
            img_labels = np.expand_dims(np.concatenate((img_labels, fill), axis=0), axis=0)

            labels = np.concatenate((labels, img_labels), axis=0)

        return labels.reshape((labels.shape[0],) + (18, num_classes + 6))

    def annot_to_tuple(self, 
                       xdivs:int=12,
                       ydivs:int=9,
                       num_classes:int=13,
                       sample_size:Union[int, None]=None):

        if sample_size:
            try:
                img_data = self.lookup['img_data'][:sample_size]
            except IndexError:
                print(f"That index is out of bounds for your data! Returning the entire set instead.")
                img_data = self.lookup['img_data']
        else:
            img_data = self.lookup['img_data']

        img_ids = [(x['id'], x['file_name']) for x in img_data]
        coco = self.lookup['coco']
        output = []
        for id, file_name in img_ids:
            # print(f"searching image: {file_name}")
            annot = coco.getAnnIds(id)
            output.append(annot)
            # annot = coco.loadAnns(annot)
            # # [1, 2, 3, ... 13, Pc, x1, y1, w, h, phi]
            # img_labels = np.zeros((0, num_classes + 6), dtype=np.float32) 
            # for entry in annot:
            #     y1, x1, w, h, phi = self.translate_points(entry['bbox'])
            #
            #     cat = entry['category_id']
            #     label_vec = np.zeros((1, num_classes + 6), dtype=np.float32)
            #     label_vec[:, 13:] = 1, x1, y1, w, h, phi
            #     label_vec[:, cat] = 1 
            #     img_labels = np.concatenate((img_labels, label_vec), axis=0)
            #
            # fill = np.zeros((labels.shape[1] - img_labels.shape[0],) + (num_classes + 6,), dtype=np.float32)
            # img_labels = np.expand_dims(np.concatenate((img_labels, fill), axis=0), axis=0)
            #
            # labels.append(img_labels)

        return tuple(output)

    def translate_points(self, entry:list):
        """
        process individual annotations from the MVTec Screws Dataset. Takes the 'bbox' list and
        returns the adjusted coordinates of the bounding box as a tuple with three lists of points.

        returns:
            boundingbox_corners, crop_corners(top right, and left points), center_line([x1, y1], [x2, y2])
        """
        # grab bbox info
        row, col, width, height, phi = entry
        if phi > np.pi:
            phi = phi - (2 * np.pi)
        # -pi to pi -> 0 to 2*pi
        # initial bounds
        col = col / self.input_size[1] # * self.target_size[1] / self.input_size[1]
        row = row / self.input_size[0]# * self.target_size[0] / self.input_size[0]
        width = width / self.input_size[1] # * self.target_size[1] / self.input_size[1]
        height = height / self.input_size[0] # * self.target_size[0] / self.input_size[0]
        phi = (phi + np.pi) / (2 * np.pi)
        return row, col, width, height, phi

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
