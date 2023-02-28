import os
from glob import glob
from pathlib import Path
from typing import List, Union, Tuple

import math
import numpy as np
from numpy.typing import ArrayLike, NDArray
from pycocotools.coco import COCO
import tensorflow as tf
from tensorflow.keras.utils import load_img, Sequence


from .utils import *
from .old.old import make_masks, process_img_annotations, rotate


class YOLODataset(Sequence):
    def __init__(self, 
                 data_name:str,
                 coco_obj:dict,
                 image_path:str,
                 input_size:Tuple[int, int],
                 target_size:Tuple[int, int],
                 xy_grid:Tuple[int, int]=(12, 9),
                 num_classes:int=13,
                 batch_size:int=16):

        self.name = data_name
        self.lookup = coco_obj[self.name]
        self.x_path = image_path 
        self.input_size = input_size
        self.target_size = target_size
        self.batch_size = batch_size
        self.xy_grid = xy_grid
        self.num_classes = num_classes
        self.x = self.get_image_set()
        self.y = self.annot_to_tensor()
        self.epoch_mod = 0

    def annot_to_tensor(self, 
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
        labels = np.empty((0, 18, self.num_classes + 6), dtype=np.float32)
        for id, file_name in img_ids:
            # print(f"searching image: {file_name}")
            annot = coco.getAnnIds(id)
            annot = coco.loadAnns(annot)
            # [1, 2, 3, ... 13, Pc, x1, y1, w, h, phi]
            img_labels = np.zeros((0, self.num_classes + 6), dtype=np.float32) 
            for entry in annot:
                y1, x1, w, h, phi = self.translate_points(entry['bbox'])

                cat = entry['category_id']
                label_vec = np.zeros((1, self.num_classes + 6), dtype=np.float32)
                label_vec[:, 13:] = 1, x1, y1, w, h, phi
                label_vec[:, cat] = 1 
                img_labels = np.concatenate((img_labels, label_vec), axis=0)

            fill = np.zeros((labels.shape[1] - img_labels.shape[0],) + (self.num_classes + 6,), dtype=np.float32)
            img_labels = np.expand_dims(np.concatenate((img_labels, fill), axis=0), axis=0)

            labels = np.concatenate((labels, img_labels), axis=0)

        return labels.reshape((labels.shape[0],) + (18, self.num_classes + 6))

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
    
    def get_image_set(self):
        print(f"getting images for {self.name} set:")
        img_data = tf.keras.utils.image_dataset_from_directory(self.x_path + self.name,
                                                               labels=None,
                                                               label_mode=None,
                                                               color_mode='rgb',
                                                               shuffle=False,
                                                               batch_size=None,
                                                               image_size=self.target_size)
        images = []
        for x in img_data.__iter__():
            images.append(x)
        return tf.stack(images, axis=0)

    def on_epoch_end(self):
        self.epoch_mod += len(self)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        idx =  idx + self.epoch_mod
        indices = tf.range(self.x.shape[0], dtype=tf.int64)
        seed_init = tf.random_uniform_initializer(0, indices[-1], seed=idx)
        seed = tf.Variable(seed_init(shape=(self.x.shape[0], 3), dtype=tf.int64), trainable=False)
        shuffled = tf.random_index_shuffle(indices, seed, indices[-1], rounds=4)
        batch_x = self.x.numpy()[shuffled[:self.batch_size], ...]
        batch_y = self.y[shuffled[:self.batch_size], ...]
        # print(f"X shape: {batch_x.shape}")
        # print(f"batch indices:\n{shuffled[:self.batch_size]}")
        # print(f"y shape: {batch_y.shape}")

        return batch_x, batch_y    

    # def __iter__(self):
    #     print(f"calling from iterator")
    #     for batch in range(len(self)):
    #         idx =  batch + self.epoch_mod
    #         indices = tf.range(self.x.shape[0], dtype=tf.int64)
    #         seed_init = tf.random_uniform_initializer(0, indices[-1], seed=idx)
    #         seed = tf.Variable(seed_init(shape=(self.x.shape[0], 3), dtype=tf.int64), trainable=False)
    #         shuffled = tf.random_index_shuffle(indices, seed, indices[-1], rounds=4)
    #         batch_x = self.x.numpy()[shuffled[:self.batch_size], ...]
    #         batch_y = self.y[shuffled[:self.batch_size], ...]
    #         # print(f"X shape: {batch_x.shape}")
    #         # print(f"batch indices:\n{shuffled[:self.batch_size]}")
    #         # print(f"y shape: {batch_y.shape}")
    #
    #         yield batch_x, batch_y

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

class DarknetTools:
    def __init__(self, data:list, image_size:List[int], project_directory:str, make:bool=False):
        self.data = data
        self.image_size = image_size
        self.project_dir = self._check_path(project_directory)
        self.image_paths = self._image_paths()
        self.img_files = self._img_files()
        self.labels = self._annot_to_darknet()
        if make:
            self._create_project()

    def _image_paths(self):
        result = []
        for subset in self.data:
            result.append(os.path.abspath(subset.x_path) + "/" + subset.name)
        return result

    def _check_path(self, path):
        assert isinstance(path, str)

        if path.endswith('/'):
            path = path[:-1]
            return path
        else:
            return path

    def _create_project(self):
        if not os.path.exists(self.project_dir):
            os.mkdir(self.project_dir)
            os.mkdir(self.project_dir + "/obj")
            print(f"created project directory at:\n{self.project_dir}")

    def _img_files(self):
        results = [] 
        for idx, search_path in enumerate(self.image_paths):
            filenames = np.char.asarray(np.char.split(sorted(glob(search_path + "/*.png")), '/'), unicode=True)
            results.append(np.char.rstrip(filenames[..., -1], '.png'))

        return results

    def _annot_to_darknet(self, image_size:List[int]=[768, 576]):
        """
        Function that takes a YOLO_Dataset object and returns a tensor containing darknet formatted
        values.
        """
        # result = {}
        result = []
        for idx, div in enumerate(self.data):
            annot = div.annot_to_tensor()
            # Center Coordinate
            center = annot[..., 14:16]
            # Width and Height Values
            corners = tf.transpose(get_corners(annot, img_width=image_size[0], img_height=image_size[1]), perm=[0, 1, 3, 2])
            sorted_points = tf.sort(corners, direction='DESCENDING', axis=-1)
            max_xy, min_xy = tf.transpose(sorted_points[..., 0:1], perm=[0, 1, 3, 2]), tf.transpose(sorted_points[..., -1:], perm=[0, 1, 3, 2])
            wh = max_xy - min_xy
            width, height = wh[..., 0] / image_size[0], wh[..., 1] / image_size[1]
            # Class Label Values
            cls_label = tf.cast(tf.argsort(annot[..., :13], direction='DESCENDING', axis=-1)[..., 0:1], dtype=tf.float32)
            # result.update({f"{div.name}": tf.concat([cls_label, center, width, height], axis=-1).numpy()})
            result.append(tf.concat([cls_label, center, width, height], axis=-1).numpy())
     
        return result

    def save_annotations(self, path_to_project=None, backup_dir:str="/content/gdrive/MyDrive/colab_files/darknet_yolo/"):
        """
        Class Method that processes the formatted data from this object and saves it to the txt files
        needed to train the darknet Yolo model.
        """
        if not path_to_project:
            path_to_project = self.project_dir
        # setting the default values for telling darknet where to find the image files
        prefix = np.char.array(["data/obj/"], unicode=True)
        suffix = np.char.array([".png"], unicode=True)
        for idx, dataset in enumerate(self.data):
            # Make sure there is a project folder created
            assert(os.path.exists(f"{path_to_project}/obj"))
            txt = prefix + self.img_files[idx] + suffix
            # This saves the train/test split designations to a file specifing which split each image
            # belongs to.
            np.savetxt(f"{path_to_project}/{dataset.name}.txt", txt, fmt='%-s')
            # No we loop through each split set from the data (train, val, test) and save a text file
            # for each image containing the class and bounding box labels for that image.
            for itm, label in enumerate(self.labels[idx]):
                indices = np.where(np.sum(label, axis=-1) > 0)
                np.savetxt(f'{path_to_project}/obj/{self.img_files[idx][itm]}.txt',
                           label[indices],
                           fmt='%-u %.9f %.9f %.9f %.9f')
        # Setting up for the basic config files
        class_num = int(np.max(self.labels[0][..., 0]) + 1)
        obj_data = [f"classes = {class_num}\n"]
        names = ["train", "valid", "test"]

        for name, dataset in zip(names, self.data):
            obj_data.append(f"{name} = data/{dataset.name}.txt\n")
        obj_data.extend(["names = data/obj.names\n", f"backup = {backup_dir}\n"])   
        # obj.data tells the model how many classes are in the dataset and where the data is along
        # with setting a directory location for saving backup weights during training
        with open(f"{path_to_project}/obj.data", 'w') as file:
            file.writelines(obj_data)
        # this last file records the names of the classes. I'd like to update this to except a list
        # of names at somepoint, but it's really unnecessary for the moment and at least for this
        # dataset there is no need to automate the creation of this file
        with open(f"{path_to_project}/obj.names", 'w') as file2:
            class_names = [f"{x}\n" for x in range(class_num)]
            file2.writelines(class_names)
