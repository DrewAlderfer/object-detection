# %%
from glob import glob
from itertools import cycle
import os
from typing import List, Union, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Rectangle, Polygon, Circle, PathPatch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PatchCollection, PolyCollection, LineCollection
import matplotlib.colors as mcolors
import numpy as np
import tensorflow as tf

from src.data_worker import YOLODataset, init_COCO
from src.utils import *
from src.disviz import setup_labels_plot

# %%
np.set_printoptions(suppress=True, precision=4)
# %load_ext autoreload

# %%
# %autoreload 2
# %aimport src.utils
# %aimport src.disviz
# %aimport src.data_worker

# %%
color = cycle(["orange", "crimson", "tomato",
               "springgreen", "aquamarine", 
               "fuchsia", "deepskyblue", 
               "mediumorchid", "gold"])
images = sorted(glob("./data/images/train/*"))

# %%
data = init_COCO("./data/", ['train', 'val', 'test'])

# %%
train_dataset = YOLODataset(data_name='train',
                            coco_obj=data,
                            image_path='./data/images/',
                            input_size=(1440, 1920),
                            target_size=(576, 768))

# %%
val_dataset = YOLODataset(data_name='val',
                            coco_obj=data,
                            image_path='./data/images/',
                            input_size=(1440, 1920),
                            target_size=(576, 768))

# %%
test_dataset = YOLODataset(data_name='test',
                            coco_obj=data,
                            image_path='./data/images/',
                            input_size=(1440, 1920),
                            target_size=(576, 768))

# %%
os.path.abspath(train_dataset.x_path) + "/" + train_dataset.name + "/"

# %%
train_labels = train_dataset.annot_to_tensor()
# adj = tf.constant([[[768, 576, 768, 576]]], dtype=tf.float32)
# phi = (train_labels[..., -1:]  * 2 * np.pi) - np.pi
# train_labels_adj = train_labels[..., -5:-1] * adj
# train_labels = tf.concat([train_labels[..., :14], train_labels_adj, phi], axis=-1)
print(f"train_labels: {train_labels.shape}, {train_labels.dtype}")
# print(f"train_labels_adj: {train_labels_adj.shape}, {train_labels_adj.dtype}")
# print(f"phi: {phi.shape}")
print(f"train_labels[0]:\n{train_labels[0, 1]}")

# %%
train_corners = get_corners(train_labels, img_width=768, img_height=576)
print(f"train_corners: {train_corners.shape}, {train_corners.dtype}")
print(f"train_corners:\n{train_corners[0, 1]}")

image = tf.keras.utils.load_img(images[0], target_size=(576, 768)) 
box = train_corners[0, 1]

dis_boxes = train_corners

img_sample = 0
box_collection = []
idx = 0
for i in dis_boxes[img_sample]:
    if tf.reduce_sum(i) == 0:
        continue
    n_color = next(color)
    xy, w, h = rect_vals[0, idx, 0:2], rect_vals[0, idx, 2], rect_vals[0, idx, 3]
    # box_collection.append(Circle(c, radius=5, color='springgreen'))
    box_collection.append(Rectangle(xy, w, h, color=n_color, fill=False))
    box_collection.append(Polygon(i, color=n_color, fill=False))
    idx += 1
box_collection = PatchCollection(box_collection, match_original=True)

fig, ax = setup_labels_plot()
ax[0].imshow(image)
ax[0].add_collection(box_collection)
plt.show()


# %%
def dis_std_annot(std_annot, image_size:List[int]=[768, 576]):
    cx, cy, w, h = std_annot[..., 1:2], std_annot[..., 2:3], std_annot[..., 3:4], std_annot[..., 4:]
    x = cx - w / 2 
    y = cy - h / 2
    min_xy = tf.concat([x, y], axis=-1) * tf.constant(image_size, shape=(1, 1, 2), dtype=tf.float32)
    print(min_xy.shape)
    width, height = w * image_size[0], h * image_size[1]

    return tf.concat([min_xy, width, height], axis=-1)
    
rect_vals = dis_std_annot(darknet_annot)
print(f"rect_vals: {rect_vals.shape}\n{rect_vals[0]}")

# %%
from glob import glob
import os
import shutil

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
        if not path_to_project:
            path_to_project = self.project_dir
        prefix = np.char.array(["data/obj/"], unicode=True)
        suffix = np.char.array([".png"], unicode=True)
        for idx, dataset in enumerate(self.data):
            assert(os.path.exists(f"{path_to_project}/obj"))
            txt = prefix + self.img_files[idx] + suffix
            np.savetxt(f"{path_to_project}/{dataset.name}.txt", txt, fmt='%-s')
            for itm, label in enumerate(self.labels[idx]):
                indices = np.where(np.sum(label, axis=-1) > 0)
                np.savetxt(f'{path_to_project}/obj/{self.img_files[idx][itm]}.txt',
                           label[indices],
                           fmt='%-u %.9f %.9f %.9f %.9f')

        class_num = int(np.max(self.labels[0][..., 0]) + 1)
        obj_data = [f"classes = {class_num}\n"]
        names = ["train", "valid", "test"]

        for name, dataset in zip(names, self.data):
            obj_data.append(f"{name} = data/{dataset.name}.txt\n")
        obj_data.extend(["names = data/obj.names\n", f"backup = {backup_dir}\n"])   

        with open(f"{path_to_project}/obj.data", 'w') as file:
            file.writelines(obj_data)

        with open(f"{path_to_project}/obj.names", 'w') as file2:
            class_names = [f"{x}\n" for x in range(class_num)]
            file2.writelines(class_names)
             



darknet = DarknetTools(data=[train_dataset, val_dataset, test_dataset], image_size=[768, 576], project_directory='./darknet_yolo', make=True)

# %%
darknet.save_annotations()

# %%
int(np.max(darknet.labels[0][..., 0]) + 1)

# %%
for idx, dataset in enumerate(darknet.data):
    print("-" * 40)
    print(f"{dataset.name}\n")
    print(f"img_names:      {darknet.img_files[idx].shape}")
    print(f"darknet_annots: {darknet.labels[idx].shape}")
    print(f"max value: {tf.reduce_max(darknet.labels[idx][..., 1:]).numpy()}")
    print(f"min value: {tf.reduce_min(darknet.labels[idx][..., 1:]).numpy()}")
    print("-" * 40)
    print(f"Sample Class Labels:")
    print(f"image: {darknet.img_files[idx][0]}")
    print(f"{darknet.labels[idx][0, :, 0]}")
# for idx, (name, annot) in enumerate(darknet_annots.items()):
#     print(name)
#     print(darknet.img_files[idx][0])
#     print(annot[0, 0])
