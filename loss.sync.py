# %%
import os
from itertools import cycle
from glob import glob
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import load_img

import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Rectangle, Polygon, Circle, PathPatch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PatchCollection, PolyCollection, LineCollection
import matplotlib.colors as mcolors
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras import Model
from tensorflow.types.experimental import TensorLike

from src.classes import CategoricalDataGen
from src.data_worker import LabelWorker, init_COCO
from src.utils import *
from src.models.layers import *
from src.models.models import YOLO_Loss
from src.disviz import setup_labels_plot

# %%
# %load_ext autoreload

# %%
# %autoreload 2
# %aimport src.utils
# %aimport src.classes
# %aimport src.data_worker
# %aimport src.models.layers
# %aimport src.models.models

# %%
color = cycle(["orange", "crimson", "tomato",
               "springgreen", "aquamarine", 
               "fuchsia", "deepskyblue", 
               "mediumorchid", "gold"])
images = sorted(glob("./data/images/train/*"))

# %%
data = init_COCO("./data/", ['train', 'val', 'test'])

# %%
labeler = LabelWorker(data_name='train',
                      coco_obj=data,
                      image_path='./data/images/',
                      input_size=(1440, 1920),
                      target_size=(576, 768))

# %%
num_anchors = 9
labels = labeler.annot_to_tensor()
anchors = generate_anchors(labels, boxes_per_cell=num_anchors, random_state=42)
label_corners = get_corners(labels, img_width=768, img_height=576)
anchor_corners = get_corners(anchors, img_width=768, img_height=576)
label_edges = get_edges(label_corners)
anchor_edges = get_edges(anchor_corners)
print(f"labels shape: {labels.shape}")
print(f"label_corners shape: {label_corners.shape}")
print(f"label_edges shape: {label_edges.shape}")
print(f"anchors shape: {anchors.shape}")
print(f"anchor_corners shape: {anchor_corners.shape}")
print(f"anchor_edges shape: {anchor_edges.shape}")

# %%
arr = labels[..., -1:].flat
max_idx = tf.argsort(arr, direction="DESCENDING")
max_idx = np.unravel_index(max_idx, (16, 18, 1))

labels[..., -1:][max_idx]

# %%




# %%
class yolo_dataset(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size):
        assert x_set.shape[0] == y_set.shape[0]
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        indices = tf.range(self.x.shape[0], dtype=tf.int64)
        seed_init = tf.random_uniform_initializer(0, indices[-1], seed=idx)
        seed = tf.Variable(seed_init(shape=(self.x.shape[0], 3), dtype=tf.int64), trainable=False)
        shuffled = tf.random_index_shuffle(indices, seed, indices[-1], rounds=4)
        batch_x = self.x.numpy()[shuffled[:self.batch_size]]
        batch_y = self.y[shuffled[:self.batch_size]]

        return batch_x, batch_y, shuffled[:self.batch_size]

# %%
img_data = tf.keras.utils.image_dataset_from_directory('./data/images/train_imgs/train/',
                                                       labels=None,
                                                       label_mode=None,
                                                       color_mode='rgb',
                                                       shuffle=False,
                                                       batch_size=None,
                                                       image_size=(576, 768))
images = []
for x in img_data.__iter__():
    images.append(x)
image_set = tf.stack(images, axis=0)
print(f"image_set: {image_set.shape}")
train_datagen = yolo_dataset(image_set, labels, 16)

# %%
x = tf.keras.layers.Rescaling(1./255)(train_datagen[1][0][0:1])
x = tf.keras.layers.Conv2D(16, 3, activation='relu', strides=2, padding='same')(x)
x = tf.keras.layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(x)
x = tf.keras.layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
x = tf.keras.layers.Conv2D(128, 3, activation='relu', strides=2, padding='same')(x)
x = tf.keras.layers.Conv2D(128, 3, activation='relu', strides=2, padding='same')(x)
x = tf.keras.layers.Conv2D(256, 3, activation='relu', strides=2, padding='same')(x)
# x = tf.keras.layers.Multiply()([x, 0])
outputs = tf.keras.layers.Dense(19 * 9, activation='softsign')(x)
# x = tf.keras.layers.Reshape(target_shape=(108, 9, 19))(x)
# z = tf.keras.layers.Add()([x[..., -5:], tf.reshape(anchors, (1, 108, 9, 5))])
#
# # components = BoxCutter(num_classes=13, units=5)(x)
# # bboxes = AddAnchors(anchors[0])(components[2])
# outputs = tf.keras.layers.Concatenate(axis=-1)([x[..., -5:], z])


# %%

# %%
fig, ax = setup_labels_plot()
ax = ax[0]
# --------------------
# Object Setup
# --------------------
cell = 46
bb_list = [0, 6]
itm_list = [1, 4]
x = 0
y = 388
# for i, a in zip(bb_list, itm_list):
for i in range(18):
    if label_corners[0, i, 0, 0] == 0:
        continue
    n_color = next(color)
    ax.add_patch(Polygon(label_corners[0, i], fill=False, edgecolor=n_color))
    ax.imshow(imgs[0] / 255)
    x += 150
plt.show()


# %%
# test_arr = tf.zeros((1, 108, 9, 5), dtype=tf.float32)
# test_val = tf.constant([2], shape=(1,5), dtype=tf.float32)
# indices = tf.constant([[0, 49, 0]])
# test_arr = tf.tensor_scatter_nd_add(test_arr, indices, test_val)
# tf.sigmoid(outputs).numpy().reshape(1, 108, 9, 19)[0, 0, 0, -5:]
y_preds = outputs.numpy().reshape(1, 108, 9, 19)[..., -5:] + anchors.reshape(1, 108, 9, 5)
y_corners = get_corners(y_preds, img_width=768, img_height=567)
print(f"anchors: {anchors[0, 0, 0, 0]}")
print(f"y_preds: {y_preds[0, 0, 0]}")
print(f"y_corners: {y_corners.shape}")
img = train_datagen[1][2][0]
print(f"img: {img}")
fig, ax = setup_labels_plot(num_plots=(1, 2))
ax1 = ax[0]
ax2 = ax[1]
# --------------------
# Object Setup
# --------------------
points = tf.reduce_mean(anchor_corners[..., 0:1, :, :], axis=-2)
y_points = tf.reduce_mean(y_corners[..., 0:1, :, :], axis=-2)
label_points = tf.reduce_mean(label_corners[img, ..., 0:1, :, :], axis=-2)
print(f"points: {points.shape}")
print(f"y_points: {y_points.shape}")
print(f"label_points: {label_points.shape}")
points = tf.reshape(points, shape=(1, 108, 1, 2))
y_points = tf.reshape(y_points, shape=(1, 108, 1, 2))
# points = tf.transpose(points, perm=(0, 2, 1, 3, 4))
print(f"points: {points.shape}")
color_idx = tf.range(108)
print(f"color_idx: {color_idx.shape}")
# for i, a in zip(bb_list, itm_list):
ax2.add_patch(Polygon(y_corners[0, 49, 1], color='springgreen', fill=False))
ax1.scatter(points[0, ..., 0], points[0, ..., 1], c=color_idx, cmap='jet')
ax2.scatter(y_points[0, ..., 0], y_points[0, ..., 1], c=color_idx, cmap='jet')
ax1.imshow(train_datagen[1][0][0] / 255)
ax2.imshow(train_datagen[1][0][0] / 255)
plt.show()


# %%
a = tf.constant(tf.range(10, dtype=tf.float32))
tf.keras.activations.sigmoid(a)
