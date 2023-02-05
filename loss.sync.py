# %%
import os
from itertools import cycle
from glob import glob
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import load_img

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
labels = labeler.label_list()[:16]
anchors = stack_anchors(generate_anchors(labels, boxes_per_cell=num_anchors, random_state=42))
label_corners = get_corners(labels, img_width=768, img_height=576)
anchor_corners = get_corners(anchors)
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
img_data = tf.keras.utils.image_dataset_from_directory('./data/images/train/',
                                                       labels=None,
                                                       color_mode='rgb',
                                                       batch_size=16,
                                                       shuffle=False,
                                                       image_size=(576, 768))
imgs = img_data.take(1).get_single_element()
print(f"x: {imgs.shape}, {imgs.dtype}, {tf.size(imgs)}")


# %%
def get_model(img_size, batch_size):
    inputs = tf.keras.Input(shape=img_size + (3,))
    x = tf.keras.layers.Rescaling(1./255)(inputs)
    x = tf.keras.layers.Conv2D(16, 3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding='same')(x)
    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(256, 3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(19 * 9)(x)
    components = BoxCutter(num_classes=13, units=5)(x)
    bboxes = AddAnchors(anchors[0])(components[2])
    components[2] = bboxes
    outputs = tf.keras.layers.Concatenate(axis=-1)(components)
    model = Model(inputs, outputs)

    return model

model = get_model(img_size=(576, 768), batch_size=16)
model.summary()


# %%
def get_model(img_size, batch_size):
    inputs = tf.keras.Input(shape=img_size + (3,))
    x = tf.keras.layers.Rescaling(1./255)(inputs)
    x = tf.keras.layers.Conv2D(16, 3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding='same')(x)
    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(256, 3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(19 * 9)(x)
    components = BoxCutter(num_classes=13, units=5)(x)
    bboxes = AddAnchors(anchors[0])(components[2])
    components[2] = bboxes
    outputs = tf.keras.layers.Concatenate(axis=-1)(components)
    model = Model(inputs, outputs)

    return model

model = get_model(img_size=(576, 768), batch_size=16)
model.summary()

# %%
model.compile(optimizer='adam', loss=YOLO_Loss())
history = model.fit(x, labels,
                    epochs=10,
                    batch_size=16)

# %%
loss = YOLO_Loss()(labels, detectors)


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
