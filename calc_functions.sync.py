# %%
from itertools import cycle
from glob import glob
from typing import Union, Tuple, List
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Rectangle, Polygon, Circle
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PolyCollection
import matplotlib.colors as mcolors

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pycocotools import coco
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import Loss
from tensorflow.keras.utils import load_img
import pprint as pp

from src.utils.box_cutter import BoundingBox_Processor
from src.utils.classes import CategoricalDataGen
from src.utils.data_worker import LabelWorker
from src.utils.funcs import (init_COCO, make_predictions, 
                             generate_anchors, translate_points,
                             display_label, stack_anchors,
                             get_corners, get_edges,
                             get_intersections, rolling_intersection)

# %%
# %load_ext autoreload

# %%
# %autoreload 2
# %aimport src.utils.funcs
# %aimport src.utils.box_cutter
# %aimport src.utils.classes
# %aimport src.utils.data_worker

# %%
pp.PrettyPrinter(indent=4)

# %%
data = init_COCO("./data/", ['train', 'val', 'test'])
box_cutter = BoundingBox_Processor()

# %%
labeler = LabelWorker(data_name='train',
                      coco_obj=data,
                      image_path='./data/images/',
                      input_size=(1440, 1920),
                      target_size=(384, 512))

labels = labeler.label_list()[0:2]

anchors = generate_anchors(labels, random_state=103)
a_list = stack_anchors(anchors)
print(f"anchors shape: {anchors.shape}")
print(f"stacked anchors: {a_list.shape}")
label_corners = get_corners(labels)

anchor_corners = get_corners(a_list)
# print(f"anchors: {anchors.shape}")
# print(f"{anchors[0, 0, 0]}")
print(f"a_list: {a_list.shape}")
print(f"{a_list[0, 0}")
# print(f"anchor_corners: {anchor_corners.shape}")
# print(f"{anchor_corners[0, 0]}")

# %%
old_corners = box_cutter.get_corners(labels)[0]
old_anchors = box_cutter.get_corners(anchors)[0]
print(f"old_corners: {old_corners.shape}")
print(f"old_anchros: {old_anchors.shape}")
old_intersections = box_cutter.rolling_intersection(old_corners, old_anchors)[0]
old_intersections.shape

# %%
label_edges = get_edges(label_corners).numpy()
print(f"label_edges: {label_edges.shape}")
anchr_edges = get_edges(anchor_corners)
step1 = label_edges.reshape(2, 1, 108, 4, 2, 2)
step2 = np.full((2, 108, 108, 4, 2, 2), step1)
print(f"step_2: {step2.shape}")
print(anchr_edges[0, 0])


# %%
anchr_r = anchr_edges.numpy().reshape(2, 1, 324, 4, 2, 2)
print(f"label_edges: {step2[..., 0:1, :, :].shape}")
print(f"anchr_edges: {anchr_edges[..., 0:, :, :].shape}")
print(f"anchr_r: {anchr_r.shape}")

# %%
intersections, l_edge, a_edge = rolling_intersection(step2, anchr_r)
x = intersections.numpy()
x[0, 1, 4::9]

# %%
# (a, b, c, c)
# (a, b, 1, c, c)
# (a, b, b, c, c)
# calc = np.where(label_edges > 0.0, label_edges, label_edges/0)
array = np.array([1, 2, 3, 4]).reshape(4, 1, 1, 1, 1)
# arr = np.array(array).reshape(1, 1, 1, 1, 1)
print(f"array: {array.shape}")
arr = np.full((4, 108, 4, 2, 2), array, dtype=np.float32)
print(f"arr: {arr.shape}")
arr

# %%
all_boxes = np.full((1, 108, 1, 4, 2, 2), arr, dtype=np.float32)
all_boxes

# %%
color = cycle(["orange", "crimson", "tomato",
               "springgreen", "aquamarine", 
               "fuchsia", "deepskyblue", 
               "mediumorchid", "gold"])
images = sorted(glob("./data/images/train/*"))

# %%
label_edges[0, 0].shape

# %%
fig, ax = plt.subplots(figsize=(8, 6))
ax.set(
        ylim=[0, 384],
        xlim=[0, 512],
        xticks=list(range(0, 512,int(np.ceil(512/12)))),
        yticks=list(range(0, 384, int(np.ceil(384/9)))),
        )
for i in range(24):
    ax.add_collection(mpl.collections.LineCollection(step2[0, 0, i]))
for i in range(9):
    ax.add_collection(mpl.collections.LineCollection(anchr_edges[0, i:108:9].numpy().reshape(24, 4, 2), color="springgreen"))
    ax.add_collection(mpl.collections.LineCollection(anchr_edges[0, i+108:216:9].numpy().reshape(24, 4, 2), color="tomato"))
    ax.add_collection(mpl.collections.LineCollection(anchr_edges[0, i+216:324:9].numpy().reshape(24, 4, 2), color="orange"))
plt.savefig("./images/anchor_box_illustration.png")
plt.show()

# %%
fig, axs = plt.subplots(2, 1, figsize=(8, 10))
# axs = np.concatenate([ax1, ax2], axis=-1)
for img, ax in enumerate(axs):
    lines = []
    for i in range(1, 12, 1):
        line = i * 512/12
        lines.append([(line, 0), (line, 384)])
    for i in range(1, 9, 1):
        line = i * 384/9
        lines.append([(0, line), (512, line)])
    grid_lines = mpl.collections.LineCollection(lines, colors='black', lw=1, ls='--', alpha=.4)
    ax.set(
            ylim=[0, 384],
            xlim=[0, 512],
            xticks=list(range(0, 512,int(np.ceil(512/12)))),
            yticks=list(range(0, 384, int(np.ceil(384/9)))),
            )
    ax.axis('off')
    ax.imshow(load_img(images[img], target_size=(384, 512)))
    ax.add_collection(grid_lines)
    for idx, box in enumerate(label_corners[img]):
        bbox, arrow = display_label(np.reshape(labels, (labels.shape[0],) + (12 * 9, labels.shape[-1]))[img, idx, 14:], ax, color=next(color))
        ax.add_patch(bbox)
        ax.add_patch(arrow)
        # ax.add_patch(Polygon(label_corners.reshape(269, 12 * 9, 4, 2)[img, idx], fill=None, edgecolor="tomato", lw=5))
fig.tight_layout()
plt.savefig("./images/bounding_box_examples_2.png")
plt.show()

# %%
fig, [ax1, ax2, ax3, ax4] = plt.subplots(4, 3, figsize=(12, 12))
axs = np.concatenate([ax1, ax2, ax3, ax4], axis=-1)
for img, ax in enumerate(axs):
    lines = []
    for i in range(1, 12, 1):
        line = i * 512/12
        lines.append([(line, 0), (line, 384)])
    for i in range(1, 9, 1):
        line = i * 384/9
        lines.append([(0, line), (512, line)])
    grid_lines = mpl.collections.LineCollection(lines, colors='black', lw=1, ls='--', alpha=.4)
    ax.set(
            ylim=[0, 384],
            xlim=[0, 512],
            xticks=list(range(0, 512,int(np.ceil(512/12)))),
            yticks=list(range(0, 384, int(np.ceil(384/9)))),
            )
    ax.axis('off')
    ax.imshow(load_img(images[img], target_size=(384, 512)))
    ax.add_collection(grid_lines)
    for idx, box in enumerate(label_corners[img]):
        bbox, arrow = display_label(np.reshape(labels, (labels.shape[0],) + (12 * 9, labels.shape[-1]))[img, idx, 14:], ax, color=next(color))
        ax.add_patch(bbox)
        ax.add_patch(arrow)
        # ax.add_patch(Polygon(box, fill=None, edgecolor="tomato", lw=5))
fig.tight_layout()
plt.savefig("./images/bounding_box_examples_16.png")
plt.show()

# %%
print(labels[0, ..., 14])
np.floor((labels[0, ..., 14] / 512) * 12)
