# %%
"""
Hello
"""
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

from src.utils.box_cutter import BoundingBox_Processor
from src.utils.classes import CategoricalDataGen
from src.utils.data_worker import LabelWorker, init_COCO
from src.utils.funcs import *
from src.models.layers import *

# %%
# %load_ext autoreload

# %%
# %autoreload 2
# %aimport src.utils.funcs
# %aimport src.utils.box_cutter
# %aimport src.utils.classes
# %aimport src.utils.data_worker
# %aimport src.models.layers

# %%
color = cycle(["orange", "crimson", "tomato",
               "springgreen", "aquamarine", 
               "fuchsia", "deepskyblue", 
               "mediumorchid", "gold"])
images = sorted(glob("./data/images/train/*"))

# %%
data = init_COCO("./data/", ['train', 'val', 'test'])
box_cutter = BoundingBox_Processor()

# %%
labeler = LabelWorker(data_name='train',
                      coco_obj=data,
                      image_path='./data/images/',
                      input_size=(1440, 1920),
                      target_size=(384, 512))

# %%
num_anchors = 9
labels = labeler.label_list()[:16]
anchors = stack_anchors(generate_anchors(labels, boxes_per_cell=num_anchors, random_state=42))
label_corners = get_corners(labels)
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
x = np.asarray(load_img("./data/images/train/screws_002.png",
                          color_mode='rgb',
                          target_size=(576, 768)), dtype=np.float32)

x = tf.expand_dims(x, axis=0)
print(f"input:  {x.shape}")
x = tf.keras.layers.Conv2D(16, 3, strides=2, padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
print(f"x:      {x.shape}")
x = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
print(f"x:      {x.shape}")
print(f"x:      {x.shape}")
x = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
print(f"x:      {x.shape}")
x = tf.keras.layers.Conv2D(128, 3, strides=2, padding='same')(x)
print(f"x:      {x.shape}")
x = tf.keras.layers.Conv2D(128, 3, strides=2, padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Conv2D(256, 3, strides=2, padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
print(f"x:      {x.shape}")
x = tf.keras.layers.Dense(19 * 9)(x)
print(f"x:      {x.shape}")

detectors = BoxCutter(num_classes=13, units=5)(x)
x3 = AddAnchors(anchors[0:1])(detectors[2])
detectors[2] = x3
print(f"x3: {x3.shape}")

# %%
def YOLO_Loss(y_true, y_pred):
    """
    Step1. y_true is currently in shape (1, 18, 19).
        - You want to take each of these labels, and match it to the detector with the best fit
        - This means scattering the tensor into a shape (1, 108, 9, 19)
        - You should also shuffle the labels before so that if there are overlaps then over time
          hopefully the best fit detector will begin to pick up the and specialize on the objects.
    """
    pred_class, pred_obj, pred_bboxes = y_pred
    # true_bboxes = y_true[..., pred_class.shape[-1] + 1:]
    # true_bboxes = tf.reshape(true_bboxes, shape=true_bboxes.shape[:2] + (1, 1, true_bboxes.shape[-1]))
    # pred_bboxes = tf.expand_dims(pred_bboxes, axis=1)
    print(f"y_true: {y_true.shape}")
    print(f"pred_bboxes: {pred_bboxes.shape}")
    true_xy = y_true[..., 14:16]
    grid_dims = tf.constant([[12, 9]], dtype=tf.float32)
    cell_num = tf.cast(tf.math.floor(true_xy * grid_dims), dtype=tf.int32)
    middle_axis = tf.broadcast_to(tf.constant(tf.range(18, dtype=tf.int32), shape=[1, 18, 1]), shape=[1, 18, 1])
    batch_idx = tf.zeros(cell_num.shape[:-1] + (1,), dtype=tf.int32)
    true_idx = tf.concat([batch_idx, middle_axis, cell_num[..., 0:1] * 9 + cell_num[..., 1:]], axis=-1)
    # shoot for shape (1, 18, 108, 19) because then each label will have it's own grid
    # then you can compare the best fit for each cell on each anchor
    background = tf.zeros([1, 18, 108, 19], dtype=tf.float32)
    updates = y_true
    print(f"background: {background.shape}")
    print(f"true_idx:   {true_idx.shape}")
    print(f"updates:    {updates.shape}")
    print(f"true_idx:\n{true_idx[0]}")
    true_cells = tf.scatter_nd(true_idx, updates, shape=(1, 18, 108, 19))
    print(f"y_true: {y_true[0, 0, 14]}")
    true_cells = tf.broadcast_to(tf.expand_dims(true_cells, axis=-2), shape=(1, 18, 108, 9, 19))
    pred_bboxes = tf.broadcast_to(tf.expand_dims(pred_bboxes, axis=1), shape=true_cells.shape[:-1] + (5,))
    print(f"true_cells: {true_cells.shape}")
    print(f"pred_bboxes: {pred_bboxes.shape}")
    intersection_points, label_corners, anchor_corners = construct_intersection_vertices(true_cells, pred_bboxes) 
    intersection = intersection_area(intersection_points)
    union = union_area(label_corners, anchor_corners, intersection, num_pumps=9)
    giou = calculate_giou(label_corners, anchor_corners, union, intersection)
    giou = tf.reshape(giou, true_cells.shape[:-1] + (1,)) * true_cells[..., 13:14]
    best_bboxes = tf.argsort(giou, axis=-2)


    return best_bboxes, anchor_corners

results, test_corners = YOLO_Loss(labels[0:1], detectors)
print(f"results: {results.shape}, {results.dtype}")
print(f"results:\n{results[0, 2:5, 77]}")

# %%
print(f"results:\n{results[0, 14:16, 52]}")

# %%
y = AnchorLayer(anchors[0], units=5, num_boxes=9, xdivs=12, ydivs=9)(p_bboxes)
y_list = tf.reshape(y, (1, 972, 5))
y_corners = get_corners(y_list)
print(f"y: {y.shape}")
print(f"y_corners: {y_corners.shape}")

# %%
preds = nonmaxsuppresion(labels[0:1], anchors[0:1])

# %%
(1, 18, 972, 4, 2) == (1, 18, 972, 4, 2)

# %%
y = tf.reshape(y, (1, 972, 5))
preds = nonmaxsuppresion(labels[0:1], y)
print(f"preds: {preds.shape}")


# %%
# (1, 108, 19)
def YOLO_Loss(y_true, y_pred):
    lambda_coord = 5.0
    lambda_noobj = 0.5
    p_class, p_obj, p_bboxes = y_pred 
    class_obj_len = p_class.shape[-1] + 1
    b_units = y_true.shape[-1] - (p_class.shape[-1] + 1)
    print(f"b_units: {b_units}")
    background = tf.zeros(shape=p_class.shape[:2] + (class_obj_len,), dtype=tf.float32)
    x = y_true[..., -5:-4]
    y = y_true[..., -4:-3]
    z = tf.zeros(y_true.shape[:-1] + (1,), dtype=tf.int32)

    x_idx = tf.cast(tf.math.floor(x * 12), dtype=tf.int32)
    y_idx = tf.cast(tf.math.floor(y * 9), dtype=tf.int32)
    cell_idx = tf.concat([z, x_idx * 9 + y_idx], axis=-1)
    cell_labels = tf.tensor_scatter_nd_add(background, cell_idx, y_true[..., :class_obj_len])
    cell_labels = tf.cast(cell_labels > 0, dtype=tf.float32)
    print(f"background: {background.shape}")
    print(f"cell_idx: {cell_idx.shape}")
    print(f"updates: {y_true[..., :class_obj_len].shape}")
    print(f"cell_labels: {cell_labels.shape}")
    obj_exists = cell_labels[..., 13:14]
    obj_loss = tf.reduce_sum(tf.math.squared_difference(cell_labels[..., 13:], p_obj)) * obj_exists
    class_loss = tf.reduce_sum(tf.math.squared_difference(cell_labels[..., :13], p_class), axis=-1) * obj_exists
    print(f"obj_exists: {obj_exists.shape}")

    
    true_x = tf.reshape(x, shape=y_true.shape[:2] + (1, 1, 1))
    true_x = tf.broadcast_to(true_x, shape=(1, 18, 108, 9, 1))
    pred_x = tf.expand_dims(p_bboxes[..., 0:1], axis=1)
    true_y = tf.reshape(y, shape=y_true.shape[:2] + (1, 1, 1))
    true_y = tf.broadcast_to(true_y, shape=(1, 18, 108, 9, 1))
    pred_y = tf.expand_dims(p_bboxes[..., 1:2], axis=1)
    print(f"x_true: {true_x.shape}")
    print(f"x_true:\n{true_x[0, 0, 0]}")
    xy_loss = (tf.math.squared_difference(true_x, pred_x) + tf.math.squared_difference(true_y, pred_y)) * tf.reshape(obj_exists, (1, 1, 108, 1, 1))
    xy_loss = tf.reduce_sum(xy_loss) * lambda_coord
    true_w = tf.reshape(y_true[..., -3:-2], shape=y_true.shape[:2] + (1, 1, 1))
    true_w = tf.broadcast_to(true_w, shape=(1, 18, 108, 9, 1))
    pred_w = tf.expand_dims(p_bboxes[..., 2:3], axis=1)
    true_h = tf.reshape(y_true[..., -2:-1], shape=y_true.shape[:2] + (1, 1, 1))
    true_h = tf.broadcast_to(true_y, shape=(1, 18, 108, 9, 1))
    pred_h = tf.expand_dims(p_bboxes[..., 3:4], axis=1)
    print(f"w_true: {true_w.shape}")
    print(f"h_true: {true_h.shape}")
    wh_loss = (tf.math.squared_difference(tf.math.sqrt(true_w), tf.math.sqrt(pred_w)) + tf.math.squared_difference(tf.math.sqrt(true_h), tf.math.sqrt(pred_h))) * tf.reshape(obj_exists, (1, 1, 108, 1, 1))
    wh_loss = tf.reduce_sum(wh_loss) * lambda_coord
    print(f"class_loss: {class_loss.shape}")
    print(f"obj_loss: {obj_loss.shape}")
    print(f"xy_loss: {xy_loss.shape}")
    print(f"wh_loss: {wh_loss.shape}")

    return xy_loss + wh_loss
results = YOLO_Loss(labels[0:1], detectors)
print(f"results: {results.shape}")
print(f"results: {results[0]}")

# %%
class YOLO_Output(Layer):
    def __init__(self):
        super().__init__() 

    def call(self, p_class, p_obj, bboxes):
        return (p_class, p_obj, bboxes)


# %%
a = anchors[0:1]
x = tf.random_normal_initializer()
x = x(shape=(anchors[0:1].shape), dtype=tf.float32)
print(f"a:  {a.shape}")
print(f"x:  {x.shape}")
print(f"y:  {y.shape}")
y = AnchorLayer(a, units=5, num_boxes=9, xdivs=12, ydivs=9)(x)
print(f"y: {y.shape}\n{y[0,0]}")


# %%
np.set_printoptions(suppress=True)

# %%
x3_dis = tf.reshape(x3, shape=(1, 972, 5))
preds = nonmaxsuppresion(labels[0:1], x3_dis)
preds.shape

# %%
preds = nonmaxsuppresion(labels[0:1], anchors[0:1])
preds.shape

# %%

true_xy = labels[0, 2:5:2, 14:16]
grid_dims = tf.constant([[12, 9]], dtype=tf.float32)
cell_num = tf.cast(tf.math.floor(true_xy * grid_dims), dtype=tf.int32)
cell_num


# %%
print(f"anchors: {anchors[0, 0]}")
print(f"test_corners: {test_corners.shape}")
# dis_anchors = tf.reshape(anchors[0:1], (1, 108, 9, 5))
dis_anchors = get_corners(anchors)
print(f"dis_anchors: {dis_anchors.shape}")
print(f"anchors: {anchors.shape}")
test_corners[0, 2, 77, 2]

# %%
test_corners = tf.sqrt(tf.math.square(tf.reshape(test_corners, (1, 18, 108, 9, 4, 2))))
axs = []
# pred_corners = get_corners(pred_boxes) 
fig, ax = plt.subplots(figsize=(8, 6))
img = 0
ax.set(
        ylim=[-40, 384+40],
        xlim=[-40, 512+40],
        )
lines = []
for i in range(0, 13, 1):
    line = i * 512/12
    lines.append([(line, 0), (line, 384)])
for i in range(0, 10, 1):
    line = i * 384/9
    lines.append([(0, line), (512, line)])
grid_lines = mpl.collections.LineCollection(lines, colors='black', lw=1, alpha=.4, zorder=200)
ax.add_collection(grid_lines)
# for i in range(preds.shape[1]):
a = 7
for i in [14, 15]:
# for i in range(9):
    n_color = next(color)
    ax.add_patch(Polygon(label_corners[0, i], fill=False, edgecolor=n_color))
    ax.add_patch(Polygon(test_corners[0, i, 52, a], fill=False, edgecolor=n_color, ls="--"))
    a = a - 3 
    # ax.add_patch(Polygon(dis_anchors[0, 77, i], fill=False, edgecolor=n_color, ls="--"))
    # ax.add_patch(Polygon(get_corners(preds)[img, i], fill=False, edgecolor=n_color, ls="--"))
ax.tick_params(
        axis='both',
        which='both',
        bottom = False,
        left = False,
        labelleft = False,
        labelbottom=False
        )
plt.show()

# %%
a = tf.reshape(tf.constant(tf.range(108, dtype=tf.float32)), shape=(1, 12, 9, 1))
a = tf.broadcast_to(a, shape=(1, 12, 9, 256))
a = tf.reshape(a, (1, 12, 9, 256))
print(f"a:\n{a[0, 0]}")

# %%
d = 5
i = tf.keras.initializers.Constant(value=1)
# i = tf.keras.initializers.RandomNormal(stddev=0.01)
# a = tf.reshape(tf.constant(tf.range(1, 12 * 19 + 1)), shape=(1, 12, 19))
# a = tf.broadcast_to(a, shape=(1, 9, 12, 19))
print(f"a: {a.shape}")
# print(tf.transpose(a, perm=(0, 2, 1, 3)))
# a = tf.keras.layers.Flatten()(a)
a = tf.keras.layers.Dense(13 + 5 * 9, use_bias=True, kernel_initializer=i)(a)
print(f"a size: {a.numpy().size}")
print(f"a shape: {a.shape}")
classes = a[..., :12]
object = a[..., 12:13]
boxes = tf.concat(tf.split(tf.expand_dims(a[..., 13:], axis=-2), 9, axis=-1), axis=-2)
print(f"boxes: {boxes.shape}")

# %%
print(f"a:\n{a[0, 0]}")
print(f"boxes:\n{boxes[0, 0]}")

# %%
# a = tf.constant(tf.range(0, 768, delta=768/12, dtype=tf.float32))
# b = tf.constant(tf.range(0, 512, delta=512/9, dtype=tf.float32))
# X, Y = tf.meshgrid(a, b)
# X = tf.reshape(X, (1, 9, 12))
# Y = tf.reshape(Y, (1, 9, 12))
x = labels[..., 14:15]
y = labels[..., 15:16]
X = 768
Y = 576
x_idx = tf.cast(tf.math.floor(x * 12), dtype=tf.int32)
y_idx = tf.cast(tf.math.floor(y * 9), dtype=tf.int32)
# x = tf.broadcast_to(x, (1, 9, 12))
# y = tf.broadcast_to(y, (1, 9, 12))
# A = tf.searchsorted(X, x, side='left').numpy()
# B = tf.searchsorted(Y, y, side='left').numpy()
print(f"x: {x.shape}")
print(f"y: {y.shape}")
print(f"x: {x[0]}")
print(f"x_idx: {x_idx[0]}")
print(f"y_idx: {y_idx[0]}")
print(x_idx[0] * 9 + y_idx[0])

# %%
a = tf.zeros([8, 5], dtype=tf.int32)
# b = tf.constant([], dtype=tf.int32)
b = tf.constant([[[0, 0], [2, 4], [0, 2], [0, 3], [0, 4]],
                 [[1, 0], [1, 1], [1, 2], [1, 3], [1, 4]], 
                 [[2, 0], [2, 1], [2, 2], [2, 3], [2, 4]]], dtype=tf.int32)
c = tf.reshape(tf.constant(tf.range(15, dtype=tf.int32)), (3, 5))
# c = tf.broadcast_to(c, shape=(3, 5))
print(f"a: {a.shape}")
print(f"b: {b.shape}")
print(f"c:\n{c}")
tf.tensor_scatter_nd_add(a, b, c)


# %%
tf.constant([1], shape=(3,), dtype=tf.int32)
