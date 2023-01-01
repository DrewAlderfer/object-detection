# %%
import pprint as pp
pp.PrettyPrinter(indent=4)
from typing import Tuple, Union, List
import numpy as np
from numpy.typing import ArrayLike, NDArray
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle
import tensorflow as tf
from tensorflow.keras.utils import load_img

from src.utils.funcs import init_COCO, process_img_annotations, rotate
from src.utils.classes import CategoricalDataGen, bbox_worker
from src.models.models import IoU, YOLOLoss
from src.models.loss import BoundingBox_Processor



# %%
# %load_ext autoreload

# %%
# %autoreload 2
# %aimport src.utils.funcs
# %aimport src.utils.classes
# %aimport src.models

# %%
np.set_printoptions(formatter={'float_kind':"{:.4f}".format})

# %%
data = init_COCO("./data/", ["train", "val", "test"])

# %%
test = CategoricalDataGen('test', data, "./data/images", target_size=(384, 512))

# %%
xdivs = 12
ydivs = 9

# %%
labels = test.get_labels(divs=(ydivs, xdivs), num_classes=13, num_boxes=3,
                input_size=(1440, 1920))
labels.shape

# %%
preds = np.zeros((60, xdivs, ydivs, 13 + 3 * 6), dtype=np.float32)
preds[...,:19] = preds[...,:19] + labels
preds[..., 19:25] = labels[..., 13:19] 
preds[..., 25:] = labels[..., 13:19] 
b_e = preds[..., 13]
b_e = b_e.reshape((preds.shape[:-1] + (1,)))
print(b_e.shape)
box_exists = np.concatenate((b_e, b_e, b_e, b_e, b_e), axis=-1)
known_loss = np.asarray([10, 10, 5, 20, np.pi * .05])
known_loss = np.full(preds.shape[:-1] + (5,), known_loss, dtype=np.float32)
preds[..., 14:19] = known_loss + preds[..., 14:19]
preds[..., 20:25] = box_exists * (preds[..., 20:25] + np.random.normal(0, 2, preds.shape[:-1] + (5,)))
preds[..., 26:] = box_exists * (preds[..., 26:] + np.random.normal(0, 2, preds.shape[:-1] + (5,)))

# %%
box_cutter = BoundingBox_Processor()

# %%
label_corners = box_cutter.get_corners(labels)
pred_corners = box_cutter.get_corners(preds)
pred_corners = pred_corners[0]
label_corners = label_corners[0]
print(pred_corners.shape)

# %%
img = np.asarray(load_img("./data/images/test/screws_006.png", target_size=(384, 512)), dtype=np.float32)
img = img / 255
fig, ax = plt.subplots(figsize=(xdivs, ydivs))
ax.imshow(img)
y_true = label_corners[0]
print(y_true.shape)
y_true = tf.reshape(y_true, [ydivs * xdivs, 4, 2])
for label in y_true:
    # print(f"Label:\n{label}")
    bbox = label
    ax.add_patch(Polygon(bbox, fill=None, edgecolor='tab:purple', lw=2))
    ax.add_patch(Circle(np.mean(bbox, axis=0), radius=3, fill=None, edgecolor="red", lw=2))
    ax.scatter(bbox[2][0], bbox[2][1], marker='x', color='springgreen', s=60)
    ax.scatter(bbox[0][0], bbox[0][1], marker='x', color='tomato', s=60)
y_pred = pred_corners[0]
y_pred = tf.reshape(y_pred, [ydivs * xdivs, 4, 2])
for pred in y_pred:
    bbox = pred
    ax.add_patch(Polygon(bbox, fill=None, edgecolor='tomato', lw=2))
#     ax.add_patch(Circle(np.mean(bbox, axis=0), radius=3, fill=None, edgecolor="tab:blue", lw=2))
#     ax.scatter(bbox[2][0], bbox[2][1], marker='x', color='tab:blue', s=60)
#     ax.scatter(bbox[0][0], bbox[0][1], marker='x', color='tab:blue', s=60)
ax.axis('off')
plt.show()



# %%
fig, ax = plt.subplots(figsize=(8,6))
ax.set(
        xlim=[0, 512],
        ylim=[0, 384],
        xticks=list(range(0, 512,int(512/12))),
        yticks=list(range(0, 384, int(384/9))),
        # yticklabels=np.linspace(0, 9, 10),
        # xticklabels=np.linspace(0, 12, 13)
        )
ax.grid(visible=True, zorder=0)
# ax.add_patch(Polygon(boxes.label_box, fill=None, edgecolor='springgreen', lw=1, zorder=10))
# ax.add_patch(Polygon(boxes.pred_box, fill=None, edgecolor='tomato', lw=1, zorder=10))
ax.add_patch(Polygon(test_box, fill=None, edgecolor='tab:blue', lw=1, zorder=10))
ax.add_patch(Polygon(test_pred, fill=None, edgecolor='gray', lw=1, zorder=10))
ax.add_patch(Polygon(test_inter, facecolor='gray', edgecolor='gray', lw=1, alpha=.4, zorder=10))
# if boxes.intersection:
#     ax.add_patch(Polygon(boxes.intersection, edgecolor='tab:purple', zorder=10, alpha=.4))
# ax.add_patch(Polygon(get_Gbbox(boxes), fill=None, edgecolor='black', lw=1.5, zorder=10, alpha=.6))

# ax.scatter(i_x, i_y, marker='x', s=30, color='tab:purple')
ax.axhline(0, -10, 10, lw=.5, color='black', linestyle='--')
ax.axvline(0, -10, 10, lw=.5, color='black', linestyle='--')
plt.show()



# %%
print(type(labels), type(preds))


# %%
intersections = box_cutter.construct_intersection(label_corners, pred_corners)
print(intersections.shape)

# %%
areas = box_cutter.calculate_iou(label_corners, pred_corners)

# %%
print(f"Intersection Area: {areas.shape}\n{areas[0, 5, 6]}")

# %%
box = label_corners[0, 5, 8].numpy()
print(box.shape)
area = np.empty((4, 2), dtype=np.float32)
for i in range(4):
    area[i] = np.abs(box[-1 + i] - box[i])
area = np.sum(area, axis=0)
area = area[0] * area[1]
print(area)


# %%
true_edges = get_edges(label_corners)
pred_edges = get_edges(pred_corners)
print(pred_edges.shape)


# %%
label_1 = [4, 6, 1.5, 4, np.deg2rad(-13)]
pred = [3.2, 5.63, 1.2, 3.6, np.deg2rad(-25)]
boxes = bbox_worker(label_1, pred) 
boxes.label_box

# %%
# print(boxes.intersection)
test_label = label_corners[0, 5, 4]
print(test_label)
test_pred = pred_corners[0, 5, 4]
print(test_pred)


# %%
fig, ax = plt.subplots(figsize=(8,6))
ax.set(
        xlim=[0, 512],
        ylim=[0, 384],
        xticks=list(range(0, 512,int(512/12))),
        yticks=list(range(0, 384, int(384/9))),
        # yticklabels=np.linspace(0, 9, 10),
        # xticklabels=np.linspace(0, 12, 13)
        )
ax.grid(visible=True, zorder=0)
ax.add_patch(Polygon(test_label.numpy(), fill=None, edgecolor='springgreen', lw=1, zorder=10))
ax.add_patch(Polygon(test_pred.numpy(), fill=None, edgecolor='tomato', lw=1, zorder=10))
# if boxes.intersection:
#     ax.add_patch(Polygon(boxes.intersection, edgecolor='tab:purple', zorder=10, alpha=.4))
# ax.add_patch(Polygon(get_Gbbox(boxes), fill=None, edgecolor='black', lw=1.5, zorder=10, alpha=.6))

# ax.scatter(i_x, i_y, marker='x', s=30, color='tab:purple')
ax.axhline(0, -10, 10, lw=.5, color='black', linestyle='--')
ax.axvline(0, -10, 10, lw=.5, color='black', linestyle='--')
plt.show()


# %%
def get_intersections(edge1, edge2):
    edge_a = edge1[..., 2:3, :, :]
    edge_b = edge2[..., 0:, :, :]
    if self.debug:
        print(f"edge_a shape: {edge_a.shape}")
        print(f"edge_b shape: {edge_b.shape}")
        print(f"edge_a points:\n{tf.squeeze(edge_a[0,5, 4, 0:])}")
        print(f"edge_b points:\n{tf.squeeze(edge_b[0,5, 4, 0:])}")

    x1 = edge_a[..., 0:1, 0:1]
    y1 = edge_a[..., 0:1, 1:]
    x2 = edge_a[..., 1:, 0:1]
    y2 = edge_a[..., 1:, 1:]
    if self.debug:
        print(f"x1 shape: {x1.shape}")
        print(f"y1 shape: {y1.shape}")
        print(f"x1 value: {x1[0,5,4]}")
        print(f"y1 value: {y1[0,5,4]}")
        print(f"x2 value: {x2[0,5,4]}")
        print(f"y2 value: {y2[0,5,4]}")

    x3 = edge_b[..., 0:1, 0:1]
    y3 = edge_b[..., 0:1, 1:]
    x4 = edge_b[..., 1:, 0:1]
    y4 = edge_b[..., 1:, 1:]
    if self.debug:
        print(f"x3 value: {x3[0,5,4]}")
        print(f"y3 value: {y3[0,5,4]}")
        print(f"x4 value: {x4[0,5,4]}")
        print(f"y4 value: {y4[0,5,4]}")
    
    denom =  (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)

    print(f"denom shape: {denom.shape}")
    print(f"denom:\n{tf.squeeze(denom[0, 5, 4], axis=[-2, -1])}")

    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom

    zeros = tf.fill(tf.shape(ua), 0.0)
    ones = tf.fill(tf.shape(ua), 1.0)

    hi_pass_a = tf.math.less_equal(ua, ones)
    lo_pass_a = tf.math.greater_equal(ua, zeros)
    mask_a = tf.logical_and(hi_pass_a, lo_pass_a)

    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
    hi_pass_b = tf.math.less_equal(ub, ones)
    lo_pass_b = tf.math.greater_equal(ub, zeros)
    mask_b = tf.logical_and(hi_pass_b, lo_pass_b)

    mask = tf.logical_and(mask_a, mask_b)

    if self.debug:
        print(f"\nua value:       {tf.squeeze(ua[0, 5, 4], axis=[-2, -1])}")
        print(f"hi_pass value:  {tf.squeeze(hi_pass_a[0, 5, 4], axis=[-2, -1])}")
        print(f"lo_pass value:  {tf.squeeze(lo_pass_a[0, 5, 4], axis=[-2, -1])}")
        print(f"\nmask_a value:   {tf.squeeze(mask_a[0, 5, 4], axis=[-2, -1])}")
        print(f"\nub value:       {tf.squeeze(ub[0, 5, 4], axis=[-2, -1])}")
        print(f"hi_pass value:  {tf.squeeze(hi_pass_b[0, 5, 4], axis=[-2, -1])}")
        print(f"lo_pass value:  {tf.squeeze(lo_pass_b[0, 5, 4], axis=[-2, -1])}")
        print(f"\nmask_b value:   {tf.squeeze(mask_b[0, 5, 4], axis=[-2, -1])}")
        print(f"\nmask value:     {tf.squeeze(mask[0, 5, 4], axis=[-2])}")

    
    xnum = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
    ynum = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
    x_i = (xnum / denom)
    y_i = (ynum / denom)
    mask = tf.cast(tf.squeeze(mask, axis=[-1]), dtype=tf.float32)
    intersections = tf.multiply(tf.squeeze(tf.stack([x_i, y_i], axis=-3), axis=[-2, -1]), mask)
    if self.debug:
        print(f"mask_reshape: {mask.shape}")
        print(f"Intersection shape: {intersections.shape}")
        print(f"Intersection Points:\n{intersections[0, 5, 4]}")




get_intersections(true_edges, pred_edges)

# %%
value = np.array([[1, 1],
                  [2, 2],
                  [3, 3],
                  [4, 4],
                  [5, 5],
                  [6, 6],
                  [7, 7],
                  [8, 8]])

x = np.full((12, 9, 8, 2), value, dtype=np.float32)
tensor = tf.constant(x)
tile = np.random.randint(2, size=(12 * 9, 8))

# %%
r_tensor = tf.reshape(tensor, [12  * 9, 8, 2])
tile_2d = tf.transpose(tf.reshape(tf.tile(tile, [1, 2]), [12 * 9, 2, 8]), perm=[0, 2, 1])
mask = tf.sort(tile_2d, direction="DESCENDING", axis=-2)
print(mask.shape)
ma_tensor = tf.ragged.boolean_mask(r_tensor, tf.cast(mask, dtype=tf.bool))
ma_tensor = tf.roll(ma_tensor, shift=-1, axis=-2)
# tensor = tf.roll(tensor, shift=-1, axis=-2)
print(ma_tensor.shape)
print(ma_tensor[5:7].numpy())

