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
from src.utils.box_cutter import BoundingBox_Processor



# %%
# %load_ext autoreload

# %%
# %autoreload 2
# %aimport src.utils.box_cutter
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
true_edges = box_cutter.get_edges(label_corners)
pred_edges = box_cutter.get_edges(pred_corners)
print(pred_edges.shape)

# %%
intersection_points = box_cutter.construct_intersection(label_corners, pred_corners, return_centered=False)

# %%
iou = box_cutter.calculate_iou(label_corners, pred_corners)
print(f"IoU: {iou.shape}\n{iou[0, 5]}")

# %%
giou = box_cutter.calculate_GIoU(label_corners, pred_corners)
giou[0, 5]

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
test_box = label_corners[0, 5, 6]
test_pred = pred_corners[0, 5, 6]
test_inter = tf.constant(intersection_points[0, 5, 6, :6])

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
ax.add_patch(Polygon(test_box, fill=None, edgecolor='tab:blue', lw=1, zorder=10))
ax.add_patch(Polygon(test_pred, fill=None, edgecolor='gray', lw=1, zorder=10))
ax.add_patch(Polygon(test_inter, facecolor='gray', edgecolor='gray', lw=1, alpha=.4, zorder=10))
ax.axhline(0, -10, 10, lw=.5, color='black', linestyle='--')
ax.axvline(0, -10, 10, lw=.5, color='black', linestyle='--')

