# %%
import pprint as pp

import matplotlib as mpl
from matplotlib.animation import FuncAnimation, PillowWriter
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

from IPython.display import HTML



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
# y_pred = pred_corners[0]
# y_pred = tf.reshape(y_pred, [ydivs * xdivs, 4, 2])
# for pred in y_pred:
#     bbox = pred
#     ax.add_patch(Polygon(bbox, fill=None, edgecolor='tomato', lw=2))
#     ax.add_patch(Circle(np.mean(bbox, axis=0), radius=3, fill=None, edgecolor="tab:blue", lw=2))
#     ax.scatter(bbox[2][0], bbox[2][1], marker='x', color='tab:blue', s=60)
#     ax.scatter(bbox[0][0], bbox[0][1], marker='x', color='tab:blue', s=60)
ax.axis('off')
plt.savefig('./screws_with_label_boxes.png')
plt.show()


# %%
print(label_corners[0, 5, 6].shape,label_corners[0, 5, 6].shape)
test_box = label_corners[0, :, :]
test_pred = pred_corners[0, :, :]
test_inter = tf.constant(intersection_points[0, :, :, :8])

# %%
for idx in range(6):

    break

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
ax.set_title("Example of Calculated Shape Intersection")
ax.imshow(np.asarray(load_img("./data/images/test/screws_006.png", target_size=(384, 512))), zorder=0, alpha=.6)
ax.set_facecolor('black')
for cell in range(12 * 9):
    cl = divmod(cell, 12) 
    print(cl[1], cl[0])
    check = tf.cast(tf.reduce_sum(test_box[cl[1], cl[0]]) < .001, dtype=tf.bool).numpy()
    if check:
        continue
    mask = tf.cast(tf.reduce_sum(test_inter[cl[1], cl[0]], axis=-1) > 0.001, dtype=tf.bool)
    test_inter_ma = tf.boolean_mask(test_inter[cl[1], cl[0]], mask)
    ax.add_patch(Polygon(test_box[cl[1], cl[0]], fill=None, edgecolor='chartreuse', lw=1, zorder=20))
    ax.add_patch(Polygon(test_pred[cl[1], cl[0]], fill=None, edgecolor='fuchsia', lw=1, zorder=20))
    ax.add_patch(Polygon(test_inter_ma, facecolor='palegreen', edgecolor='springgreen', lw=1, alpha=.4, zorder=10))
ax.axhline(0, -10, 10, lw=1, color='black', linestyle='--', zorder=5)
ax.axvline(0, -10, 10, lw=1, color='black', linestyle='--', zorder=5)
plt.savefig("./images/bbox_intersection.png")
plt.show()

# %%
mpl.rcParams['animation.embed_limit'] = 400
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
ax.set_title("Example of Calculated Shape Intersection")
ax.set_facecolor('black')
ax.imshow(np.asarray(load_img("./data/images/test/screws_006.png", target_size=(384, 512))), zorder=0, alpha=.6)
def show_frame(frame):
    for cell in range(0 + frame, 9 * 12, 1):
        cl = divmod(cell, 12) 
        check = tf.cast(tf.reduce_sum(test_box[cl[1], cl[0]]) < .001, dtype=tf.bool).numpy()
        if check:
            return None, timer 
        y_true = Polygon(test_box[cl[1], cl[0]], fill=None, edgecolor='chartreuse', lw=1, zorder=20)
        y_pred = Polygon(test_pred[cl[1], cl[0]], fill=None, edgecolor='fuchsia', lw=1, zorder=20)
        intersection = Polygon(test_inter_ma, facecolor='palegreen', edgecolor='springgreen', lw=1, alpha=.4, zorder=10)
        timer = frame
        return True, timer,   
        

def animation(i):
    check,  = show_frame(i)
    if not check:
        return
    mask = tf.cast(tf.reduce_sum(test_inter[cl[1], cl[0]], axis=-1) > 0.001, dtype=tf.bool)
    test_inter_ma = tf.boolean_mask(test_inter[cl[1], cl[0]], mask)
    ax.add_patch(y_true)
    ax.add_patch()
    ax.add_patch()

pillow = PillowWriter(fps=15)
ax.axhline(0, -10, 10, lw=1, color='black', linestyle='--', zorder=5)
ax.axvline(0, -10, 10, lw=1, color='black', linestyle='--', zorder=5)
anim = FuncAnimation(fig, animation, frames=9*12)
# anim.save("./images/intersection_anim.gif", writer=pillow, dpi=100)
HTML(anim.to_jshtml(fps=15))
# plt.close()

# %%
points = []
knudge = 512 / 12
start = np.array([knudge/2, knudge/2])
for cell in range(12 * 9):
    v, u = divmod(cell, 12)
    points.append(start + [u*knudge, v*knudge])
    x, y = [[x for x, y in points],
            [y for x, y in points]]
    print(x[:10])
    plt.scatter(x, y)

