# %%
from itertools import cycle
from glob import glob
from typing import Union, Tuple, List
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Rectangle, Polygon, Circle, PathPatch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PatchCollection, PolyCollection, LineCollection
import matplotlib.colors as mcolors
from IPython.display import HTML

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pandas.core.window import rolling
from pycocotools import coco
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import Loss
from tensorflow.keras.utils import load_img
import pprint as pp

from src.utils.box_cutter import BoundingBox_Processor
from src.utils.classes import CategoricalDataGen
from src.utils.data_worker import LabelWorker, init_COCO
from src.utils.funcs import *
from src.utils.disviz import *

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
anchor_edges[0, an::108].shape

# %%
old_corners = box_cutter.get_corners(labels)[0]
old_anchors = box_cutter.get_corners(anchors)[0]
print(f"old_corners: {old_corners.shape}")
print(f"old_anchros: {old_anchors.shape}")
old_intersections = box_cutter.rolling_intersection(old_corners, old_anchors)[0]
old_intersections.shape

# %%
x_points = construct_intersection_vertices(labels, anchors, num_pumps=num_anchors)
print(f"x_points: {x_points.shape}")

# %%
intersection = intersection_area(x_points)
union = union_area(label_corners, anchor_corners, intersection, num_pumps=num_anchors)
print(f"areas: {intersection.shape}, {intersection.dtype}")
print(f"areas:\n{intersection[0, bb, an::108]}")
print(f"union: {union.shape}, {union.dtype}")
print(f"union:\n{union[0, bb, an::108]}")

# %%
giou = calculate_giou(label_corners, anchor_corners, union, intersection)
print(f"giou: {giou[img, bb, an::108]}")

# %%
iou = intersection / union
triangles, int_edges, tri_areas = intersection_shapes(labels, anchors, num_pumps=num_anchors)
print(f"IoU: {iou.shape}, {iou.dtype}")
print(f"IoU:\n{iou[0, bb, an::108]}")
print(f"triangles: {triangles.shape}, {triangles.dtype}")
print(f"int_edges: {int_edges.shape}, {int_edges.dtype}")

# %%
# Select a set of bounding boxes and anchors to display
def get_coords(img, bb, x, y, num):
    an = x * 9 + y + 108 * num
    return img, bb, an

# %%
fig, axs = plt.subplots(figsize=(8, 6))
axs.set(
        ylim=[-40, 384+40],
        xlim=[-40, 512+40],
        xticks=list(range(0, 512,int(np.ceil(512/12)))),
        yticks=list(range(0, 384, int(np.ceil(384/9)))),
        )
lines = []
verts = []
xpoints = []
ypoints = []
an_num = x_points[img, bb, an::108].shape[0]
for x in range(an_num):
#     print(x, an + x * 108)
    for i in range(x_points.shape[-2]):
        point = x_points[img, bb, an + x * 108, i]
        if point[0] == 0:
            continue
        xpoints.append(point[0])
        ypoints.append(point[1])
for i in range(0, 13, 1):
    line = i * 512/12
    lines.append([(line, 0), (line, 384)])
for i in range(0, 10, 1):
    line = i * 384/9
    lines.append([(0, line), (512, line)])
grid_lines = mpl.collections.LineCollection(lines, colors='black', lw=1, alpha=.4, zorder=200)
axs.add_collection(grid_lines)
axs.add_collection(mpl.collections.LineCollection(label_edges[img, bb]))
axs.add_collection(mpl.collections.LineCollection(anchor_edges[img, an::108].numpy().reshape(14,4,2), lw=1, color="springgreen"))
axs.scatter(xpoints, ypoints, color="tomato", marker="o", s=5, lw=1, zorder=250)
axs.axis('off')
plt.show()

# %%
plt.close()
img, bb, an = get_coords(0, 7, 6, 1, 5)
print(f"img: {img} | bb: {bb} | an: {an} (anchor number = {divmod(an, 108)[0] + 1})")
fig, axs = set_plot(img, bb, an, label_corners, anchor_corners, padding=20)
axs.add_collection(mpl.collections.LineCollection(label_edges[img, bb], lw=1))
axs.add_collection(mpl.collections.LineCollection(anchor_edges[img, an].numpy(), lw=1, color="springgreen"))
tri_shape, count = triangle_shapes(triangles, img, bb, an)
gbox = dis_Gbox(label_corners, anchor_corners, img, bb, an)
sum_block_starts, block_starts, area_sum, area_len = display_area_addition(axs, tri_areas, gbox, img, bb, an)

def animation(x):
    tri_colors = []
    for i in range(count):
        if i == x:
            tri_colors.append("lightskyblue")
            continue
        tri_colors.append("dodgerblue")
    tri_shape.set(facecolors=tri_colors,
                  edgecolors=tri_colors)
    axs.add_patch(Rectangle(sum_block_starts[x], area_sum[x], area_sum[x], facecolor="springgreen", zorder=200))
    axs.add_patch(Rectangle(block_starts[x], area_len[x], area_len[x], facecolor='dodgerblue', zorder=200)) 
    axs.add_collection(tri_shape)

axs.add_patch(gbox)
x, y = mark_points(x_points, img, bb, an)
axs.scatter(x, y, color="tomato", marker="o", s=5, lw=1, zorder=250)
axs.axis('off')
axs.set_title("Calculating Intersection Example")
anim = FuncAnimation(fig, animation, frames=list(range(count)))
image_name = glob("./images/triangle_anim_**.gif")
num = len(image_name) + 1
image_name = image_name[-1][:-7] + f"{num:03d}.gif"
print(f"saving animations to:\n{image_name}")
anim.save(image_name, fps=1.5)
plt.show()

# %%
fig, axes = plt.subplots(3, 3, figsize=(12, 9))
axs = []
for i in range(3):
    axs.extend(axes[i])
def animation(frame):
    for i, ax in enumerate(axs):
        img, bb, an = get_coords(0, 7, 6, 1, i)
        ax.add_collection(mpl.collections.LineCollection(label_edges[img, bb], lw=1))
        ax.add_collection(mpl.collections.LineCollection(anchor_edges[img, an].numpy(), lw=1, color="springgreen"))
        set_ax(ax, img, bb, an, label_corners, anchor_corners)
        x, y = mark_points(x_points, img, bb, an)
        tri_shape, count = triangle_shapes(triangles, img, bb, an)
        gbox = dis_Gbox(label_corners, anchor_corners, img, bb, an)
        sum_block_starts, block_starts, area_sum, area_len = display_area_addition(ax, tri_areas, gbox, img, bb, an)

        tri_colors = []

        for num in range(count):
            print(f"num: {num} | frame: {frame}")
            if num == frame:
                tri_colors.append("lightskyblue")
                continue
            tri_colors.append("dodgerblue")
        tri_shape.set(facecolors=tri_colors,
                      edgecolors=tri_colors)
        for a in range(count):
            if a == frame:
                print(f"a: {a}")
                ax.add_patch(Rectangle(sum_block_starts[a], area_sum[a], area_sum[a], facecolor="springgreen", zorder=200))
                ax.add_patch(Rectangle(block_starts[a], area_len[a], area_len[a], facecolor='dodgerblue', zorder=200)) 
        ax.add_collection(tri_shape)

        ax.add_patch(gbox)
        ax.tick_params(
                axis='both',
                which='both',
                labelbottom=False,
                labelleft=False,
                bottom=False,
                left=False
                )
        ax.scatter(x, y, color="tomato", marker="o", s=5, lw=1, zorder=250)

anim = FuncAnimation(fig, animation, frames=list(range(8)))
image_name = glob("./images/triangle_anim_**.gif")
num = len(image_name) + 1
image_name = image_name[-1][:-7] + f"{num:03d}.gif"
print(f"saving animations to:\n{image_name}")
anim.save(image_name, fps=1.5)

plt.show()

# %%
fig, axs = plt.subplots(2, 1, figsize=(8, 10))
# axs = np.concatenate([ax1, ax2], axis=-1)
for img, axs in enumerate(axs):
    lines = []
    for i in range(1, 12, 1):
        line = i * 512/12
        lines.append([(line, 0), (line, 384)])
    for i in range(1, 9, 1):
        line = i * 384/9
        lines.append([(0, line), (512, line)])
    grid_lines = mpl.collections.LineCollection(lines, colors='black', lw=1, ls='--', alpha=.4)
    axs.set(
            ylim=[0, 384],
            xlim=[0, 512],
            xticks=list(range(0, 512,int(np.ceil(512/12)))),
            yticks=list(range(0, 384, int(np.ceil(384/9)))),
            )
    axs.axis('off')
    axs.imshow(load_img(images[img], target_size=(384, 512)))
    axs.add_collection(grid_lines)
    for idx, box in enumerate(label_corners[img]):
        n_color = next(color)
        bbox, arrow = display_label(labels[img, idx, 14:], color=n_color)
        axs.add_patch(bbox)
        axs.add_patch(arrow)
        # ax.add_patch(Polygon(label_corners.reshape(269, 12 * 9, 4, 2)[img, idx], fill=None, edgecolor="tomato", lw=5))
fig.tight_layout()
plt.savefig("./images/bounding_box_examples_2.png")
plt.show()

# %%
fig, [ax1, ax2, ax3, ax4] = plt.subplots(4, 3, figsize=(12, 12))
axs = np.concatenate([ax1, ax2, ax3, ax4], axis=-1)
for img, axs in enumerate(axs):
    lines = []
    for i in range(1, 12, 1):
        line = i * 512/12
        lines.append([(line, 0), (line, 384)])
    for i in range(1, 9, 1):
        line = i * 384/9
        lines.append([(0, line), (512, line)])
    grid_lines = mpl.collections.LineCollection(lines, colors='black', lw=1, ls='--', alpha=.4)
    axs.set(
            ylim=[0, 384],
            xlim=[0, 512],
            xticks=list(range(0, 512,int(np.ceil(512/12)))),
            yticks=list(range(0, 384, int(np.ceil(384/9)))),
            )
    axs.axis('off')
    axs.imshow(load_img(images[img], target_size=(384, 512)))
    axs.add_collection(grid_lines)
    for idx, box in enumerate(label_corners[img]):
        n_color = next(color)
        bbox, arrow = display_label(labels[img, idx, 14:], color=n_color)
        axs.add_patch(bbox)
        axs.add_patch(arrow)
        # ax.add_patch(Polygon(box, fill=None, edgecolor="tomato", lw=5))
fig.tight_layout()
plt.savefig("./images/bounding_box_examples_16.png")
plt.show()
