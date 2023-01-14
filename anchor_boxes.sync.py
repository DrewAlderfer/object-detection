# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Rectangle, Polygon, Circle
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PolyCollection
import matplotlib.colors as mcolors

import numpy as np
from pycocotools import coco
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import Loss
from tensorflow.keras.utils import load_img
import pprint as pp

from src.utils.box_cutter import BoundingBox_Processor
from src.utils.classes import CategoricalDataGen
from src.utils.data_worker import LabelWorker
from src.utils.funcs import init_COCO, make_predictions, generate_anchors, translate_points, display_label

# %%
# %load_ext autoreload

# %%
# %autoreload 2
# %aimport src.utils.funcs
# %aimport src.utils.classes
# %aimport src.utils.data_worker

# %%
pp.PrettyPrinter(indent=4)

# %%
data = init_COCO("./data/", ['train', 'val', 'test'])
box_cutter = BoundingBox_Processor()

# %%
train = CategoricalDataGen(data_name='train',
                           coco_obj=data,
                           image_path='./data/images/',
                           target_size=(384, 512))

# %%
labeler = LabelWorker(data_name='train',
                      coco_obj=data,
                      image_path='./data/images/',
                      input_size=(1440, 1920),
                      target_size=(384, 512))

# %%
labels = labeler.label_list()
rowsum = labels[0].sum(-1)
labels_d = labels[0, rowsum > 0]
corners = box_cutter.get_corners(labels)[0]
bboxes = corners[0].numpy().reshape((12*9, 4, 2))
rs = bboxes.sum(-1).sum(-1)
bboxes = bboxes[rs > 0]
bboxes


# %%
arr = np.asarray(np.fromfunction(lambda x, y: [x, y], (12, 9), dtype=np.float32), dtype=np.float32)
dots = (np.stack((arr[0], arr[1]), axis=-1) + .5) / np.array([12, 9])
scatter_dots = dots.reshape(12 * 9, 2).T
dot_x, dot_y = scatter_dots[0] * 512, scatter_dots[1] * 384


# %%
anchors = generate_anchors(labels, random_state=1)
anchor_tensor = np.full((batch,) + anchors.shape, anchors)
a_boxes = box_cutter.get_corners(anchor_tensor)
anchor_tensor.shape

# %%
# Okay I am realizing that I was probably being too literal again in the way I was processing labels.
# There really isn't any reason they need to be split up into these matrices where the cell relates
# to their location on the image. 
# 
# If I am testing each anchor box against each ground truth and then just returning the max values
# matching for each, then there isn't any need for this structure. In fact the structre should be
# more like lists of vectors, and all that really matters is that the values of each vector
# coorespond to a cell in the prediction output.
#
# I keep running into this thing where it's not clear to me that something like grid cells do not
# need to be represented in a literal way. That in the end it's just a way to think about the process
# and not really a literal process with physical relationships, it's just all represented abstractly
# through the data .duh, I mean I guess what I mean is that the layer of abstraction I'm working at
# is too literal. Having a list of 108 vectors is the same as having a matrix of vectors in a 12 x 9
# grid.
#
# So I need to rewrite a lot of the IoU and label/box_cutter stuff. But it probably wont end up being
# that much. We'll see... maybe there is an interesting way to line these things up and do the calculations
# in tensors.

a_box = anchor_tensor[..., :19]
y_box = labels
print(a_box.shape, y_box.shape)



# %%
fig, ax = plt.subplots(figsize=(8,6))
ax.set(
        ylim=[0, 384],
        xlim=[0, 512],
        xticks=list(range(0, 512,int(np.ceil(512/12)))),
        yticks=list(range(0, 384, int(np.ceil(384/9)))),
        )
ax.grid(visible=True, zorder=0)
ax.set_title("Bounding Boxes w/ Orientation")
ax.imshow(np.asarray(load_img("./data/images/train/screws_002.png", target_size=(384, 512))), zorder=0, alpha=.6)
pick = np.random.choice
colors = list(mcolors.TABLEAU_COLORS.values())
for label in labels_d:
    print(label[14:])
    color = pick(colors)
    bbox, arrow = display_label(label[14:], ax, color)
    ax.add_patch(bbox)
    ax.add_patch(arrow)
for i in range(3):
    ax.add_patch(Polygon(box_cutter.get_corners(labels)[0][0, 5, 6], fill=None, edgecolor='chartreuse', lw=1, zorder=20))
    ax.add_patch(Polygon(a_boxes[i][0,5,6], fill=None, edgecolor='fuchsia', lw=1, zorder=20))
#     ax.add_patch(Polygon(test_box_corners, fill=None, edgecolor='tab:blue', lw=1, zorder=20))
#     display_label(labels)
    # ax.add_patch(Rectangle(*test_box2[:3], angle=test_box2[-1], rotation_point="center", fill=None, edgecolor="tab:red", ls="--", zorder=200))
    # ax.add_patch(Arrow(x, y, vec_x_mag, vec_y_mag, width=20, color='springgreen', zorder=100))
# for i in range(12 * 9):
#     ax.add_patch(Circle(dots.reshape((12 * 9, 2))[i] * np.array([512, 384]), radius=3))
ax.scatter(dot_x, dot_y, marker='x', color='tomato', alpha=.6)
ax.axis('off')
plt.savefig("./images/bbox_intersection.png")
plt.show()

# %%
print(f"n_labels: {n_labels[0, 5, 6]}")
print(f"anchor_tensor: {anchor_tensor[0, 5, 6]}")

# %%
int(divmod(.5, 1/12)[0])
int(divmod(.65, 1/9)[0])

# %%
class AnchorLayer(Layer):
    def __init__(self, *, units, classes, boxes, anchors):
        super(AnchorLayer, self).__init__()
        self.units = units
        self.classes = classes
        self.boxes = boxes
        self.anchors = anchors
        self.box_cutter = BoundingBox_Processor()

    def build(self, input_shape):
        self.w = self.add_weight(
                shape=input_shape[1:-1] + (self.units * self.boxes,),
                initializer="random_normal",
                trainable=True,
                )
        self.b = self.add_weight(
                shape=(self.boxes * self.units,),
                initializer="random_normal",
                trainable=True,
                )

    def call(self, inputs):
        x = inputs[..., 13:]
        for i in range(self.boxes):
            anchor = anchors[..., self.boxes*i:self.boxes + i * self.boxes])
            self.box_cutter.IoU(x, anchor)


# %%
anchor_layer = AnchorLayer(units=6, classes=13, boxes=3, anchors=None)
w, b = anchor_layer(labels)
print(f"w: {w.shape} | b: {b.shape}")

# %%
rng = np.random.default_rng()
a1 = rng.random((5, 5), dtype=np.float32)
a2 = rng.random((7, 5), dtype=np.float32)
a3 = np.zeros((24 - a1.shape[0],) + (5,), dtype=np.float32)
A = np.concatenate((a1, a3), axis=0)
A
