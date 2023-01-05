# %%
from src.utils.classes import CategoricalDataGen
from src.utils.funcs import *
from src.utils.box_cutter import BoundingBox_Processor

from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.animation import FuncAnimation, HTMLWriter, PillowWriter
from IPython.display import HTML

from sklearn.cluster import KMeans

# %%
# %load_ext autoreload

# %%
# %autoreload 2
# %aimport src.utils.box_cutter
# %aimport src.utils.funcs
# %aimport src.utils.classes

# %%
data = init_COCO("./data/", ["train", "val", "test"])

# %%
# Configuration
input_size = (1440, 1920)
target_size = (512, 512)
num_classes = 13

def convert_points(b_x, output_size=(512, 384)):
    adjX, adjY = output_size
    adjustment = np.array([1920, 1440])
    result = []
    b_x = b_x / adjustment
    b_x = tf.where(b_x > 0, b_x, 0).numpy()
    b_x = tf.where(b_x < 1, b_x, 1).numpy()
    box_tipped = np.sort(b_x.T, axis=-1)
    box_tipped = box_tipped.T
    max = box_tipped[-1:]
    min = box_tipped[0:1]
    w, h = np.squeeze(np.abs(np.diff(np.concatenate((max, min), axis=0).T, axis=-1)).T)
    x, y = np.squeeze(min)
    # result.append([y, x, h, w])  
    return y, x, h, w


# %%
annotations = []
for key in data.keys():
    print(f"Working on {key} data!")
    train_data = CategoricalDataGen(data_name=key, 
                                    coco_obj=data,
                                    image_path="./data/images/",
                                    target_size=target_size)

    img_data = train_data.lookup['img_data']
    boxes = train_data.lookup['annotations']
    for img in img_data:
        print(img['file_name'])
        file_name = img['file_name']
        id = img['id']
        h = img['height']
        w = img['width']
        for idx in boxes:
            if idx['image_id'] == id:
                bbox, _, _ = process_img_annotations(idx['bbox'])
                y, x, h, w = convert_points(np.asarray(bbox, dtype=np.float32))
                category = int(idx['category_id']) - 1
                annotations.append((f"./data/yolo_v3/obj/{file_name[:-3]}txt", f"{category} {y:.6f} {x:.6f} {h:.6f} {w:.6f}\n"))
                # print(annotations)

    # for img in img_data:
    #     with open(f"data/yolo_v3/{key}.txt", 'a') as file:
    #         file.write(f"data/obj/{img['file_name']}\n")

count = 0
for obj in annotations:
    with open(f'{obj[0]}', 'a') as file:
        file.write(obj[1])
    count += 1
print(f"Saved {count} files!")


# %%
print(annotations[0])
# test_set = [x[1].split() for x in annotations[:16]]
# test_set = [[float(y) for y in x[1:]] for x in test_set]
# test_set[0]

# %%
def get_em(data):
    result = np.empty((0,5), dtype=np.float32)
    for key in data.keys():
        print(f"Working on {key} data!")
        train_data = CategoricalDataGen(data_name=key, 
                                        coco_obj=data,
                                        image_path="./data/images/",
                                        target_size=target_size)

        img_data = train_data.lookup['img_data']
        boxes = train_data.lookup['annotations']
        for img in img_data:
            file_name = img['file_name']
            id = img['id']
            h = img['height']
            w = img['width']
            for idx in boxes:
                if idx['image_id'] == id:
                    bbox, _, _ = process_img_annotations(idx['bbox'])
                    bbox = np.asarray(convert_points(bbox), dtype=np.float32)
                    bbox = np.reshape(bbox, (1, 4))
                    cat = np.reshape(np.array([idx['category_id']]), (1, 1))
                    result = np.append(result, np.concatenate((cat, bbox), axis=-1), axis=0)
    return result

# %%
bboxes = get_em(data)
bboxes.shape

# %%
train_data = CategoricalDataGen(data_name="train", coco_obj=data, image_path="./data/images/", target_size=(384, 512))

# %%
training_labels = train_data.get_labels(divs=(9, 12), num_boxes=3, num_classes=13)

# %%
batch, xdiv, ydiv, _ = training_labels.shape
box_cutter = BoundingBox_Processor()

# %%
box_labels = tf.reshape(training_labels, [batch * xdiv * ydiv, 19])[..., 14:].numpy()
x, y = box_labels[:, 0:1], box_labels[:, :1]
w, h = box_labels[:, 2:3], box_labels[:, 3:4]
box_labels[:, 0:2] = [384/2, 512/2]
box_labels[:, 0:1] = x - w/2
box_labels[:, 1:2] = y - h/2
box_labels[:, 4:] = -1 * (box_labels[:, 4:] - np.pi)

mask_labels = tf.cast(tf.reduce_sum(box_labels, axis=-1) > .001, dtype=tf.bool)
box_labels = tf.boolean_mask(box_labels, mask_labels, axis=0).numpy()

clusters = KMeans(n_clusters=9, max_iter=100, random_state=1)
clusters.fit(box_labels)

cls = clusters.predict(box_labels)
cls = np.expand_dims(cls, axis=-1)

centroid_locations = []
for idx in range(13):
    filter = tf.where(tf.equal(cls, idx),
                             box_labels,
                             np.zeros(box_labels.shape, dtype=np.float32))
    mask = tf.cast(tf.abs(tf.reduce_sum(filter, axis=-1)) > .001, dtype=tf.float32)
    average = tf.reduce_sum(filter, axis=0) / tf.reduce_sum(mask, axis=0)
    centroid_locations.append(average.numpy())


# %%
plt.close()
fig, ax = plt.subplots(figsize=(4, 3))
ax.set(
        xlim=[0, 512],
        ylim=[0, 384]
        )
pick = np.random.choice
color = ["tomato", "springgreen", "orange", "deepskyblue", "tab:purple"]
knudge = 512 / 12
start = np.array([knudge/2, knudge/2])

def animation(i):
    ax.axis('off')
    v, u = divmod(i, 12)
    bump = np.array([u * knudge, v * knudge]) 
    x, y = start + bump
    ax.add_patch(Circle((x, y), 2, fill=True, facecolor='gray', alpha=.6))
    anchors = np.random.choice(range(0, 12, 1), 4)
    for idx in anchors:
        _, _, w, h, a = centroid_locations[idx]
    # for box in centroid_locations:
    #     _, _, w, h, a = box
        ax.add_patch(Rectangle((x-w/2, y-h/2), w, h, angle=a*180/np.pi, rotation_point="center", fill=None, lw=.5, edgecolor=pick(color)))
ax.axis('off')
pillow = PillowWriter(fps=24)
anim = FuncAnimation(fig, animation, frames=9 * 12, cache_frame_data=False)
anim.save("./images/anchor_anim.gif", writer=pillow)
HTML(anim.to_jshtml(fps=24))
plt.close()

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



