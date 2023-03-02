# %%
import os
import cv2 as cv
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import tensorflow as tf

from src.data_worker import YOLODataset, init_COCO, DarknetTools
from src.utils import *
from src.disviz import setup_labels_plot

# %% [markdown]
# # Data Pipe for Background (Empty) Images
# This notebook contains the code that I wrote to generate background images from photos of OSB board
# that I took from the internet:
# ### Step 1
# - Takes high resolution images and measures them against the target size, 576 x 768 (h x w).
# - Resizes and Splits the image into equal parts.
# - Outputs a Tensor object containing the generated background images.
# ### Step 2
# - Performs some detection logic on the train dataset images that already exist
# - Saves each image in the tensor to the train images directory
# ### Step 3
# - Creates dictionary updates for the new images added to the train images directory
# - Inserts updates into the dataset object for training data
# - Updates the project directory by passing the new information through the `DarknetTools` Pipeline object.

# %%
np.set_printoptions(suppress=True, precision=4)
# %load_ext autoreload

# %%
# %autoreload 2
# %aimport src.utils
# %aimport src.disviz
# %aimport src.data_worker

# %% [markdown]
# ## Step 1
# ### Get the base images from which to generate backgrounds.

# %%
images = glob('./data/images/empty/*')
images = [cv.imread(img) for img in images]

# %% [markdown]
# #### Testing some of the logic for our sample function.

# %%
tar_size = [768, 576]
image = images[3]
image_area = np.cumsum(image.shape[:2])[-1]
tar_area = np.cumsum(tar_size)[-1]
result = image_area / tar_area

print(f"image: {image.shape}, {image.dtype}")
print(f"area_div: {result:.2f}")

# %% [markdown]
# ### The background sampling function:

# %%
def bg_rand_sample(images, target_size):
    result = np.empty(shape=(0, 576, 768, 3), dtype=np.float32)
    for image in images:
        div = np.true_divide(np.array([image.shape[:-1]], dtype=np.int16),
                             np.array(target_size, dtype=np.int16), 
                             dtype=np.float32)
        div = np.array(np.around(div)[0], dtype=np.int32)
        new_size = np.array(div * target_size, dtype=np.int32)
        img = cv.resize(image, np.flip(new_size))
        img = np.asarray(np.split(img, div[0]))
        img = np.concatenate(np.split(img, div[1], axis=2), axis=0)
        if len(images) > 1:
            result = np.concatenate([result, img], axis=0)
        else:
            result = img
    return result


# %% [markdown]
# ### Visualizing the result from our function.

# %%
result = bg_rand_sample([images[9]], target_size=[576, 768])
print(f"result: {result.shape}")


# %%
def rowcol(data, base_size=(8, 6)):
    # get the number of images
    axes = data.shape[0]

    # calculate the cols (images per row) if displayed in a square grid
    cols = np.floor(np.sqrt(axes)).astype(np.int32)

    # calculate the rows by how many cols fit
    # if there is a remainder add one row
    rows, rem = np.divmod(axes, cols, dtype=np.int32)
    rows = rows + bool(rem) 

    ratio = base_size[1] / base_size[0]
    size = (base_size[0], ((base_size[0]/cols) * ratio ) * rows)
    
    return rows, cols, size


# %%
row, col, size = rowcol(result)
fig, ax = plt.subplots(row, col, figsize=size)
ax = ax.flat
for i in range(len(result)):
    ax[i].imshow(np.flip(result[i], axis=-1))
    ax[i].axis('off')
plt.show()

# %% [markdown]
# ## Step 2
# ### Draw the background samples

# %%
backgrounds = bg_rand_sample(images, target_size=[576, 768])
print(backgrounds.shape)

# %% [markdown]
# ### Make a sorted list of all the image numbers currently in the train images directory

# %%
screws_num = sorted([int(x[-7:-4]) for x in glob('./data/images/train/screws_*.png')])
print(screws_num[-1])

# %% [markdown]
# ### Save the background images to the train images directory
# Some of the numbering here is a little extra. Before going through the whole process I wasn't
# sure how the naming convention might affect data processing downstream, so there is a lot of
# extra caution built in to avoid overwriting or conflicting with other data.

# %%
count = screws_num[-1] + 10
for image in backgrounds:
    print(f"writing file {count:03d}")
    cv.imwrite(f'./data/images/train/screwsbg_{count:3d}.png', image)
    count += 1

# %% [markdown]
# ## Step 3
# ### Initialize the dataset processing pipeline

# %%
data = init_COCO("./data/", ['train', 'val', 'test'])

# %% [markdown]
# ### Create a list of ids for each image in the training dataset

# %%
train_ids = list(range(1, len(glob('./data/images/train/*.png')) + 1, 1))
print(f"train_ids:\n{train_ids[:10]}\n...\n{train_ids[-10:]}")

# %%
train_ids[-len(bg_imgs)]

# %% [markdown]
# ### Taking a look at the json data descriptions used to create image annotations

# %%
print(f"data: {data['train'].keys()}")
print(f"annotation_sample:")
data['train']['annotations'][-1], data['train']['img_data'][-1]

# %% [markdown]
# ### Create a sorted list of all the background images in the train images directory

# %%
bg_imgs = sorted([os.path.basename(x) for x in glob("./data/images/train/screwsbg_*.png")])
print(f"bg_imgs: {len(bg_imgs)}")

# %% [markdown]
# ### Create the list of dictionary entries and add them to the COCO train dataset object

# %%
h = 576
w = 768
update = []
for id, img in zip(train_ids[-len(bg_imgs):], bg_imgs):
    update.append({'file_name': img,
                   'height': h,
                   'width': w,
                   'id': id,
                   'license': 1})
data['train']['img_data'].extend(update)

# %% [markdown]
# ### Initialize the Dataset `Sequence` objects

# %%
train_dataset = YOLODataset(data_name='train',
                            coco_obj=data,
                            image_path='./data/images/',
                            input_size=(1440, 1920),
                            target_size=(576, 768))

# %%
val_dataset = YOLODataset(data_name='val',
                            coco_obj=data,
                            image_path='./data/images/',
                            input_size=(1440, 1920),
                            target_size=(576, 768))

# %%
test_dataset = YOLODataset(data_name='test',
                            coco_obj=data,
                            image_path='./data/images/',
                            input_size=(1440, 1920),
                            target_size=(576, 768))

# %% [markdown]
# ### Initialize my Darknet data pipe class

# %%
darknet = DarknetTools(data=[train_dataset, val_dataset, test_dataset], image_size=[768, 576], project_directory='./darknet_yolo', make=True)

# %% [markdown]
# ## Output the updated annotations for darknet training.

# %%
darknet.save_annotations()

# %% [markdown]
# ### Generate Anchors Based on the darknet labels

# %%
from sklearn.cluster import KMeans
def get_darknet_anchors(box_labels, boxes_per_cell:int=9, target_size=[768, 576], **kwargs):

    mask_labels = tf.cast(tf.reduce_sum(box_labels, axis=-1) > .001, dtype=tf.bool)
    box_labels = tf.boolean_mask(box_labels, mask_labels, axis=0).numpy()

    clusters = KMeans(n_clusters=boxes_per_cell, max_iter=100, **kwargs)
    clusters.fit(box_labels)

    centers = clusters.cluster_centers_ * np.array([target_size], dtype=np.float32)
    c_areas = np.prod(centers, axis=-1)
    idx_srt = np.argsort(c_areas)
   
    return centers[idx_srt].astype(np.int32)

result = get_darknet_anchors(darknet_labels[..., 3:], 6, random_state=42)
np.savetxt('./darknet_yolo/anchors.txt', result, fmt='%-d', delimiter=',', newline=',  ')

# %% [markdown]
# ### Resize images

# %%
darknet_images = glob('./darknet_yolo/images/screws_*.png')
for img in darknet_images:
    png = cv.imread(img)
    png = cv.resize(png, [768, 576])
    cv.imwrite(img, png)

# %% [markdown]
# # Finished
# That's it. There is now a full set of background images included in the dataset to use for training
# darknet. Make some tar balls, throw them in colab and have fun!
