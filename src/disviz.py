import numpy as np
import tensorflow as tf
from tensorflow.types.experimental import TensorLike
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Rectangle, Circle, Polygon
from matplotlib.collections import PatchCollection, PathCollection, LineCollection

def format_layers_for_display(activations:np.ndarray, layer_names:list, images_per_row:int=16):
    """
    Function that takes a list of conv_layer activations and returns a tuple of objects to display them.
    """
    import numpy as np
    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros(((size + 1) * n_cols - 1,
                                images_per_row * (size + 1) - 1))
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_index = col * images_per_row + row
                channel_image = layer_activation[0, :, :, channel_index].copy() 

                if channel_image.sum() != 0:
                    channel_image -= channel_image.mean()
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype("uint8")
                display_grid[col * (size + 1): (col + 1) * size + col,
                             row * (size + 1) : (row + 1) * size + row] = channel_image
        scale = 1. / size
        yield display_grid, scale, layer_name

def display_label(label, color, target_size=(384, 512)):
    """
    Function:
        Takes a Label [x, y, w, h, angle] and an Axis object, with the image (h, w) and the plot (h, w).

    Returns:
    
        A tuple of mpl.axis objects. One contains the bounding box and the other contains the an 
        arrow showing the orientation and directoin of the object.
    """
    from matplotlib.patches import Arrow, Polygon
    from .funcs import get_corners
    x, y, w, h, phi = label
    phi = phi + np.pi / 2
    x, y = x * target_size[1], y * target_size[0]
    w, h = w * target_size[1], h * target_size[1]
    # vec_x_mag, vec_y_mag = ((h / 2) * np.cos(phi)), ((h / 2) * np.sin(phi))
    vec_x_mag, vec_y_mag = ((w / 2) * np.sin(phi)), ((w / 2) * np.cos(phi))

    label_tensor = np.ones((1, 19), dtype=np.float32)
    t_label = label # translate_points(label.tolist(), input_size, target_size)
    label_tensor[..., 14:] = label_tensor[..., 14:] * t_label
    test_box_corners = np.squeeze(get_corners(label_tensor), axis=0)

    bbox = Polygon(test_box_corners, fill=None, edgecolor=color, lw=1, zorder=20)
    arrow = Arrow(x, y, vec_x_mag, vec_y_mag, width=15, color=color, zorder=100)

    return bbox, arrow

def intersection_shapes(labels:TensorLike,
                        predictions:TensorLike,
                        **kwargs) -> tuple[TensorLike, TensorLike]:
    """
    Function that takes a set of intersection points and returns a set of the areas.
    """
    from .funcs import construct_intersection_vertices
    intersection_points = construct_intersection_vertices(labels, predictions, **kwargs)
    nonzero = tf.cast(intersection_points > 0, dtype=tf.float32)
    mask = tf.cast(intersection_points > 0, dtype=tf.bool)

    denomenator = tf.reduce_sum(nonzero, axis=-2)
    center = tf.expand_dims(tf.reduce_sum(intersection_points, axis=-2) / denomenator, axis=-2)
    center_adj_points = tf.where(mask, 
                                 intersection_points - center,
                                 nonzero)
    # Get the angles of each point in the intersection to the center point of the intersection shape.
    angles = tf.math.atan2(center_adj_points[..., :, 1:], center_adj_points[..., :, 0:1]) 
    point_angle = tf.where(mask, 
                           tf.add(angles, tf.constant(np.pi, dtype=tf.float32, shape=(1,))),
                           nonzero)
    # order the points by their angle. The transpose here was necessary reordering the points. So
    # for a few steps the x and y values are not paired in the last axis of the tensor but remain
    # linked by their shared index in the ordering.
    point_indices = tf.argsort(tf.transpose(point_angle, perm=[0, 1, 2, 4, 3]), direction="DESCENDING", axis=-1)
    # just transposing the points to match the ordered indexes
    points_T = tf.transpose(intersection_points, [0, 1, 2, 4, 3])
    """
    Create a template to double up the point_order indices. This is so that when we create gather
    the intersection points into a tensor sorted by the angle of the points to the center point
    it will insert a copy of each point right below it's ordered spot. This is neccessary because
    for creating the full set of points in the edges of our triangles.
    """
    idx_template = tf.broadcast_to(tf.range(point_indices.shape[-1]), shape=point_indices.shape)
    # this creates the doubled set of ordered indexes
    point_order = tf.gather(point_indices, tf.sort(tf.concat([idx_template, idx_template], axis=-1), axis=-1), batch_dims=-1)
    # this gathers the point into a tensor ordered by their angle to the center point.
    edge_points = tf.transpose(tf.gather(points_T, point_order, batch_dims=-1)[..., :16], perm=[0, 1, 2, 4, 3])
    # this rolls the non-zero values one step. This sets up the tensor to be easily split into the
    # outer edge of the triangle making up the intersection shape.
    edge_points = tf.where(edge_points > 0, tf.roll(edge_points, shift=-1, axis=-2), 0)
    """
    The next few steps function as follows:
        1. Find the index of the first [0, 0] point in each set of intersection points.
            - after rolling the point axis of the tensor this index is the last point of our edges
        2. create a tensor that just represents the indexes for each x, y value in our tensor
        3. create a tensor that is just the first point of every x, y set.
        4. compare the last point index with the full index set and where they match insert the first
          point from each set.

    Now we have a full ordered set of edges for each intersection. These will be combined with the
    center point from before to create a set of traingles from which we can calculate the area of
    the intersection.
    """
    # 1
    last_point = tf.reduce_sum(edge_points, axis=-1)
    last_point = tf.math.count_nonzero(last_point, keepdims=True, axis=-1, dtype=tf.int32)
    last_point = tf.expand_dims(last_point, axis=-1)
    # 2
    idx = tf.transpose(tf.broadcast_to(np.arange(16, dtype=np.int32), last_point.shape[:-1] + (16,)), perm=[0, 1, 2, 4, 3])
    # 3
    first_point = tf.squeeze(tf.gather(edge_points, last_point - last_point, axis=-2, batch_dims=-2), axis=-2)
    # 4
    outer_edges = tf.reshape(tf.where(idx != last_point, edge_points, first_point), edge_points.shape[:-2] + (8, 2, 2))
    # Create Triangles and take their area via the determinant
    # Reshape the center point of each intersection to match the outer edges 
    center = tf.broadcast_to(tf.expand_dims(center, axis=-2), outer_edges.shape[:-2] + (1, 2))
    # Insert the center point into each edge to form the triangles
    triangle_points = tf.concat([outer_edges, center], axis=-2, name="triangles")
    ones = tf.ones(triangle_points.shape[:-1] + (1,), dtype=tf.float32)
    # find the area of each triangle using the determinant and then sum them to find the area of
    # intersection.
    triangle_areas = tf.abs(tf.divide(tf.linalg.det(tf.concat([triangle_points, ones], axis=-1)), 2))
    return triangle_points, outer_edges, triangle_areas

def get_gbox(label_corners:TensorLike, anchor_corners:TensorLike, **kwargs):
    from .funcs import pump_tensor, stretch_tensor
    label_corners = pump_tensor(label_corners, **kwargs)
    anchor_corners = tf.broadcast_to(stretch_tensor(anchor_corners), shape=label_corners.shape)
    all_points = tf.concat([label_corners, anchor_corners], axis=-2)
    axes = tuple(range(len(all_points.shape)))
    all_points = tf.sort(tf.transpose(all_points, axes[:-2] + (axes[-1], axes[-2])), axis=-1)
    all_points = tf.transpose(all_points, axes[:-2] + (axes[-1], axes[-2]))
    gMax = tf.reduce_max(all_points, axis=-2)
    gMin = tf.reduce_min(all_points, axis=-2)
    wh = gMax - gMin

    return (gMin, wh)

def triangle_shapes(triangles_tensor, img, bb, an):
    triangles_list = []
    count = 0
    for i in range(triangles_tensor.shape[-3]):
        triangle = triangles_tensor[img, bb, an, i] 
        if triangle[0, 0] == 0:
            continue
        triangles_list.append(Polygon(triangle))
        count += 1
    return PatchCollection(triangles_list), count

def mark_points(points, img, bb, an):
    xpoints = []
    ypoints = []
    for i in range(points.shape[-2]):
        point = points[img, bb, an, i]
        if point[0] == 0:
            continue
        xpoints.append(point[0])
        ypoints.append(point[1])
    return xpoints, ypoints
    # points.append(Circle(point, radius=5, fill=False))

def image_grid_lines(xsize:int=512, ysize:int=384, xdivs:int=12, ydivs:int=9):
    lines = []
    for i in range(0, xdivs+1, 1):
        line = i * xsize/xdivs
        lines.append([(line, 0), (line, ysize)])
    for i in range(0, ydivs+1, 1):
        line = i * ysize/ydivs
        lines.append([(0, line), (xsize, line)])
    return LineCollection(lines, colors='black', lw=1, alpha=.4, zorder=200)

def dis_Gbox(label_corners, anchor_corners, img, bb, an, **kwargs):
    xy, wh = get_gbox(label_corners, anchor_corners, num_pumps=9)
    return Rectangle(xy[img, bb, an], wh[img, bb, an, 0], wh[img, bb, an, 1], fill=None, edgecolor="tomato", alpha=.7, **kwargs)

def set_plot(img, bb, an, label_corners, anchor_corners, padding:int=40):

    xy, wh = get_gbox(label_corners, anchor_corners, num_pumps=9)
    w, h = wh[img, bb, an, 0], wh[img, bb, an, 1]
    x, y = xy[img, bb, an, 0], xy[img, bb, an, 1]
    cx = x + w/2
    cy = y + h/2
    w = h * (8/6) + padding
    h = h + padding
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set(
            xlim = [cx-w/2, cx+w/2],
            ylim = [cy-h/2, cy+h/2]
            )

    return fig, ax 

def set_ax(ax, img, bb, an, label_corners, anchor_corners, padding:int=40):
    xy, wh = get_gbox(label_corners, anchor_corners, num_pumps=9)
    w, h = wh[img, bb, an, 0], wh[img, bb, an, 1]
    x, y = xy[img, bb, an, 0], xy[img, bb, an, 1]
    cx = x + w/2
    cy = y + h/2
    w = h * (8/6) + padding
    h = h + padding
    ax.set(
            xlim = [cx-w/2, cx+w/2],
            ylim = [cy-h/2, cy+h/2]
            )
    return ax

def display_area_addition(ax, tri_areas, gbox, img, bb, an):

    xlim = ax.get_xlim()
    xy = gbox.get_xy()
    area_len = np.sqrt(tri_areas[img, bb, an])
    area_sum = np.sqrt(np.cumsum(tri_areas[img, bb, an]))
    yline = xy[1] - area_sum[-1] - 10
    ylim = ax.get_ylim()
    dy = ylim[0] - yline
    dx = dy * 8/6

    ax.set(
            xlim = [xlim[0] - dx/2, xlim[1] + dx/2],
            ylim = [yline, ylim[1]]
           )
    ylim = ax.get_ylim()

    area_x_min = np.array(np.cumsum(np.roll(area_len, shift=1)))
    area_x_min = area_x_min + np.array(np.arange(area_x_min.shape[0])) * 5

    total_len = area_sum[-1] + area_x_min[-1] - area_x_min[0]

    center_x_min = (sum(xlim) / 2) - (total_len / 2) + area_x_min
    center_y_min = (ylim[0] + area_sum[-1] / 2) - area_len/2

    block_starts = np.append(np.expand_dims(center_x_min, axis=-1), np.expand_dims(center_y_min, axis=-1), axis=-1)
    
    sum_block_center = center_x_min[-1] + (area_sum[-1] / 2)
    sum_block_x = sum_block_center - area_sum / 2
    sum_block_y = (ylim[0] + area_sum[-1] / 2) - area_sum / 2
    sum_block_starts = np.asarray([sum_block_x, sum_block_y], dtype=np.float32).T

    return sum_block_starts, block_starts, area_sum, area_len

def setup_labels_plot(num_plots:Tuple[int, int]=(1, 1),
                      img_width:int=768,
                      img_height:int=576,
                      margin_size:int=40,
                      display_grid:bool=True):
    # --------------------
    # Graph Setup
    # --------------------
    figsize = np.array([8, 6]) * np.array(num_plots)[::-1]
    fig, axs = plt.subplots(num_plots[0], num_plots[1], figsize=figsize)
    axs = np.array([axs]).flatten()
    for ax in axs:
        ax.set(
                ylim=[-margin_size, img_height+margin_size],
                xlim=[-margin_size, img_width+margin_size],
                )
        ax.tick_params(
                axis='both',
                which='both',
                bottom = False,
                left = False,
                labelleft = False,
                labelbottom=False
                )
        # --------------------
        # Grid Setup
        # --------------------
        if display_grid:
            lines = []
            for i in range(0, 13, 1):
                line = i * img_width/12
                lines.append([(line, 0), (line, img_height)])
            for i in range(0, 10, 1):
                line = i * img_height/9
                lines.append([(0, line), (img_width, line)])
            grid_lines = mpl.collections.LineCollection(lines, colors='black', lw=1, alpha=.4, zorder=200)
            ax.add_collection(grid_lines)

    return fig, axs
