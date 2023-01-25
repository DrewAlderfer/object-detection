import numpy as np
import tensorflow as tf
from tensorflow.types.experimental import TensorLike

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
    from funcs import get_corners
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
    return triangle_points, outer_edges
