import numpy as np

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
