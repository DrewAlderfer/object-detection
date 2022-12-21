import tensorflow as tf
from tensorflow.keras import layers, losses, metrics, Model
from tensorflow.keras.layers import Layer

def db_print(message: str, debug: bool = True):
    if db:
        return print(message)

def conv2d_block(x, filters, kernel_size=3, reps: int = 2, pooling: bool = False, debug=False, **kwargs):

    residual = x
    db_print(f"beginning  of block: \nX = {x.shape}\nResidual {residual.shape}", debug=debug)  #
    options = {}

    if kwargs:
        options.update(**kwargs)

    for rep in range(reps):
        if not rep:
            options.update({"strides": 2})
        else:
            options["strides"] = 1

        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, kernel_size, padding="same", use_bias=False, **options)(x)

        db_print(f"end of first loop: \nX = {x.shape}\nResidual {residual.shape}", debug=debug)  #

    db_print(f"loops finished: \nX = {x.shape}\nResidual {residual.shape}", debug=debug)  #
    if pooling:
        x = layers.MaxPooling2D(2, padding="same")(x)
        residual = layers.Conv2D(filters, 1, strides=2)(residual)
        residual = layers.Conv2D(filters, 1, strides=2)(residual)
        db_print(f"after pooling: \nX = {x.shape}\nResidual {residual.shape}", debug=debug)  #
    elif filters != residual.shape[-1]:
        db_print(f"residual no pooling loop: \nX = {x.shape}\nResidual {residual.shape}" ,debug=debug)  #
        residual = layers.Conv2D(filters, 1)(residual)

    db_print(f"adding residual: \nX = {x.shape}\nResidual {residual.shape}", debug=debug)  #
    x = layers.add([x, residual])
    return x


class CNNBlock(layers.Layer):
    def __init__(self, out_channels, kernel_size=3, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = layers.Conv2D(out_channels, kernel_size, padding='same', **kwargs)
        self.bn = layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x

