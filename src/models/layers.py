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

class BoxCutter(Layer):
    def __init__(self, num_classes, units):
        super().__init__()
        self.C = num_classes
        self.units = units

    def build(self, input_shape):
        assert(len(input_shape) == 4)
        print(f"input_shape: {input_shape}")
        self.B = int(input_shape[-1] / (self.C + 1 + self.units))

    def call(self, inputs):
        batch, x, y = inputs.shape[:3]
        print(f"batch: {batch}, x: {x}, y: {y}, num_boxes: {self.B}")
        detectors = tf.concat(tf.split(tf.expand_dims(inputs, axis=-2), self.B, axis=-1), axis=-2)
        new_shape = (detectors.shape[0], x * y) + detectors.shape[-2:]
        print(f"new_shape: {new_shape}")
        detectors = tf.reshape(detectors, shape=new_shape)
        classes = detectors[..., :self.C]
        object = detectors[..., self.C:self.C+1]
        boxes = detectors[..., -self.units:]
        print(f"detectors: {detectors.shape}")
        # classes = tf.reshape(inputs[..., :self.C], (batch, x * y, self.C))
        # object = tf.reshape(inputs[..., self.C:self.C+1], (batch, x * y, 1))
        # boxes = tf.concat(tf.split(tf.expand_dims(inputs[..., self.C+1:], axis=-2), self.B, axis=-1), axis=-2)
        # boxes = tf.reshape(boxes, [batch, x * y, self.B, self.units])

        return [classes, object, boxes]

class AddAnchors(Layer):
    def __init__(self, anchors):
        super().__init__()
        self.anchors = anchors

    def build(self, input_shape):
        print(f"input_shape: {input_shape}")
        self.units = input_shape[-1]
        self.B = input_shape[-2]
        w_init = tf.random_normal_initializer(stddev=0.01)
        self.w = tf.Variable(
                initial_value = w_init(shape=(input_shape), dtype=tf.float32),
                trainable=True
                )
        print(f"weights: {self.w.shape}")
        print(f"anchors: {self.anchors.shape}")
        b_values = tf.reshape(self.anchors, input_shape)
        print(f"b_values: {b_values.shape}")
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
                initial_value=b_init(shape=(b_values.shape), dtype=tf.float32) + b_values,
                trainable=True
                )

    def call(self, inputs):
        x = tf.math.l2_normalize(inputs, axis=-2)
        return x * self.w + self.b
