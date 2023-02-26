import os
from typing import List, Union, Tuple, BinaryIO
import numpy as np
from numpy.typing import NDArray, ArrayLike

import tensorflow as tf
from tensorflow.keras import Model, layers, losses, metrics
from tensorflow.data import Dataset
from tensorflow.types.experimental import TensorLike

from ..utils import calc_best_anchors

class YOLO_Loss(tf.keras.losses.Loss):
    def __init__(self, lamb_coord=.5, lamb_phi=.2, lamb_noobj=.5, name="yolo_loss"):
        super().__init__(name=name)
        self.lamb_coord = lamb_coord
        self.lamb_phi = lamb_phi
        self.lamb_noobj = lamb_noobj
                 

    def call(self, y_true:TensorLike, y_pred:TensorLike):
        """
        YOLO Loss Function
        """
        p_class = y_pred[..., :13]
        p_obj = y_pred[..., 13:14]
        p_bboxes = y_pred[..., 14:]
        # print(f"y_true: {y_true.shape}")
        # print(f"p_bboxes: {p_bboxes.shape}")
        p_bestboxes, detector_idx = calc_best_anchors(y_true, p_bboxes)
        # ------------------------------
        # Scatter True Values into a Tensor shape: 
        # ------------------------------
        # print(f"y_true shape[0]: {y_true.shape[0]}")
        # print(f"y_true shape[0]: {tf.range(y_true.shape[0], dtype=tf.int32)}")
        # print(f"detector_idx: {detector_idx.shape}")
        # print(f"batch_idx step 1:")
        # print(f"{tf.range(y_true.shape[0], dtype=tf.int32)}")
        # step1 = tf.range(y_true.shape[0], dtype=tf.int32)
        # print(f"batch_idx step 2:")
        # print(f"{tf.reshape(step1, shape=(16, 1, 1))}")
        # step2 = tf.reshape(step1, shape=(16, 1, 1))
        batch_idx = tf.reshape(tf.range(y_true.shape[0], dtype=tf.int32), shape=(y_true.shape[0], 1, 1))
        # print(f"batch_idx: {batch_idx.shape}")
        batch_idx = tf.broadcast_to(batch_idx, shape=detector_idx.shape[:-1] + (1,))
        detector_idx = tf.concat([batch_idx, detector_idx], axis=-1)
        gnd_truth = tf.scatter_nd(detector_idx, y_true, shape=p_class.shape[:-1] + (19,))
        # ------------------------------
        # Assign loss components
        # ------------------------------
        t_class = gnd_truth[..., :13] 
        t_obj = tf.squeeze(gnd_truth[..., 13:14], axis=-1)
        p_obj = tf.squeeze(p_obj, axis=-1)
        t_noobj = tf.cast(gnd_truth[..., 13:14] < 1, dtype=tf.float32)
        t_bboxes = y_true[..., 14:]
        t_box_exists = y_true[..., 13]

        t_xy = t_bboxes[..., :2]
        t_wh = t_bboxes[..., 2:4]
        t_phi = t_bboxes[..., -1:]

        p_xy = p_bestboxes[..., :2]
        p_wh = p_bestboxes[..., 2:4]
        p_phi = p_bestboxes[..., -1:]

        # print(f"t_wh: {t_wh.shape}, {t_wh.dtype}\n{t_wh[0]}")
        # print(f"p_wh: {p_wh.shape}, {p_wh.dtype}\n{p_wh[0]}")

        sq_dif = tf.math.squared_difference
        # ------------------------------
        # XY, WH, Angle Loss
        # ------------------------------
        xy_loss = tf.reduce_sum(tf.reduce_sum(sq_dif(t_xy, p_xy), axis=-1) * tf.squeeze(t_box_exists), axis=-1) * self.lamb_coord
        wh_loss = tf.reduce_sum(tf.reduce_sum(sq_dif(tf.sqrt(t_wh), tf.sqrt(p_wh)), axis=-1) * tf.squeeze(t_box_exists), axis=-1) * self.lamb_coord
        phi_loss = tf.reduce_sum(tf.reduce_sum(sq_dif(t_phi, p_phi), axis=-1) * tf.squeeze(t_box_exists), axis=-1) * self.lamb_phi
        # ------------------------------
        # Confidence Loss
        # ------------------------------
        conf_loss = tf.reduce_sum(tf.reduce_sum(sq_dif(t_obj, p_obj) * t_obj, axis=-1), axis=-1)
        noobj_loss = tf.reduce_sum(tf.reduce_sum(sq_dif(t_obj, p_obj) * tf.squeeze(t_noobj, axis=-1), axis=-1), axis=-1) * self.lamb_noobj
        # ------------------------------
        # Class Loss
        # ------------------------------
        class_loss = tf.reduce_sum(tf.reduce_sum(sq_dif(t_class, p_class), axis=-1) * t_obj, axis=[-1, -2])

        # print(f"xy_loss: {xy_loss}")
        # print(f"wh_loss: {wh_loss}")
        # print(f"phi_loss: {phi_loss}")
        # print(f"conf_loss: {conf_loss}")
        # print(f"noobj_loss: {noobj_loss}")
        # print(f"class_loss: {class_loss}")

        return xy_loss + wh_loss + phi_loss + conf_loss + noobj_loss + class_loss

def YOLO_convnet(img_size):
    inputs = tf.keras.Input(shape=img_size + (3,))
    x = tf.keras.layers.Rescaling(2./255)(inputs)

    x = tf.keras.layers.Conv2D(filters=16, kernel_size=7, use_bias=False)(x)
    for size in [32, 64, 128, 256, 513, 1026]:
        residual = x

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.SeparableConv2D(size, 3, padding='same', use_bias=False)(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.SeparableConv2D(size, 3, padding='same', use_bias=False)(x)

        x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)

        residual = tf.keras.layers.Conv2D(
            size, 1, strides=2, padding='same', use_bias=False)(residual)
        x = tf.keras.layers.add([x, residual]) 

    # x = tf.keras.layers.Flatten()(x)
    print(f"ConvNet Output: {x.shape}")
    # x = tf.keras.layers.Dropout(.5)(x)     
    x = tf.keras.layers.Dense(19 * 9, activation='sigmoid')(x)
    x = tf.keras.layers.Reshape(target_shape=(108, 9, 19))(x)
    # print(f"x: {x.shape}")
    # bb_x = x[..., -5:]
    # print(f"anchors: {anchors.shape}")
    # print(f"bb_x: {bb_x.shape}")
    # bb_x = tf.keras.layers.Add()([bb_x, anchors])

    # outputs = tf.keras.layers.Concatenate(axis=-1)([x[..., :-5], bb_x])
    outputs = x
    model = tf.keras.Model(inputs, outputs, name='convnet')

    return model

class YOLO(tf.keras.Model):
    def __init__(self, convnet, anchors, **kwargs):
        super().__init__(**kwargs)
        self.convnet = convnet
        self.anchors = anchors

    @tf.function
    def train_step(self, input):
        x, y = input
        # # print(f"\nx shape: {x.shape}")
        # print(f"\ny shape: {y.shape}")
        with tf.GradientTape() as tape:
            # print(f"\nx shape: {x.shape}")
            preds = self.convnet(x, training=True)
            anchors_base = tf.zeros(shape=(1,) + preds.shape[1:-1]
                               + (preds.shape[-1] - self.anchors.shape[-1],),
                               dtype=tf.float32) 
            anchor_vals = tf.keras.layers.Concatenate(axis=-1)([anchors_base, self.anchors])
            anchor_vals = tf.broadcast_to(anchor_vals, shape=preds.shape)
            anchor_zeros = tf.zeros(shape=preds.shape)
            anchors = tf.keras.layers.Add()([anchor_zeros, anchor_vals])
            preds = tf.keras.layers.Add()([preds, anchors])
            loss = self.compiled_loss(y, preds, regularization_losses=self.losses)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights)) 
        return {"loss": loss}
    
    @tf.function
    def test_step(self, data):
        x, y = data
        preds = self.convnet(x, training=False)
        anchors_base = tf.zeros(shape=(1,) + preds.shape[1:-1]
                            + (preds.shape[-1] - self.anchors.shape[-1],),
                            dtype=tf.float32) 
        anchor_vals = tf.keras.layers.Concatenate(axis=-1)([anchors_base, self.anchors])
        anchor_vals = tf.broadcast_to(anchor_vals, shape=preds.shape)
        anchor_zeros = tf.zeros(shape=preds.shape)
        anchors = tf.keras.layers.Add()([anchor_zeros, anchor_vals])
        preds = tf.keras.layers.Add()([preds, anchors])
        loss = self.compiled_loss(y, preds, regularization_losses=self.losses)
        return {"loss": loss}

    @tf.function
    def predict(self, data):
        preds = self.convnet(data, training=False)
        anchors_base = tf.zeros(shape=(1,) + preds.shape[1:-1]
                            + (preds.shape[-1] - self.anchors.shape[-1],),
                            dtype=tf.float32) 
        anchor_vals = tf.keras.layers.Concatenate(axis=-1)([anchors_base, self.anchors])
        anchor_vals = tf.broadcast_to(anchor_vals, shape=preds.shape)
        anchor_zeros = tf.zeros(shape=preds.shape)
        anchors = tf.keras.layers.Add()([anchor_zeros, anchor_vals])
        preds = tf.keras.layers.Add()([preds, anchors])

        return preds
