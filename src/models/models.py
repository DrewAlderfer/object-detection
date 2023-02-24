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
