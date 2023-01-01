import os
from typing import List, Union, Tuple, BinaryIO
import numpy as np
from numpy.typing import NDArray, ArrayLike

import tensorflow as tf
from tensorflow.keras import Model, layers, losses, metrics
from tensorflow.data import Dataset

class AnchorBoxes(layers.Layer):
    def __init__(self, anchors_per_cell:int=3, grid:Union[Tuple[int, int], int]=7):
        super(AnchorBoxes, self).__init__()
        self.anchor_num = anchors_per_cell
        self.divisions = grid

    def build(self, input_shape):
        self.w = self.add_weight(
                shape=(input_shape[-1], self)
                )



class YOLOLoss(losses.Loss):
    def __init__(self, obj_threshold:float=0.5, grid_divs:Tuple=(6,8), num_boxes:int=3, num_classes:int=13):
        super(YOLOLoss, self).__init__()
        self.threshold = obj_threshold
        self.Sx = grid_divs[0]
        self.Sy = grid_divs[1]
        self.B = num_boxes
        self.C = num_classes

    def call(self, y_true, y_pred):
        # assert y_pred.shape == (self.Sx, self.Sy + (self.C + self.B * 6))
        # preds = y_pred.reshape(self.Sy, self.Sx, self.C + (self.B * 6))
        preds = y_pred
        box_exists = y_true[..., 13]
        
        # Box Loss Value
        box_1_mse = np.mean(
                            np.subtract(y_true[..., 14], preds[..., 14])**2
                          + np.subtract(y_true[..., 15], preds[..., 15])**2
                            )
        box_1_mse += np.mean(
                            np.subtract(y_true[..., 16], preds[..., 16])**2
                          + np.subtract(y_true[..., 17], preds[..., 17])**2
                            )
        box_1_mse += np.mean(
                            np.subtract(y_true[..., 18], preds[..., 18])**2
                            )
        box_2_mse = np.mean(
                            np.subtract(y_true[..., 14], preds[..., 20])**2
                          + np.subtract(y_true[..., 15], preds[..., 21])**2
                            )
        box_2_mse += np.mean(
                            np.subtract(y_true[..., 16], preds[..., 22])**2
                          + np.subtract(y_true[..., 17], preds[..., 23])**2
                            )
        box_2_mse += np.mean(
                            np.subtract(y_true[..., 18], preds[..., 24])**2
                            )
        box_3_mse = np.mean(
                            np.subtract(y_true[..., 14], preds[..., 26])**2
                          + np.subtract(y_true[..., 15], preds[..., 27])**2
                            )
        box_3_mse += np.mean(
                            np.subtract(y_true[..., 16], preds[..., 28])**2
                          + np.subtract(y_true[..., 17], preds[..., 29])**2
                            )
        box_3_mse += np.mean(
                            np.subtract(y_true[..., 18], preds[..., 30])**2
                            )
        best_box = np.min([box_1_mse, box_2_mse, box_3_mse])
        return best_box 


def IoU(y_pred, y_true):
    x1, y1, x2, y2 = y_true
    x_i, y_i, x_j, y_j = y_pred

    t_area = abs(x1 - x2) * abs(y1 - y2)
    p_area = abs(x_i - x_j) * abs(y_i - y_j)
    print(f"True Area: {t_area}\nPredicted Area: {p_area}")

    x_max = np.max([x1, x_i])
    x_min = np.min([x2, x_j])
    y_max = np.max([y1, y_i])
    y_min = np.min([y2, y_j])
    print(f"x_max: {x_max}, x_min: {x_min}\ny_max: {y_max}, y_min: {y_min}")

    intersection = np.clip((x_min - x_max), 0, None) * np.clip((y_min - y_max), 0, None)
    union =  (t_area + p_area) - intersection
    print(f"Intersection: {intersection}\nUnion: {union}")

    return intersection / union

