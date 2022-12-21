from glob import iglob
import os
from typing import Union, Tuple, List

import numpy as np
import matplotlib as plt
from numpy.typing import ArrayLike
from tensorflow.keras.utils import load_img

class data_generator:
    def __init__(
        self,
        X_path: str,
        y_path: str,
        categories: List[int],
        input_shape: Tuple[int, int],
        target_size: Union[Tuple[int,],Tuple[int, int], None] = None,
        batch_size=32):
        """
        Class to prepare data for the Object Detection model. It does a series of things to process
        the image files into a form that can be dropped directly into the model.

        TODO: Make a generator that will return batches. Also make it so that it will do augmentation.
        """
        self.X_path = self.clean_path(X_path)
        self.y_path = self.clean_path(y_path)
        self.input_shape = input_shape
        self.target_size = self._compute_target_size(target_size)
        self.X_stack = self._get_X()
        self.y_stack = self._get_y()

    def clean_path(self, path):
        if path.endswith("/"):
            return path[:-1]

    def _compute_target_size(self, target_size):
        if target_size == (0,):
            target = target_size[0]
            a_val = self.input_shape[0]
            b_val = self.input_shape[1]

            a_target = np.sqrt(np.multiply(target**2, a_val) / b_val)
            b_target = np.sqrt(np.multiply(target**2, b_val) / a_val)

            output = (round(a_target), round(b_target))
            return output
        try:
            assert len(target_size) == 2 or target_size is None
        except AssertionError as err:
            print(
                "target_size must be in the form Tuple[int,], Tuple[int, int] or None"
            )
            raise
        return target_size

    def _get_X(self) -> np.ndarray:
        # get my images
        #   use the list of images to get
        #   labels and form them into an
        #   array of class labels
        X_set = np.empty((0,), dtype=np.float32)
        print(f"Working on grabbing images...")
        for file in iglob(f"{self.X_path}/*"):
            X_img = np.asarray(
                load_img(file, target_size=self.target_size), dtype=np.float32
            )
            if X_set.shape == (0,):
                X_set = np.empty(shape=(0,) + X_img.shape, dtype=np.float32)
            X_set = np.append(X_set, X_img.reshape((1,) + X_img.shape), axis=0)
            if len(X_set) % 10 == 0:
                print("[]", end="")
        print(f"\n\nReturning Image stack with shape: {X_set.shape}")
        return X_set

    def _get_y(self) -> np.ndarray:
        y_set = np.empty((0,), dtype=np.int8)
        print("\n\nCompiling mask layers for each image...")
        for x_file in iglob(f"{self.X_path}/*"):
            y_mask = np.empty((0,), dtype=np.int8)

            for y_file in iglob(f"{self.y_path}/**/*{os.path.basename(x_file)}"):
                y_layer = np.asarray(
                    load_img(
                        y_file, color_mode="grayscale", target_size=self.target_size
                    ),
                    dtype=np.float32,
                )
                y_layer = y_layer.reshape((1,) + y_layer.shape + (1,))

                if y_mask.shape == (0,):
                    y_mask = y_layer
                    continue
                y_mask = np.append(y_mask, y_layer, axis=3)

            if y_set.shape == (0,):
                y_set = np.empty((0,) + y_mask.shape[1:], dtype=np.int8)
            y_set = np.append(y_set, y_mask, axis=0)
            if len(y_set) % 10 == 0:
                print("[]", end="")
        print(f"\n\nReturning mask stack with shape: {y_set.shape}")
        return y_set

class CategoricalDataGen:
    def __init__(self, data_type:str, data_ref:dict, image_path:str, target_size:Union[ArrayLike, None]=None):
        self.name = data_type
        self.lookup = data_ref[self.name]
        self.X_path = image_path 
        self.target_size = target_size

    def batch(self, batch_size:int=32, normalize:bool=True):

        
        _annot_copy = self.lookup['annotations']

        data_len = len(self.lookup['img_data'])
        if batch_size is None:
            batch_size = data_len
        batches = divmod(data_len, batch_size)
        count = 0
        for batch in range(batches[0] + 1):
            batch_X = np.empty((0,), dtype=np.float32)
            batch_y = np.empty((0, 13), dtype=np.float32)
            for i in range(batch_size):
                file_name = self.lookup['img_data'][i]['file_name']
                img_id = self.lookup['img_data'][i]['id']

                img = np.asarray(load_img(f"{self.X_path}/{self.name}/{file_name}", target_size=self.target_size), dtype=np.float32)
                if normalize:
                    img = img / 255
                img = img.reshape((1,) + img.shape)

                img_classes = np.zeros((1, 13), dtype=np.float32)
                for pos, entry in enumerate(_annot_copy):
                    if entry['image_id'] == img_id:
                        img_classes[0, entry['category_id'] - 1] = 1
                        _annot_copy.pop(pos)
                if batch_X.shape == (0,):
                    batch_X = np.empty((0,) + img.shape[1:], dtype=np.float32)
                batch_X = np.append(batch_X, img, axis=0)
                batch_y = np.append(batch_y, img_classes, axis=0)
                count += 1
                if count == data_len:
                    break
            yield batch_X, batch_y
                    
            
