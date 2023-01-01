from glob import iglob
import os
from typing import Union, Tuple, List

import numpy as np
import matplotlib as plt
from numpy.typing import ArrayLike
from tensorflow.keras.utils import load_img

from .funcs import make_masks, process_img_annotations, rotate

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
    def __init__(self, data_name:str, coco_obj:dict, image_path:str, target_size:Union[Tuple[int, int], None]=None):
        self.name = data_name
        self.lookup = coco_obj[self.name]
        self.X_path = image_path 
        self.target_size = target_size

    def get_masks(self, sample_size:Union[int, None]=None, **kwargs):
        if sample_size:
            try:
                img_data = self.lookup['img_data'][:sample_size]
            except IndexError:
                print(f"That index is out of bounds for your data! Returning the entire set instead.")
                img_data = self.lookup['img_data']
        else:
            img_data = self.lookup['img_data']

        img_ids = [(x['id'], x['file_name']) for x in img_data]
        coco = self.lookup['coco']
        images = np.empty((0,), dtype=np.float32)
        masks = np.empty((0,), dtype=np.float32)

        for id, file_name in img_ids:
            annot = coco.getAnnIds(id)
            annot = coco.loadAnns(annot)
            img = np.asarray(load_img(f"{self.X_path}/{self.name}/{file_name}", target_size=self.target_size), dtype=np.float32)
            mask = make_masks(annot, input_size=(1440, 1920), target_size=self.target_size, **kwargs)
            if images.shape == (0,):
                images = np.empty(((0,) + img.shape), dtype=np.float32)
                masks = np.empty(((0,) + mask.shape), dtype=np.float32)
            images = np.append(images, img.reshape((1,) + img.shape), axis=0)
            masks = np.append(masks, mask.reshape((1,) + mask.shape), axis=0)

        return images, masks

    def get_labels(self, 
                   divs:Tuple[int, int]=(6, 8), 
                   num_boxes:int=3, num_classes:int=13,
                   sample_size:Union[int, None]=None,
                   input_size:Tuple[int, int]=(1440, 1920),
                   save:bool=False, 
                   save_path:Union[os.PathLike, str, None]=None):

        if sample_size:
            try:
                img_data = self.lookup['img_data'][:sample_size]
            except IndexError:
                print(f"That index is out of bounds for your data! Returning the entire set instead.")
                img_data = self.lookup['img_data']
        else:
            img_data = self.lookup['img_data']

        img_ids = [(x['id'], x['file_name']) for x in img_data]
        coco = self.lookup['coco']
        labels = np.empty((0, divs[1], divs[0], num_classes + 6), dtype=np.float32)
        for id, file_name in img_ids:
            # print(f"searching image: {file_name}")
            annot = coco.getAnnIds(id)
            annot = coco.loadAnns(annot)
            # [1, 2, 3, ... 13, Pc, x1, y1, w, h, phi]
            img_labels = np.zeros((divs[1], divs[0], num_classes + 6), dtype=np.float32) 
            for entry in annot:
                # print(entry)
                y1, x1, w, h, phi = self.translate_points(entry['bbox'], input_size)
                x_cell = int(divmod(x1, self.target_size[1]/divs[1])[0])
                y_cell = int(divmod(y1, self.target_size[0]/divs[0])[0])
                cat = entry['category_id']
                label_vec = np.zeros((num_classes + 6), dtype=np.float32)
                label_vec[13:] = 1, x1, y1, h, w, phi
                label_vec[cat] = 1 
                img_labels[x_cell, y_cell, :] = label_vec

            labels = np.append(labels, img_labels.reshape((1,) + img_labels.shape), axis=0)
                # print(f"found object at cell: ({x_cell}, {y_cell})")
        return labels




    def translate_points(self, entry:list, input_size:Tuple[int, int]):
        """
        process individual annotations from the MVTec Screws Dataset. Takes the 'bbox' list and
        returns the adjusted coordinates of the bounding box as a tuple with three lists of points.

        returns:
            boundingbox_corners, crop_corners(top right, and left points), center_line([x1, y1], [x2, y2])
        """
        # grab bbox info
        row, col, width, height, phi = entry
        # -pi to pi -> 0 to 2*pi
        # initial bounds
        col = col * self.target_size[1] / input_size[1]
        row = row * self.target_size[0] / input_size[0]
        width = width * self.target_size[1] / input_size[1]
        height = height * self.target_size[0] / input_size[0]
        return row, col, width, height, phi
        
        
        



    def crop_dataset(self, save_to:Union[os.PathLike, str], sample_size:Union[int, None]=None, **kwargs):

        if sample_size:
            try:
                img_data = self.lookup['img_data'][:sample_size]
            except IndexError:
                print(f"That index is out of bounds for your data! Returning the entire set instead.")
                img_data = self.lookup['img_data']
        else:
            img_data = self.lookup['img_data']

        img_ids = [(x['id'], x['file_name']) for x in img_data]
        coco = self.lookup['coco']

        for id, file_name in img_ids:
            annot = coco.getAnnIds(id)
            annot = coco.loadAnns(annot)
            img = load_img(f"{self.X_path}/{self.name}/{file_name}")
            count = 1
            for i, entry in enumerate(annot):
                crop, _, _ = process_img_annotations(entry['bbox'])
                category = entry['category_id']
                x, y = [[x for x, y in crop],
                        [y for x, y in crop]]
                img_cropped = img.crop((min(x) - 10, min(y) - 10, max(x) + 10, max(y) + 10))

                save_path = f"{save_to}/{self.name}/{category}"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                img_cropped.save(f"{save_path}/{count:03d}_{file_name}")
                count += 1 

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
                    
            
class test_box:
    def __init__(self, width, height, center:tuple=(0, 0), angle:np.float32=np.pi/4, xy_offset:Tuple[int, int]=(1, 2), debug:bool=False):
        self.w = width
        self.h = height
        self.center = center
        self.phi = angle
        self.T = xy_offset
        self.debug = debug
        self.box1, self.box2 = self.make_boxes()
        self.intersection = self._intersection_points()
        self.inter_center = np.mean(self.intersection, axis=0)

    def make_boxes(self):
        x = np.asarray([self.center[0] + self.w/2, self.center[0] + self.w/2, self.center[0] - self.w/2, self.center[0] - self.w/2], dtype=np.float32)
        y = np.asarray([self.center[1] + self.h/2, self.center[1] - self.h/2, self.center[1] - self.h/2, self.center[1] + self.h/2], dtype=np.float32)
        box1 = np.asarray(list(zip(x, y)), dtype=np.float32)

        tx = self.T[0]
        ty = self.T[1]

        x2 = x + tx
        y2 = y + ty
        box2 = [rotate(point, self.phi, (self.center[0] + tx, self.center[1] + ty)) for point in zip(x2, y2)]

        return box1, box2

    def move_b(self, amount:Union[Tuple[int, int], None]=None, angle:Union[np.float32, None]=None):
        if amount:
            x, y = [[x for x, y in self.box2],
                    [y for x, y in self.box2]]
            x = np.asarray(x, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)

            x = x + amount[0]
            y = y + amount[1]

            self.T[0] += amount[0]
            self.T[1] += amount[1]

            self.box2 = np.asarray(list(zip(x, y)), dtype=np.float32)

        if angle:
            self.box2 = [rotate(point, angle, self.T) for point in self.box2]
            self.phi = angle

    def is_inside(self, point, box):
        x, y = point[0], point[1]

        result = False

        for i, corner in enumerate(box):
            j = i - 1
            if i == 0:
                j = len(box) - 1
            x_i = corner[0]
            y_i = corner[1]
            x_j = box[j][0]
            y_j = box[j][1]

            check = (y_i > y) != (y_j > y)
            inside = x < (x_j - x_i) * (y - y_i) / (y_j - y_i) + x_i
            if check and inside:
                result = abs(result * result - 1)
                # print(result)

        return result

    def check_corners(self):
        results = np.empty((0, 2), dtype=np.float32)
        for corner in self.box1:
            if self.is_inside(corner, self.box2):
                print(f"\nfound corner of box1:\n{corner}\nin box2")
                results = np.append(results, corner.reshape(1, 2), axis=0)

        for corner in self.box2:
            if self.is_inside(corner, self.box1):
                print(f"\nfound corner of box2:\n{corner}\nin box1")
                results = np.append(results, corner.reshape(1, 2), axis=0)
        return results

    def edge_intersections(self, edge1, edge2) -> np.ndarray:
        x1, y1 = edge1[0]
        x2, y2 = edge1[1]

        x3, y3 = edge2[0]
        x4, y4 = edge2[1]

        denom =  (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if denom == 0:
           return 

        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
        if ua < 0 or ua > 1:
            return
        if ub < 0 or ub > 1:
            return
        xnum = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
        ynum = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
        x_i = xnum / denom
        y_i = ynum / denom
        if self.debug:
            print(f"Denom = {denom}\nua = {ua}\nub = {ub}")
            print(f"From edge: {edge1}\nand\n{edge2}")
            print(f"\nFound Intersection at:\n({x_i}, {y_i})\n")

        return np.full((1, 2), [x_i, y_i], dtype=np.float32)
          


    def _intersection_points(self):

        edges1 = [(self.box1[0], self.box1[1]),
                  (self.box1[1], self.box1[2]),
                  (self.box1[2], self.box1[3]),
                  (self.box1[3], self.box1[0])]
        
        edges2 = [(self.box2[0], self.box2[1]),
                  (self.box2[1], self.box2[2]),
                  (self.box2[2], self.box2[3]),
                  (self.box2[3], self.box2[0])]

        results = np.empty((0, 2))

        for edge in edges2:
            for test_edge in edges1:
                intersection = self.edge_intersections(edge, test_edge)
                if intersection is None:
                    continue
                results = np.append(results, intersection, axis=0)

        results = np.append(results, self.check_corners(), axis=0) 
        self.inter_center = np.mean(results, axis=0)

        return sorted(results, key=self.clockwise_sort)

    def intersection_area(self):
        P = self.intersection
        C = self.inter_center
        result = 0
        for i in range(len(self.intersection)):
            j = i - 1
            triangle = np.array([[P[i - 1][0], P[i - 1][1], 1],
                                 [P[i][0],     P[i][1],     1],
                                 [C[0],        C[1],        1]], dtype=np.float32)
            result += np.abs(np.linalg.det(triangle) * .5)

        return result
    
    def clockwise_sort(self, point):
        x, y = point - self.inter_center
        # print(f"Point at ({x}, {y})\nhas arctan of {np.arctan(y/x):.4f}")
        return np.arctan2(y, x)
    
    def IoU(self):

        determinate_1 = np.ones((3,3), dtype=np.float32)
        determinate_2 = np.ones((3,3), dtype=np.float32)
        determinate_1[:,:2] = determinate_1[:,:2] * self.box1[1:]
        determinate_2[:,:2] = determinate_2[:,:2] * self.box2[1:]
        
        box1_area = np.abs(np.linalg.det(determinate_1))
        box2_area = np.abs(np.linalg.det(determinate_2))
        print(box1_area, box2_area) 
        intersection = self.intersection_area()
        union = (box1_area + box2_area) - intersection
        print(f"{intersection}\n---------------\n{union}")

        return intersection / union

class bbox_worker:
    def __init__(self, label_vector:list, pred_vector:list, debug:bool=False):

        self.debug = debug

        self.label_box = self.make_box(label_vector)
        self.pred_box = self.make_box(pred_vector)

        self.intersection = self._intersection_points()
        self.inter_center = np.mean(self.intersection, axis=0)

    def make_box(self, box_vector):

        box_cx = box_vector[0]
        box_cy = box_vector[1]
        box_w = box_vector[2]
        box_h = box_vector[3]
        box_a = box_vector[4]

        x = np.asarray([box_cx + box_h/2, box_cx + box_h/2, box_cx - box_h/2, box_cx - box_h/2], dtype=np.float32)
        y = np.asarray([box_cy + box_w/2, box_cy - box_w/2, box_cy - box_w/2, box_cy + box_w/2], dtype=np.float32)
        box = list(zip(x, y))
        box = [rotate(point, box_a, [box_cx, box_cy]) for point in box]

        return box

    def is_inside(self, point, box):
        x, y = point[0], point[1]

        result = False

        for i, corner in enumerate(box):
            j = i - 1
            if i == 0:
                j = len(box) - 1
            x_i = corner[0]
            y_i = corner[1]
            x_j = box[j][0]
            y_j = box[j][1]

            check = (y_i > y) != (y_j > y)
            x_check = (x_j - x_i) * (y - y_i) / (y_j - y_i) + x_i
            inside = x < x_check 
            if check and inside:
                result = abs(result * result - 1)
        
        return result

    def check_corners(self):
        results = np.empty((0, 2), dtype=np.float32)
        for corner in self.label_box:
            if self.is_inside(corner, self.pred_box):
                # print(f"\nfound corner of box1:\n{corner}\nin box2")
                results = np.append(results, corner.reshape(1, 2), axis=0)

        for corner in self.pred_box:
            if self.is_inside(corner, self.label_box):
                # print(f"\nfound corner of box2:\n{corner}\nin box1")
                results = np.append(results, corner.reshape(1, 2), axis=0)
        return results

    def edge_intersections(self, edge1, edge2) -> np.ndarray:
        x1, y1 = edge1[0]
        x2, y2 = edge1[1]

        x3, y3 = edge2[0]
        x4, y4 = edge2[1]

        denom =  (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if denom == 0:
           return 

        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
        if ua < 0 or ua > 1:
            return
        if ub < 0 or ub > 1:
            return
        xnum = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
        ynum = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
        x_i = xnum / denom
        y_i = ynum / denom
        if self.debug:
            print(f"Denom = {denom}\nua = {ua}\nub = {ub}")
            print(f"From edge: {edge1}\nand\n{edge2}")
            print(f"\nFound Intersection at:\n({x_i}, {y_i})\n")

        return np.full((1, 2), [x_i, y_i], dtype=np.float32)
          


    def _intersection_points(self):

        edges1 = [(self.label_box[0], self.label_box[1]),
                  (self.label_box[1], self.label_box[2]),
                  (self.label_box[2], self.label_box[3]),
                  (self.label_box[3], self.label_box[0])]
        
        edges2 = [(self.pred_box[0], self.pred_box[1]),
                  (self.pred_box[1], self.pred_box[2]),
                  (self.pred_box[2], self.pred_box[3]),
                  (self.pred_box[3], self.pred_box[0])]

        results = np.empty((0, 2))

        for edge in edges2:
            for test_edge in edges1:
                intersection = self.edge_intersections(edge, test_edge)
                if intersection is None:
                    continue
                results = np.append(results, intersection, axis=0)

        results = np.append(results, self.check_corners(), axis=0) 
        self.inter_center = np.mean(results, axis=0)

        return sorted(results, key=self.clockwise_sort)

    def intersection_area(self):
        P = self.intersection
        C = self.inter_center
        result = 0
        for i in range(len(self.intersection)):
            j = i - 1
            triangle = np.array([[P[i - 1][0], P[i - 1][1], 1],
                                 [P[i][0],     P[i][1],     1],
                                 [C[0],        C[1],        1]], dtype=np.float32)
            result += np.abs(np.linalg.det(triangle) * .5)

        return result
    
    def clockwise_sort(self, point):
        x, y = point - self.inter_center
        # print(f"Point at ({x}, {y})\nhas arctan of {np.arctan(y/x):.4f}")
        return np.arctan2(y, x)
    
    def IoU(self):

        determinate_1 = np.ones((3,3), dtype=np.float32)
        determinate_2 = np.ones((3,3), dtype=np.float32)
        determinate_1[:,:2] = determinate_1[:,:2] * self.label_box[1:]
        determinate_2[:,:2] = determinate_2[:,:2] * self.pred_box[1:]
        
        box1_area = np.abs(np.linalg.det(determinate_1))
        box2_area = np.abs(np.linalg.det(determinate_2))
        # print(box1_area, box2_area) 
        intersection = self.intersection_area()
        union = (box1_area + box2_area) - intersection
        # print(f"{intersection}\n---------------\n{union}")

        return intersection / union

    def union(self):

        determinate_1 = np.ones((3,3), dtype=np.float32)
        determinate_2 = np.ones((3,3), dtype=np.float32)
        determinate_1[:,:2] = determinate_1[:,:2] * self.label_box[1:]
        determinate_2[:,:2] = determinate_2[:,:2] * self.pred_box[1:]
        
        box1_area = np.abs(np.linalg.det(determinate_1))
        box2_area = np.abs(np.linalg.det(determinate_2))
        intersection = self.intersection_area()
        union = (box1_area + box2_area) - intersection

        return union

    def get_Gbbox(self):
        box1 = self.label_box.copy()
        box2 = self.pred_box.copy()
        box = []
        box.extend(box1)
        box.extend(box2)

        x, y = [[x for x, y in box],
                [y for x, y in box]]

        maxX = np.max(x)
        minX = np.min(x)
        maxY = np.max(y)
        minY = np.min(y)

        c_box = np.asarray([[maxX, maxY],
                            [maxX, minY],
                            [minX, minY],
                            [minX, maxY]], dtype=np.float32)

        return c_box

    def GIoU(self):
        box1 = self.label_box.copy()
        box2 = self.pred_box.copy()
        box = []
        box.extend(box1)
        box.extend(box2)

        x, y = [[x for x, y in box],
                [y for x, y in box]]

        maxX = np.max(x)
        minX = np.min(x)
        maxY = np.max(y)
        minY = np.min(y)

        c_box = np.asarray([[maxX, maxY],
                            [maxX, minY],
                            [minX, minY],
                            [minX, maxY]], dtype=np.float32)
        C = np.ones((3, 3), dtype=np.float32)
        C[:,:2] = C[:,:2] * c_box[:-1]
        C = np.abs(np.linalg.det(C))
        union = self.union()

        return self.IoU() - np.abs(C / union) / C 

