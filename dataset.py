# vim: set ts=4 sw=4 :
# _*_ coding: utf-8 _*_
#
# Copyright (c) 2023 Fyusion Inc.

import os
import numpy as np
import json
from enum import IntEnum

import numpy
from PIL import Image
from torchvision.transforms import transforms
import torch
from augmentation import NoAugmentation
import json
from lru import LRU
import random


# FIXME (mw) obsolete now
class CarState(IntEnum):
    CAR_CLEAN = (0,)
    CAR_DIRTY = 1

    def __str__(self):
        strs = ["Clean", "Dirty"]
        return strs[self.value]


class CarDirtyDataset(torch.utils.data.Dataset):
    @staticmethod
    def __get_label(label_file):
        labels = set()
        with open(label_file, "r") as fd:
            data = json.load(fd)
            for a in data["annotations"]:
                labels.add(a["label"])
        return labels.pop() if len(labels) == 1 else None

    def add_dataset(self, img_dir, label_dir, label_suffix):
        """
        Scans a directory for image files and adds them to the dataset with the given label
        :param img_dir: Directory that contains image files
        :param label_dir: Directory that contains label files
        """
        for subdir in os.walk(img_dir):
            for file in subdir[2]:
                try:
                    img_file = os.path.join(img_dir, file)
                    Image.open(img_file)
                    comp = file.split(".")
                    comp[-1] = label_suffix
                    label_file = os.path.join(label_dir, ".".join(comp))
                    label = CarDirtyDataset.__get_label(label_file)
                    if label:
                        self.labels.append(self.label_dict[label])
                        self.files.append(img_file)
                except:
                    continue

    def __init__(
        self,
        image_dir,
        label_dir,
        label_dict,
        label_suffix="json",
        train=False,
        center_crop=448,
        augmentation=NoAugmentation(),
        cache_items=0,
    ):
        """
        Constructor
        :param image_dir: Directory that contains images of cars
        :param label_dir: Directory that contains the car labels with identical basenames to the images
        :param label_dict: Dictionary that maps label names to label numbers
        :param label_suffix:
        :param train:
        :param center_crop: Size of the crop window in pixels, no crop is done when center_crop is None
        :param augmentation:
        :param cache_items:
        """
        super().__init__()
        self.train = train
        self.label_dict = label_dict
        self.files = []
        self.labels = []
        if label_dict is None:
            raise Exception("Invalid label dictionary supplied")
        if image_dir is None or label_dir is None:
            raise Exception("Invalid data directories supplied")
        self.add_dataset(image_dir, label_dir, label_suffix)
        self.center_crop = center_crop
        self.augmentation = augmentation
        self.cache = LRU(cache_items) if cache_items > 0 else None

        crop_padding = 128
        composition = []
        if train and augmentation is not None:
            self.is_augmented = True
            if (
                augmentation.downscaling_width != 0
                and augmentation.downscaling_height != 0
            ):
                target_size = [
                    augmentation.downscaling_height,
                    augmentation.downscaling_width,
                ]
                composition.append(transforms.Resize(size=target_size))
            if augmentation.random_rotation_angle > 0:
                composition.append(
                    transforms.RandomRotation(augmentation.random_rotation_angle)
                )
            if augmentation.enable_random_cropping:
                assert center_crop is not None
                composition.append(
                    transforms.CenterCrop(center_crop + crop_padding * 2)
                )
                composition.append(transforms.RandomCrop(center_crop))
            if augmentation.enable_vertical_mirroring:
                composition.append(transforms.RandomVerticalFlip())
            if augmentation.enable_horizontal_mirroring:
                composition.append(transforms.RandomHorizontalFlip())

        composition.append(transforms.ToTensor())
        # Normalization values taken from torchvision resnet18 docs
        pre = self.__network_pre_transforms()
        if pre is not None:
            composition += pre
        self.transform = transforms.Compose(composition)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        if self.cache and item in self.cache:
            img = self.cache[item]
            if self.is_augmented:
                return self.transform(img)
        else:
            # Load image using PIL
            img = Image.open(self.files[item]).convert("RGB")
            if self.cache and self.is_augmented:
                self.cache[item] = img
                img = self.transform(img)
            else:
                img = self.transform(img)
                if self.cache:
                    self.cache[item] = img
        return img, self.labels[item]


class DataloopDataset(torch.utils.data.Dataset):
    def __add_dir(self, target_dir, seed=42):
        """
        Scans a directory for image files and valid json files and adds them to the dataset
        :param target_dir: Target directory that contains "items" and "json" subdirectories
        """
        seen_filenames = set()
        self.random_file_list = set()
        random.seed(seed)
        class_to_label = {"clean": 0, "light_dirt": 1, "heavy_dirt": 1}

        # Get one image from each vehicle at random
        for subdir in os.walk(os.path.join(target_dir, "items")):
            for file in subdir[2]:
                if "_" in file:
                    unique_filename = file[: file.rfind("_")]
                else:
                    unique_filename = file
                if unique_filename not in seen_filenames:
                    seen_filenames.add(unique_filename)
                    files_with_prefix = [
                        f
                        for f in os.listdir(os.path.join(target_dir, "items"))
                        if f.startswith(unique_filename)
                    ]
                    self.random_file_list.add(random.choice(files_with_prefix))

        # Get data for randomly selected vehicle images
        for file in self.random_file_list:
            try:
                target_file = os.path.join(target_dir, "items", file)
                Image.open(target_file)
                # Try to open the json file
                annotation_file_name = os.path.join(
                    target_dir, "json", os.path.splitext(file)[0] + ".json"
                )
                with open(annotation_file_name) as annotation_file:
                    json_data = json.load(annotation_file)
                    img_width = json_data["metadata"]["system"]["width"]
                    img_height = json_data["metadata"]["system"]["height"]
                    annotations = []
                    for annotation in json_data["annotations"]:
                        label = annotation["label"]
                        try:
                            coords = annotation["coordinates"]
                            p1 = np.array([coords[0]["x"], coords[0]["y"]])
                            p2 = np.array([coords[1]["x"], coords[1]["y"]])
                            bbox_size = np.linalg.norm(p2 - p1)
                            annotations.append(
                                [bbox_size, class_to_label[label], coords]
                            )
                        except KeyError:
                            try:
                                annotations.append([1, class_to_label[label], 0])
                            except KeyError:
                                continue
                    annotations = sorted(annotations, key=lambda x: x[0], reverse=True)
                    # Only the biggest bounding box is considered
                    if len(annotations) == 0:
                        continue
                    self.files.append(target_file)
                    self.labels.append(annotations[0])
                    self.file_names.append(file)
            except FileNotFoundError:
                # We just ignore files that are not images or do not have a json file
                continue
        assert len(self.files) == len(self.labels)

    def __init__(
        self, dataset_dir, train=False, center_crop=448, augmentation=NoAugmentation()
    ):
        """
        Constructor
        :param dataset_dir: Directory that contains "items" and "json" subdirectories
        :param center_crop: Size of the crop window in pixels
        """
        super().__init__()
        self.train = train
        self.files = []
        self.labels = []
        self.file_names = []
        self.dataset_dir = dataset_dir
        if dataset_dir is not None:
            self.__add_dir(dataset_dir)
        self.center_crop = center_crop
        self.augmentation = augmentation

        # Preprocessing pipeline
        crop_padding = 128
        composition = []
        if train:
            if (
                augmentation.downscaling_width != 0
                and augmentation.downscaling_height != 0
            ):
                target_size = [
                    augmentation.downscaling_height,
                    augmentation.downscaling_width,
                ]
                composition.append(transforms.Resize(size=target_size))
            if augmentation.random_rotation_angle > 0:
                composition.append(
                    transforms.RandomRotation(augmentation.random_rotation_angle)
                )
            if augmentation.enable_random_cropping:
                composition.append(
                    transforms.CenterCrop(center_crop + crop_padding * 2)
                )
                composition.append(transforms.RandomCrop(center_crop))
            if augmentation.enable_vertical_mirroring:
                composition.append(transforms.RandomVerticalFlip())
            if augmentation.enable_horizontal_mirroring:
                composition.append(transforms.RandomHorizontalFlip())

        composition.append(transforms.ToTensor())
        # Normalization values taken from torchvision resnet18 docs
        composition.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
        self.transform = transforms.Compose(composition)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        # Load image using PIL
        img = Image.open(self.files[item]).convert("RGB")
        img = self.transform(img)

        # Only use the label of the biggest bounding box
        return self.file_names[item], img, self.labels[item][1]
        # return img, self.labels[item][1]

    def __network_pre_transforms(self):
        return [
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]


class DirtyYOLODataset(CarDirtyDataset):
    def __init__(
        self,
        image_dir,
        label_dir,
        label_dict,
        label_suffix="json",
        train=False,
        augmentation=NoAugmentation(),
        cache_items=0,
    ):
        super(DirtyYOLODataset, self).__init__(
            image_dir=image_dir,
            label_dir=label_dir,
            label_dict=label_dict,
            label_suffix=label_suffix,
            train=train,
            center_crop=None,
            augmentation=augmentation,
            cache_items=cache_items,
        )
        if augmentation is not None:
            assert not augmentation.enable_random_cropping

    def add_dataset(self, img_dir, label_dir, label_suffix):
        """
        Scans a directory for image files and adds them to the dataset with the given label
        :param img_dir: Directory that contains image files
        :param label_dir: Directory that contains label files
        """
        for subdir in os.walk(img_dir):
            for file in subdir[2]:
                try:
                    img_file = os.path.join(img_dir, file)
                    Image.open(img_file)
                    comp = file.split(".")
                    comp[-1] = label_suffix
                    label_file = os.path.join(label_dir, ".".join(comp))
                    labels = self.__get_labels(label_file)
                    if labels:
                        self.labels.append(labels)
                        self.files.append(img_file)
                except:
                    continue

    def __get_labels(self, label_file):
        labels = []
        with open(label_file, "r") as fd:
            data = json.load(fd)
            width = data["metadata"]["system"]["width"]
            height = data["metadata"]["system"]["height"]
            for a in data["annotations"]:
                label = a["label"]
                labelid = self.label_dict[label]
                boxleft = a["coordinates"][0]["x"]
                boxtop = a["coordinates"][0]["y"]
                boxright = a["coordinates"][1]["x"]
                boxbottom = a["coordinates"][1]["y"]
                centerx = (boxleft + boxright) / (2 * width)
                centery = (boxtop + boxbottom) / (2 * height)
                boxwidth = (boxright - boxleft) / width
                boxheight = (boxbottom - boxtop) / height
                labels.append(
                    torch.Tensor([labelid, centerx, centery, boxwidth, boxheight])
                )
        return labels

    def __network_pre_transforms(self):
        return None


class DataloopDatasetClean(torch.utils.data.Dataset):
    def __add_dir_1(self, target_dir):
        """
        Scans a directory for image files and valid json files and adds them to the dataset
        :param target_dir: Target directory that contains "items" and "json" subdirectories
        """
        class_to_label = {"clean": 0, "light_dirt": 2, "heavy_dirt": 1}
        for subdir in os.walk(os.path.join(target_dir, "items")):
            for file in subdir[2]:
                try:
                    target_file = os.path.join(target_dir, "items", file)
                    Image.open(target_file)
                    # Try to open the json file
                    annotation_file_name = os.path.join(
                        target_dir, "json", os.path.splitext(file)[0] + ".json"
                    )
                    with open(annotation_file_name) as annotation_file:
                        json_data = json.load(annotation_file)
                        img_width = json_data["metadata"]["system"]["width"]
                        img_height = json_data["metadata"]["system"]["height"]
                        annotations = []
                        for annotation in json_data["annotations"]:
                            label = annotation["label"]
                            coords = annotation["coordinates"]
                            p1 = np.array([coords[0]["x"], coords[0]["y"]])
                            p2 = np.array([coords[1]["x"], coords[1]["y"]])
                            bbox_size = np.linalg.norm(p2 - p1)
                            annotations.append(
                                [bbox_size, class_to_label[label], coords]
                            )
                        annotations = sorted(
                            annotations, key=lambda x: x[0], reverse=True
                        )
                        # Only the biggest bounding box is considered
                        if len(annotations) == 0:
                            continue
                        if annotations[0][1] == 2:
                            continue
                        self.files.append(target_file)
                        self.file_names.append(file)
                        self.labels.append(annotations[0][1])
                except:
                    # We just ignore files that are not images or do not have a json file
                    continue
        assert len(self.files) == len(self.labels)

    def __add_dir_2(self, target_dir, label_add=None, seed=42):
        """
        Scans a directory for image files and valid json files and adds them to the dataset
        :param target_dir: Target directory that contains "images", label_add: Label to append onto these images
        """
        seen_filenames = set()
        self.random_file_list = set()

        random.seed(42)
        for subdir in os.walk(os.path.join("Output/dirt_heavy/")):
            for file in subdir[2]:
                if "_" in file:
                    unique_filename = file[: file.rfind("_")]
                else:
                    unique_filename = file
                if unique_filename not in seen_filenames:
                    seen_filenames.add(unique_filename)
                    files_with_prefix = [
                        f
                        for f in os.listdir("Output/dirt_heavy/")
                        if f.startswith(unique_filename)
                    ]
                    self.random_file_list.add(random.choice(files_with_prefix))

        for file in self.random_file_list:
            target_file = os.path.join(target_dir, file)
            Image.open(target_file)
            self.files.append(target_file)
            self.file_names.append(file)
            if label_add is None:
                raise Exception
            else:
                self.labels.append(label_add)
        assert len(self.files) == len(self.labels)

    def __init__(
        self,
        dataset_dir,
        dataset_dir_2,
        train=False,
        center_crop=448,
        augmentation=NoAugmentation(),
    ):
        """
        Constructor
        :param dataset_dir: Directory that contains "items" and "json" subdirectories
        :param center_crop: Size of the crop window in pixels
        """
        super().__init__()
        self.train = train
        self.files = []
        self.labels = []
        self.file_names = []
        self.dataset_dir = dataset_dir
        if dataset_dir is not None:
            self.__add_dir_1(dataset_dir)
        if dataset_dir_2 is not None:
            self.__add_dir_2(dataset_dir_2, label_add=1)
        self.center_crop = center_crop
        self.augmentation = augmentation

        # Preprocessing pipeline
        crop_padding = 128
        composition = []
        if train:
            if (
                augmentation.downscaling_width != 0
                and augmentation.downscaling_height != 0
            ):
                target_size = [
                    augmentation.downscaling_height,
                    augmentation.downscaling_width,
                ]
                composition.append(transforms.Resize(size=target_size))
            if augmentation.random_rotation_angle > 0:
                composition.append(
                    transforms.RandomRotation(augmentation.random_rotation_angle)
                )
            if augmentation.enable_random_cropping:
                composition.append(
                    transforms.CenterCrop(center_crop + crop_padding * 2)
                )
                composition.append(transforms.RandomCrop(center_crop))
            if augmentation.enable_vertical_mirroring:
                composition.append(transforms.RandomVerticalFlip())
            if augmentation.enable_horizontal_mirroring:
                composition.append(transforms.RandomHorizontalFlip())

        composition.append(transforms.ToTensor())
        # Normalization values taken from torchvision resnet18 docs
        composition.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
        self.transform = transforms.Compose(composition)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        # Load image using PIL
        img = Image.open(self.files[item]).convert("RGB")
        img = self.transform(img)

        # Only use the label of the biggest bounding box
        return self.file_names[item],img, self.labels[item]

    def __network_pre_transforms(self):
        return [
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]


class NonDataloopDatasetSubset(torch.utils.data.Dataset):
    def __add_dir(self, target_dir, label_add=None, prob=0.1):
        """
        Scans a directory for image files and valid json files and adds them to the dataset
        :param target_dir: Target directory that contains "items" and "json" subdirectories
        """
        for subdir in os.walk(os.path.join(target_dir)):
            for file in subdir[2]:
                try:
                    if np.random.rand() > prob:
                        continue
                    target_file = os.path.join(target_dir, file)
                    Image.open(target_file)
                    self.files.append(target_file)
                    if label_add is None:
                        raise Exception
                    else:
                        self.labels.append(label_add)
                except:
                    # We just ignore files that are not images or do not have a json file
                    continue
        assert len(self.files) == len(self.labels)

    def __init__(
        self, dataset_dir, train=False, center_crop=448, augmentation=NoAugmentation()
    ):
        """
        Constructor
        :param dataset_dir: Directory that contains "items" and "json" subdirectories
        :param center_crop: Size of the crop window in pixels
        """
        super().__init__()
        self.train = train
        self.files = []
        self.labels = []
        self.dataset_dir = dataset_dir
        if dataset_dir is not None:
            self.__add_dir(dataset_dir, label_add=1)
        self.center_crop = center_crop
        self.augmentation = augmentation

        # Preprocessing pipeline
        crop_padding = 128
        composition = []
        if train:
            if (
                augmentation.downscaling_width != 0
                and augmentation.downscaling_height != 0
            ):
                target_size = [
                    augmentation.downscaling_height,
                    augmentation.downscaling_width,
                ]
                composition.append(transforms.Resize(size=target_size))
            if augmentation.random_rotation_angle > 0:
                composition.append(
                    transforms.RandomRotation(augmentation.random_rotation_angle)
                )
            if augmentation.enable_random_cropping:
                composition.append(
                    transforms.CenterCrop(center_crop + crop_padding * 2)
                )
                composition.append(transforms.RandomCrop(center_crop))
            if augmentation.enable_vertical_mirroring:
                composition.append(transforms.RandomVerticalFlip())
            if augmentation.enable_horizontal_mirroring:
                composition.append(transforms.RandomHorizontalFlip())

        composition.append(transforms.ToTensor())
        # Normalization values taken from torchvision resnet18 docs
        composition.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
        self.transform = transforms.Compose(composition)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        # Load image using PIL
        img = Image.open(self.files[item]).convert("RGB")
        img = self.transform(img)

        return img, self.labels[item]

    def __network_pre_transforms(self):
        return [
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
