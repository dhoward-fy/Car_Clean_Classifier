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
import dtlpy as dataloop
from augmentation import NoAugmentation
import json
import random


class CarState(IntEnum):
    CAR_CLEAN = (0,)
    CAR_DIRTY = 1

    def __str__(self):
        strs = ["Clean", "Dirty"]
        return strs[self.value]


class DataloopDataset(torch.utils.data.Dataset):
    def __add_dir(self, target_dir, email, password, project, dataset, seed=42):
        """
        Scans a directory for image files and valid json files and adds them to the dataset
        :param target_dir: Target directory that contains "items" and "json" subdirectories
        """
        dataloop.login_m2m(email=email, password=password)
        proj = dataloop.projects.get(project)
        dataset = proj.datasets.get(dataset)

        seen_filenames = set()
        all_names = {}
        self.random_file_list = set()
        random.seed(seed)
        class_to_label = {"clean": 0, "light_dirt": 1, "heavy_dirt": 1}

        # Get one image from each vehicle at random
        pages = dataset.items.list()
        for page in pages:
            for item in page:
                all_names[item.id] = item.name
        for name in all_names:
            filename = all_names[name]
            if "_" in filename:
                unique_filename = filename[: filename.rfind("_")]
            else:
                unique_filename = filename
            if unique_filename not in seen_filenames:
                seen_filenames.add(unique_filename)
                files_with_prefix = [
                    f for f in all_names if all_names[f].startswith(unique_filename)
                ]
                self.random_file_list.add(random.choice(files_with_prefix))

        # Get data for randomly selected vehicle images
        for id in self.random_file_list:
            item = dataset.items.get(item_id=id)
            try:
                item.download(local_path=f"{target_dir}/")
                target_file = os.path.join(target_dir, item.name)

                item.annotations.download(filepath=f"{target_dir}/")
                annotation_file_name = os.path.join(
                    target_dir, os.path.splitext(item.name)[0] + ".json"
                )

                with open(annotation_file_name) as annotation_file:
                    json_data = json.load(annotation_file)
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
                    self.file_names.append(item.name)
            except FileNotFoundError:
                # We just ignore files that are not images or do not have a json file
                continue
        assert len(self.files) == len(self.labels)

    def __init__(
        self,
        dataset_dir,
        email,
        password,
        project,
        dataset,
        train=False,
        center_crop=448,
        augmentation=NoAugmentation(),
    ):
        """
        Constructor
        :param dataset_dir: Directory where dataloop data will be saved
        :param email: Dataloop sign in email
        :param password: Dataloop password
        :param project: Dataloop project name
        :param project: Dataloop dataset name
        :param center_crop: Size of the crop window in pixels
        """
        super().__init__()
        self.train = train
        self.files = []
        self.labels = []
        self.file_names = []
        self.train_indices = set()
        self.dataset_dir = dataset_dir
        if dataset_dir is not None:
            self.__add_dir(dataset_dir, email, password, project, dataset)
        self.center_crop = center_crop
        self.augmentation = augmentation

        # Preprocessing pipeline
        composition = []
        if self.train:
            if augmentation.enable_center_cropping:
                composition.append(transforms.CenterCrop(center_crop))
            if augmentation.random_rotation_angle > 0:
                composition.append(
                    transforms.RandomRotation(augmentation.random_rotation_angle)
                )
            if augmentation.enable_horizontal_mirroring:
                composition.append(transforms.RandomHorizontalFlip())
            if (
                augmentation.downscaling_width != 0
                and augmentation.downscaling_height != 0
            ):
                target_size = [
                    augmentation.downscaling_height,
                    augmentation.downscaling_width,
                ]
                composition.append(transforms.Resize(size=target_size))
            pass
        else:
            if augmentation.enable_center_cropping:
                composition.append(transforms.CenterCrop(center_crop))
            if (
                augmentation.downscaling_width != 0
                and augmentation.downscaling_height != 0
            ):
                target_size = [
                    augmentation.downscaling_height,
                    augmentation.downscaling_width,
                ]
                composition.append(transforms.Resize(size=target_size))

        composition.append(transforms.ToTensor())
        self.transform = transforms.Compose(composition)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        # Load image using PIL
        img = Image.open(self.files[item]).convert("RGB")
        img = self.transform(img)

        # Only use the label of the biggest bounding box
        return img, self.labels[item][1], self.file_names[item]


class DataloopDatasetDirectory(torch.utils.data.Dataset):
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
        composition = []
        if self.train:
            if augmentation.enable_center_cropping:
                composition.append(transforms.CenterCrop(center_crop))
            if augmentation.random_rotation_angle > 0:
                composition.append(
                    transforms.RandomRotation(augmentation.random_rotation_angle)
                )
            if augmentation.enable_horizontal_mirroring:
                composition.append(transforms.RandomHorizontalFlip())
            if (
                augmentation.downscaling_width != 0
                and augmentation.downscaling_height != 0
            ):
                target_size = [
                    augmentation.downscaling_height,
                    augmentation.downscaling_width,
                ]
                composition.append(transforms.Resize(size=target_size))
            pass
        else:
            if augmentation.enable_center_cropping:
                composition.append(transforms.CenterCrop(center_crop))
            if (
                augmentation.downscaling_width != 0
                and augmentation.downscaling_height != 0
            ):
                target_size = [
                    augmentation.downscaling_height,
                    augmentation.downscaling_width,
                ]
                composition.append(transforms.Resize(size=target_size))

        composition.append(transforms.ToTensor())
        self.transform = transforms.Compose(composition)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        # Load image using PIL
        img = Image.open(self.files[item]).convert("RGB")
        img = self.transform(img)

        # Only use the label of the biggest bounding box
        return img, self.labels[item][1], self.file_names[item]


class DataloopFiles(torch.utils.data.Dataset):
    def __add_dir(self, target_dir, seed=42):
        """
        Scans a directory for image files and valid json files and adds them to the dataset
        :param target_dir: Target directory that contains "items" and "json" subdirectories
        """
        random.seed(seed)
        class_to_label = {"clean": 0, "light_dirt": 1, "heavy_dirt": 1}

        # Get data for selected vehicle images
        for file in self.file_list:
            try:
                target_file = os.path.join(target_dir, "items", file)
                Image.open(target_file)
                # Try to open the json file
                annotation_file_name = os.path.join(
                    target_dir, "json", os.path.splitext(file)[0] + ".json"
                )
                with open(annotation_file_name) as annotation_file:
                    json_data = json.load(annotation_file)
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
        self,
        dataset_dir,
        file_list,
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
        self.file_list = file_list
        if dataset_dir is not None:
            self.__add_dir(dataset_dir)
        self.center_crop = center_crop
        self.augmentation = augmentation

        # Preprocessing pipeline
        composition = []
        if self.train:
            if augmentation.enable_center_cropping:
                composition.append(transforms.CenterCrop(center_crop))
            if augmentation.random_rotation_angle > 0:
                composition.append(
                    transforms.RandomRotation(augmentation.random_rotation_angle)
                )
            if augmentation.enable_horizontal_mirroring:
                composition.append(transforms.RandomHorizontalFlip())
            if (
                augmentation.downscaling_width != 0
                and augmentation.downscaling_height != 0
            ):
                target_size = [
                    augmentation.downscaling_height,
                    augmentation.downscaling_width,
                ]
                composition.append(transforms.Resize(size=target_size))
            pass
        else:
            if augmentation.enable_center_cropping:
                composition.append(transforms.CenterCrop(center_crop))
            if (
                augmentation.downscaling_width != 0
                and augmentation.downscaling_height != 0
            ):
                target_size = [
                    augmentation.downscaling_height,
                    augmentation.downscaling_width,
                ]
                composition.append(transforms.Resize(size=target_size))

        composition.append(transforms.ToTensor())
        self.transform = transforms.Compose(composition)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        # Load image using PIL
        img = Image.open(self.files[item]).convert("RGB")
        img = self.transform(img)

        # Only use the label of the biggest bounding box
        return img, self.labels[item][1], self.file_names[item]
