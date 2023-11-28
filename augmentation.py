# vim: set ts=4 sw=4 :
# _*_ coding: utf-8 _*_
#
# Copyright (c) 2023 Fyusion Inc.


class Augmentation:
    def __init__(
        self,
        enable_center_cropping=False,
        enable_horizontal_mirroring=False,
        random_rotation_angle=15,
        noise_amount=0,
        downscaling_width=0,
        downscaling_height=0,
        center_crop=0,
    ):
        self.enable_center_cropping = enable_center_cropping
        self.enable_horizontal_mirroring = enable_horizontal_mirroring
        self.random_rotation_angle = random_rotation_angle
        self.noise_amount = noise_amount
        self.downscaling_width = downscaling_width
        self.downscaling_height = downscaling_height
        self.center_crop = center_crop


class NoAugmentation(Augmentation):
    def __init__(self):
        super().__init__(
            enable_center_cropping=False,
            enable_horizontal_mirroring=False,
            random_rotation_angle=0,
            noise_amount=0,
        )
