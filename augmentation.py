# vim: set ts=4 sw=4 :
# _*_ coding: utf-8 _*_
#
# Copyright (c) 2023 Fyusion Inc.


class Augmentation:
    def __init__(
        self,
        enable_random_cropping=False,
        enable_vertical_mirroring=False,
        enable_horizontal_mirroring=False,
        random_rotation_angle=15,
        noise_amount=0,
        downscaling_width=0,
        downscaling_height=0,
    ):
        self.enable_random_cropping = enable_random_cropping
        self.enable_vertical_mirroring = enable_vertical_mirroring
        self.enable_horizontal_mirroring = enable_horizontal_mirroring
        self.random_rotation_angle = random_rotation_angle
        self.noise_amount = noise_amount
        self.downscaling_width = downscaling_width
        self.downscaling_height = downscaling_height


class NoAugmentation(Augmentation):
    def __init__(self):
        super().__init__(
            enable_random_cropping=False,
            enable_vertical_mirroring=False,
            random_rotation_angle=0,
            noise_amount=0,
        )
