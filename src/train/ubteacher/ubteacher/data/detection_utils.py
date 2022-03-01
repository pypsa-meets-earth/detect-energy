# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import torchvision.transforms as transforms
from ubteacher.data.transforms.augmentation_impl import (
    GaussianBlur,
)


def build_strong_pypsa_augmentation(cfg, is_train,
                jitter_brightness=0.4,        
                jitter_contrast=0.4,        
                jitter_saturation=0.4,        
                jitter_hue=0.1,        
                jitter_p=0.8,
                greyscale_p=0.2,
                blur_mean=0.1,
                blur_var=2.0,
                blur_p=0.5,
                    ):

    logger = logging.getLogger(__name__)
    augmentation = []

    if is_train:
        # This is simialr to SimCLR https://arxiv.org/abs/2002.05709 (except the cropping)
        augmentation.append(
            transforms.RandomApply([transforms.ColorJitter(
                                            jitter_brightness,
                                            jitter_contrast,
                                            jitter_saturation,
                                            jitter_hue, 
                                    )], p=jitter_p)
        )
        augmentation.append(transforms.RandomGrayscale(p=greyscale_p))
        augmentation.append(transforms.RandomApply([GaussianBlur([blur_mean, blur_var])], p=blur_p))

        other_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.ToPILImage(),
            ]
        )
        augmentation.append(other_transform)

        logger.info("Augmentations used in training: " + str(augmentation))
    return transforms.Compose(augmentation)



def build_strong_augmentation(cfg, is_train):
    """
    Create a list of :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """

    logger = logging.getLogger(__name__)
    augmentation = []
    if is_train:
        # This is simialr to SimCLR https://arxiv.org/abs/2002.05709
        augmentation.append(
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        )
        augmentation.append(transforms.RandomGrayscale(p=0.2))
        augmentation.append(transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5))

        randcrop_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomErasing(
                    p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random"
                ),
                transforms.RandomErasing(
                    p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6), value="random"
                ),
                transforms.RandomErasing(
                    p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8), value="random"
                ),
                transforms.ToPILImage(),
            ]
        )
        augmentation.append(randcrop_transform)

        logger.info("Augmentations used in training: " + str(augmentation))
    return transforms.Compose(augmentation)