from PIL import ImageFilter
import random
import logging
from torchvision.transforms import transforms


class GaussianBlur:
    """
    Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709
    https://github.com/facebookresearch/moco/blob/master/moco/loader.py
    """

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


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
        
        augmentation.append(
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
                )
        augmentation.append(transforms.RandomGrayscale(p=0.2))
        augmentation.append(transforms.RandomApply([GaussianBlur([0.1, 1.5])], p=1))

        datatype_transform = transforms.Compose([transforms.ToTensor(), transforms.ToPILImage(),])
        augmentation.append(datatype_transform)
        logger.info("Augmentations used in training: " + str(augmentation))

    return transforms.Compose(augmentation)