"""
This script defines different transform functions.
"""

import torchvision.transforms as tfms


def Image_classification_transform(image_size:tuple[int,int], augmentation:bool=False, mean:list[float] = [0.5, 0.5, 0.5], std: list[float] = [0.5, 0.5, 0.5] ) -> tfms.Compose:
    """
    Transformation function designed for Image classification.
    This function comes in two modes: with and without augmentation.
    :param image_size: size of input image.
    :param augmentation: augmentation mode.
    :param mean: list of mean of each channel.
    :param std: list of std of each channel.
    :return: transformation function.
    """
    if augmentation:
        trasnforms = tfms.Compose([
            tfms.Resize(image_size),
            tfms.ToTensor(),
            tfms.Normalize(mean=mean, std=std),
            tfms.RandomHorizontalFlip(),
            tfms.RandomVerticalFlip(),
            tfms.RandomRotation(degrees=10)
            ])
    else:
        trasnforms = tfms.Compose([
            tfms.Resize(image_size),
            tfms.ToTensor(),
            tfms.ToTensor(),
            tfms.Normalize(mean=mean, std=std)
            ])
    return trasnforms


if __name__ == '__main__':
    # for testing
    print(Image_classification_transform((224, 224), augmentation=False))

