import torchvision.transforms as tfms


def Image_classification_transform(image_size:tuple[int,int], augmentation:bool=False, mean:list[float] = [0.5, 0.5, 0.5], std: list[float] = [0.5, 0.5, 0.5] ) -> tfms.Compose:

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
    print(Image_classification_transform((224, 224), augmentation=False))

