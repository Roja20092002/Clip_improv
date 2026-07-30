from torchvision import transforms


def get_default_image_transform(image_size=224):
    """
    Standard ImageNet-style preprocessing.
    Used for ResNet and UNITER-based models.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
