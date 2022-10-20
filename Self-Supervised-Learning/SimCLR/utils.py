from torchvision import datasets, transforms
import torch

def dataloader(batch_size, device):
    transform = transforms.Compose([
                transforms.RandomResizedCrop(size=(72,72)), # Random Crop and resize
                                                          # to input size
                transforms.Resize(size=(224, 224)),
                transforms.RandomHorizontalFlip(p=0.5), # Random Flip
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=.5, hue=.1), # Color Distortion
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 3)), # Random Gaussian Blur
                transforms.ToTensor(), # Transfor PIL image to Tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
    )

    dataset1 = datasets.ImageFolder('data/img', transform=transform)
    dataset2 = datasets.ImageFolder('data/img', transform=transform)

    dataloader1 = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, shuffle=False)
    dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=batch_size, shuffle=False)

    return dataloader1, dataloader2