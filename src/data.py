## Pytorch Imports
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#
# Data loading
#
def get_cifar10_loaders(batch_size: int, num_workers: int = 4):
    #Transform CIFAR10 data into tensor and transform to avoid generalisation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), #CIFAR10 normalization constants
                             (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        #Transform validation into tensor and normalize
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    #Get datasets
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    #Wrap in loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader
