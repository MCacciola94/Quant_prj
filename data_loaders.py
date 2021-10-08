import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch

available_datasets = ["Cifar10", "Cifar100", "Imagenet"]

def is_available(name):
    return name in available_datasets

def load_dataset(name, batch_size):

    if not(is_available(name)):
        print("Dataset requested not avilable")
        return None

    if name == "Cifar10":
        return cifar10_loader(batch_size)


def cifar10_loader(batch_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True)
        
    return {"train_loader": train_loader, "valid_loader": val_loader}
