import torch
import torchvision


class CustomCIFAR(torch.utils.data.Dataset):
    """
    Pytorch CIFAR dataset
    """

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.eps = 0.0001
        if self.train:
            self.train_data = torchvision.datasets.CIFAR10(
                self.root,
                train=True,
                download=self.download,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.RandomCrop(size=8),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Lambda(lambda x: torch.clamp(x, self.eps, 1 - self.eps))
                ])
            )
        else:
            self.test_data = torchvision.datasets.CIFAR10(
                self.root,
                train=False,
                download=self.download,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.RandomCrop(size=8),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Lambda(lambda x: torch.clamp(x, self.eps, 1 - self.eps))

                ])
            )

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index]
        else:
            img, target = self.test_data[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return
