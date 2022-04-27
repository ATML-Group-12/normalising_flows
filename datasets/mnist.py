import torch
import torchvision


class BinarisedMNIST(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, precompute=True):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        if self.train:
            self.data = torchvision.datasets.MNIST(
                self.root,
                train=True,
                download=self.download,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Lambda(lambda x: torch.bernoulli(x)),
                ])
            )
        else:
            self.data = torchvision.datasets.MNIST(
                self.root,
                train=False,
                download=self.download,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Lambda(lambda x: torch.bernoulli(x)),
                ])
            )
        if precompute:
            res = []
            for i in range(len(self.data)):
                res.append(self.data[i])
            self.data = res
            

    def preproc(self, t):
        img, target = t

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __getitem__(self, index):
        return self.preproc(self.data[index])

    def __len__(self):
        return len(self.data)
