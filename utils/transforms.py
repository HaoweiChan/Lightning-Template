import torch

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            return torch.flip(x, dims=[-1])
        return x

class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, x):
        return (x - self.mean) / self.std

class ToTensor:
    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            return x
        # Assuming x is a numpy array
        return torch.from_numpy(x).float()

def get_transforms(mode='train'):
    if mode == 'train':
        return Compose([
            ToTensor(),
            RandomHorizontalFlip(p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
        ])
    else:
        return Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
        ]) 