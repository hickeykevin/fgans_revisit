import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.transforms.functional import to_pil_image, pil_to_tensor

def CIFAR10DataModule(params, PATH):

    transform = transforms.Compose([
        transforms.Lambda(
            lambda img: img.resize((params['imsize'], params['imsize']), Image.BICUBIC) 
            ),
        transforms.CenterCrop(params['imsize']),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
             (0.5, 0.5, 0.5))
        ])

    train_set = torchvision.datasets.CIFAR10(root=PATH,train=True,
                                          download=True,transform=transform)
    train_loader = DataLoader(train_set,batch_size=params['bsize'],
                                              shuffle =True, num_workers=params['nwork'], drop_last=True)
    test_set = torchvision.datasets.CIFAR10(root=PATH,train=False,
                                         download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=params['bsize'],
                                              shuffle = False, num_workers=params['nwork'], drop_last=True)
    return train_set, train_loader, test_set, test_loader