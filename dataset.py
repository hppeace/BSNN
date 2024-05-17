import tonic
import tonic.transforms as transforms

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

def CIFAR10DVSLoader(T:int=10, batch_size:int=50):

    size = 48

    row_transform = transforms.Compose([tonic.transforms.ToFrame(sensor_size=tonic.datasets.CIFAR10DVS.sensor_size, n_time_bins=10),])

    trainset = tonic.datasets.CIFAR10DVS(save_to='/home/peace/code/DataSets', transform=row_transform)
    testset = tonic.datasets.CIFAR10DVS(save_to='/home/peace/code/DataSets', transform=row_transform)

    train_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: torch.nn.functional.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
        transforms.RandomCrop(size, padding=4),
        transforms.Normalize((0.2728, 0.1295), (0.2225, 0.1290))
    ])
    test_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: torch.nn.functional.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
        transforms.Normalize((0.2728, 0.1295), (0.2225, 0.1290))
    ])

    cached_trainset = tonic.DiskCachedDataset(trainset, cache_path='/home/peace/code/DataSets/CIFAR10DVS/cache/train', transform=train_transform, num_copies=10)
    cached_testset = tonic.DiskCachedDataset(testset, cache_path='/home/peace/code/DataSets/CIFAR10DVS/cache/test', transform=test_transform,num_copies=10)

    num_train = len(trainset)
    num_per_cls = num_train // 10
    indices_train, indices_test = [], []
    portion = 0.9
    for i in range(10):
        indices_train.extend(
            list(range(i * num_per_cls, round(i * num_per_cls + num_per_cls * portion))))
        indices_test.extend(
            list(range(round(i * num_per_cls + num_per_cls * portion), (i + 1) * num_per_cls)))

    trainloader = torch.utils.data.DataLoader(
        cached_trainset, batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices_train),
        pin_memory=True, num_workers=8, collate_fn=tonic.collation.PadTensors(batch_first=False)
    )

    testloader = torch.utils.data.DataLoader(
        cached_testset, batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices_test),
        pin_memory=True, num_workers=4, collate_fn=tonic.collation.PadTensors(batch_first=False)
    )

    return trainloader, testloader

def DVSGestureLoader(T:int=25, batch_size:int=24):

    size = 32

    row_transform = transforms.Compose([tonic.transforms.ToFrame(sensor_size=tonic.datasets.DVSGesture.sensor_size, n_time_bins=T),])

    trainset = tonic.datasets.DVSGesture(save_to='/home/peace/code/DataSets', transform=row_transform, train=True)
    testset = tonic.datasets.DVSGesture(save_to='/home/peace/code/DataSets', transform=row_transform, train=False)

    train_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: torch.nn.functional.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
        transforms.RandomCrop(size, padding=size // 12),
    ])
    test_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: torch.nn.functional.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
    ])

    cached_trainset = tonic.DiskCachedDataset(trainset, cache_path='/home/peace/code/DataSets/DVSGesture/cache/train', transform=train_transform, num_copies=10)
    cached_testset = tonic.DiskCachedDataset(testset, cache_path='/home/peace/code/DataSets/DVSGesture/cache/test', transform=test_transform, num_copies=10)

    trainloader = DataLoader(cached_trainset, batch_size=batch_size, pin_memory=True, num_workers=8,shuffle=True,collate_fn=tonic.collation.PadTensors(batch_first=False))
    testloader = DataLoader(cached_testset, batch_size=batch_size, pin_memory=True,  num_workers=4,shuffle=False,collate_fn=tonic.collation.PadTensors(batch_first=False))

    return trainloader, testloader

def NMNISTLoader(T:int = 18, batch_size:int =100):

    transform = transforms.Compose([tonic.transforms.Denoise(filter_time=10000), tonic.transforms.ToFrame( sensor_size=tonic.datasets.NMNIST.sensor_size, n_time_bins=T)])

    trainset = tonic.datasets.NMNIST(save_to='/home/peace/code/DataSets', train=True, transform=transform)
    testset = tonic.datasets.NMNIST(save_to='/home/peace/code/DataSets', train=False, transform=transform)

    cached_trainset = tonic.DiskCachedDataset(trainset, cache_path='/home/peace/code/DataSets/NMNIST/cache/train' )
    cached_testset = tonic.DiskCachedDataset(testset, cache_path='/home/peace/code/DataSets/NMNIST/cache/test' )

    trainloader = DataLoader(cached_trainset, batch_size=batch_size, pin_memory=True, num_workers=8,shuffle=True,collate_fn=tonic.collation.PadTensors(batch_first=False))
    testloader = DataLoader(cached_testset, batch_size=batch_size, pin_memory=True,  num_workers=4,shuffle=False,collate_fn=tonic.collation.PadTensors(batch_first=False))

    return trainloader, testloader

def SetLoader(config):
    if config["dataset"] == 'CIFAR10DVS':
        return CIFAR10DVSLoader(T=config["T"], batch_size=config["batch_size"])
    elif config["dataset"] == 'DVSGesture':
        return DVSGestureLoader(T=config["T"], batch_size=config["batch_size"])
    elif config["dataset"] == 'NMNIST':
        return NMNISTLoader(T=config["T"], batch_size=config["batch_size"])
    else:
        raise NotImplementedError
    
if __name__ == '__main__':
    trainloader, testloader = CIFAR10DVSLoader()
    for i, (inputs, targets) in enumerate(trainloader):
        print(inputs.shape, targets.shape)

    for i, (inputs, targets) in enumerate(testloader):
        print(inputs.shape, targets.shape)
