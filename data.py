from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import utils
import numpy as np

transformations = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])

def DataGenTrain(key):
    if key == 'MNIST':
        yield datasets.MNIST('../../shared/DistLearnsup', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ]))
    elif key == 'CIFAR10':
        yield datasets.CIFAR10('../../shared/DistLearnsup', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   # From PyTorch Tutorial: 
                                   # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html?highlight=cifar
                                   # transforms.Normalize((0.5, 0.5, 0.5),
                                   #                      (0.5, 0.5, 0.5)),
                                   # From cdshd Repo:
                                   # transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                        # (0.2023, 0.1994, 0.2010)),
                                   # From Pytorch Forum:
                                   # https://github.com/kuangliu/pytorch-cifar/issues/19
                                   # https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
                                   transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                        (0.247, 0.243, 0.261)),
                               ]))
    elif key == 'CIFAR100':
        yield datasets.CIFAR100('../../shared/DistLearnsup', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5),
                                                        (0.5, 0.5, 0.5)),
                               ]))


def DataGenTest(key):
    if key == 'MNIST':
        yield datasets.MNIST('../../shared/DistLearnsup', train=False, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ]))
    elif key == 'CIFAR10':
        yield datasets.CIFAR10('../../shared/DistLearnsup', train=False, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   # From PyTorch Tutorial: 
                                   # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html?highlight=cifar
                                   # transforms.Normalize((0.5, 0.5, 0.5),
                                   #                      (0.5, 0.5, 0.5)),
                                   # From cdshd Repo:
                                   # transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                        # (0.2023, 0.1994, 0.2010)),
                                   # From Pytorch Forum:
                                   # https://github.com/kuangliu/pytorch-cifar/issues/19
                                   # https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
                                   transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                        (0.247, 0.243, 0.261)),
                               ]))
    elif key == 'CIFAR100':
        yield datasets.CIFAR100('../../shared/DistLearnsup', train=False, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5),
                                                        (0.5, 0.5, 0.5)),
                               ]))

DataYielderTrain = lambda key : next(DataGenTrain(key))
DataYielderTest = lambda key : next(DataGenTest(key))

def LoadData(**kwargs):
    # train_loader = None
    wsize = kwargs.get('wsize', -1)
    assert wsize > -1
    myrank = kwargs.get('rank', -1)
    assert myrank > -1
    assert myrank < wsize
    #  Criterion for dealing with federated learning and collaborative learning both
    #  for collaborative learning server_rank will be -1
    server_rank = kwargs.get('server_rank',-1)
    num_workers = kwargs.get('num_workers', wsize)
    if not server_rank in list(range(wsize)):
        assert num_workers == wsize
    batch_size = kwargs.get('batch_size', 128)
    if (myrank != server_rank):
        dataset = DataYielderTrain(kwargs['data'])
        train_loader = utils.get_partition_dataloader(dataset, **kwargs)
        print(kwargs['data'], 'Loaded and Partitioned, total training samples for worker rank {}: '.format(myrank), len(train_loader.dataset))
    return train_loader


def LoadTestData(**kwargs):
    # test_loader = None
    wsize = kwargs.get('wsize', -1)
    assert wsize > -1
    myrank = kwargs.get('rank', -1)
    assert myrank > -1
    assert myrank < wsize
    #  Criterion for dealing with federated learning and collaborative learning both
    #  for collaborative learning server_rank will be -1
    server_rank = kwargs.get('server_rank',-1)
    num_workers = kwargs.get('num_workers', wsize)
    if not server_rank in list(range(wsize)):
        assert num_workers == wsize
    batch_size = kwargs.get('batch_size', 128)
    if (myrank != server_rank):
        test_data = DataYielderTest(kwargs['data'])
        test_loader = DataLoader(test_data, kwargs['batch_size'], shuffle=True)
        print(kwargs['data'], 'Loaded and Partitioned, total testing samples for worker rank {}: '.format(myrank), len(test_loader.dataset))
        data_dim = np.shape(test_data[0][0].numpy())
    return test_loader, data_dim

