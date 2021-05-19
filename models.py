import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from Optimizers import *
from torch.optim.lr_scheduler import StepLR
import Models

resnet_list = {
    'resnet20': Models.resnet20,
    'resnet32': Models.resnet32,
    'resnet44': Models.resnet44,
    'resnet56': Models.resnet56,
    'resnet110': Models.resnet110,
    'resnet1202': Models.resnet1202
    }

vgg_list = ['VGG11', 'VGG13', 'VGG16', 'VGG19']

class CNN(nn.Module):

    def __init__(self,ch_dim=1, fc_nodes=576, num_classes=10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(ch_dim, 32, 3, padding=1),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.25),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.MaxPool2d(2, 2),
            )
        self.classifier = nn.Linear(fc_nodes, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class mnist_CNN(nn.Module):

    def __init__(self,ch_dim, num_classes=10):
        super(mnist_CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(ch_dim, 32, 3, padding=1),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.25),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.MaxPool2d(2, 2),
            )
        self.classifier = nn.Linear(3136, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Big_CNN(nn.Module):

    def __init__(self,ch_dim, num_classes=10):
        super(Big_CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(ch_dim, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class FCN(nn.Module):
    def __init__(self, input_dim, nb_classes):
        super(FCN, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, nb_classes)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.fc1(x)
        for _ in range(20):
            x = self.fc2(x)
        x = self.fc3(x)
        return x


class LR(nn.Module):
    def __init__(self, input_dim, nb_classes):
        super(LR, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, nb_classes, bias=False)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.fc1(x)
        return x



def BuildModel(model_name, img_dim, fc_nodes, nb_classes):

    if model_name == "CNN":
        model = CNN(img_dim, fc_nodes, nb_classes)
    elif model_name == "Big_CNN":
        model = Big_CNN(img_dim, nb_classes)
    elif model_name == "FCN":
        model = FCN(img_dim, nb_classes)
    elif model_name == "LR":
        model = LR(img_dim, nb_classes)
    elif model_name == "PreResNet110" or model_name == "WideResNet28x10":
        model_cfg = getattr(Models, model_name)
        model = model_cfg.base(*model_cfg.args, num_classes=nb_classes, **model_cfg.kwargs)
    elif model_name in resnet_list.keys():
        model = resnet_list[model_name](nb_classes)
    elif model_name in vgg_list:
        model = Models.VGG(model_name, nb_classes)
    elif model_name == "mnist_CNN":
        model = mnist_CNN(img_dim, nb_classes)

    return model



def LoadModel(img_dim, **kwargs):
    #  Get required meta data
    wsize = kwargs.get('wsize', -1)
    assert wsize > -1
    myrank = kwargs.get('rank', -1)
    assert myrank > -1
    assert myrank < wsize
    #  Criterion for dealing with federated learning and collaborative learning both
    #  for collaborative learning server_rank will be -1
    server_rank = kwargs.get('server_rank',-1)
    num_workers = kwargs.get('num_workers', wsize)
    device = kwargs.get('device')

    modelkey = kwargs['model_arch']
    optkey = kwargs['optimizer']

    # Setting params for CNN:
    if kwargs["data"] == "MNIST" or kwargs["data"] == 'semeion':
        ch_dim = 1
        fc_nodes = 576
        num_classes = 10
    elif kwargs["data"] == "CIFAR10":
        ch_dim = 3
        fc_nodes = 1024
        num_classes = 10
    elif kwargs["data"] == "CIFAR100":
        ch_dim = 3
        fc_nodes = 1024
        num_classes = 100

    if modelkey == 'LR':
        ch_dim = img_dim[0]*img_dim[1]*img_dim[2]

    # Build model
    model = BuildModel(modelkey, ch_dim, fc_nodes, num_classes).to(device)


    # initialize criterion
    criterion = nn.CrossEntropyLoss()

    # initialize parameters
    for param in model.parameters():
        param.grad = torch.zeros(param.size(), requires_grad=True).to(device)
        param.grad.data.zero_()

    # initialize optimizer
    if optkey=='nesterov':
        optimizer = optim.SGD(model.parameters(), lr=kwargs.get('lr', 0.01), momentum=kwargs.get('momentum', 0.9), nesterov=True)
    elif optkey=='sgd':
        optimizer = optim.SGD(model.parameters(), lr=kwargs.get('lr', 0.0001), momentum=kwargs.get('momentum', 0.9), nesterov=False)
    elif optkey=='adam':
        optimizer = optim.Adam(model.parameters(), lr=kwargs.get('lr', 0.01))
    elif optkey=='adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=kwargs.get('lr', 0.01))
    elif optkey=='adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=kwargs.get('lr', 1.0))
    elif optkey=='CDSGD':
        optimizer = CDSGD(model.parameters(), kwargs, lr=kwargs.get('lr', 0.01), momentum=kwargs.get('momentum', 0.9))
    elif optkey=='CDMSGD':
        optimizer = CDMSGD(model.parameters(), kwargs, lr=kwargs.get('lr', 0.01), momentum=kwargs.get('momentum', 0.9))
    elif optkey == 'SGA':
        optimizer = SGA(model.parameters(), lr=kwargs.get('lr', 0.01))
    elif optkey == 'CGA':
        optimizer = CGA(model.parameters(), kwargs, lr=kwargs.get('lr', 0.01), momentum=kwargs.get('momentum', 0.9))
    elif optkey == 'CompCGA':
        optimizer = CompCGA(model.parameters(), kwargs, lr=kwargs.get('lr', 0.01), momentum=kwargs.get('momentum', 0.9))
    elif optkey == 'SGP':
        optimizer = SGP(model.parameters(), kwargs, lr=kwargs.get('lr', 0.01), momentum=kwargs.get('momentum', 0.9))
    elif optkey == 'SwarmSGD':
        optimizer = SwarmSGD(model.parameters(), kwargs, lr=kwargs.get('lr', 0.01), momentum=kwargs.get('momentum', 0.9))

    # Implement LR scheduler
    if kwargs['scheduler']:
        scheduler = StepLR(optimizer, step_size=kwargs['LR_sche_step'], gamma=kwargs['LR_sche_lamb'])
        if kwargs['verbose'] >= 1:
            if kwargs['dist'].get_rank() == 0:
                print("Applying Step LR scheduler with step_size = %s and lr_lambda = %s"%(kwargs['LR_sche_step'], kwargs['LR_sche_lamb']))
    else:
        scheduler = None
        if kwargs['dist'].get_rank() == 0:
            print('Not applying LR scheduler.')

    return model, optimizer, criterion, scheduler
