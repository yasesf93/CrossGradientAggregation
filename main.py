import os
import utils
import models
import data
import json
from collaborative import Collab
import torch
import torch.distributed as dist
import torch.nn.functional as F
import argparse
import numpy as np


if __name__ == '__main__':
    # initialize MPI environment 
    # dist.init_process_group(backend="mpi")
    dist.init_process_group(backend="gloo", init_method='env://')
    rank = dist.get_rank()
    wsize = dist.get_world_size()  # number of processes = num_workers + 1   
    server_rank = -1 # this process is the server

    #############################################################################################################################
    #                                          setup code shared by workers and server
    #############################################################################################################################

    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--use_cuda', action="store_true", help='Use CUDA if available')
    parser.add_argument('--data',type=str,default='CIFAR10', choices=("CIFAR10", "CIFAR100", "MNIST"), help='Define the data used for training')
    parser.add_argument('--data_dist',type=str,default='iid', choices=("iid", "non-iid"), help='Define the data distribution')
    parser.add_argument('--model',type=str,default='CNN', help='Define the model used for training',
        choices=("LR","CNN","Big_CNN","FCN","stl10_CNN","mnist_CNN","PreResNet110","WideResNet28x10",
                'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202',
                'VGG11', 'VGG13', 'VGG16', 'VGG19'))
    parser.add_argument('--opt', type=str, default='CDSGD', help='Optimizer choices', 
                        choices=('sgd','adam','adagrad','adadelta','nesterov','CDSGD','CDMSGD','SGA','CGA','SGP','SwarmSGD', 'CompCGA'))
    parser.add_argument('--batch_size', type=int, default=128, help='Define batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Define num epochs for training')
    parser.add_argument('--experiment', type=int, default=1, help='Experiment number of connectivity json')

    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',help='Momentum (default: 0.9)')
    parser.add_argument('-v','--verbosity', type=int, default=2, help='verbosity of the code for debugging, 0==>No outputs, 1==>graph level outputs, 2==>agent level outputs')
    parser.add_argument('-log','--log_interval', type=int, default=10,help='How many epochs to wait before logging training results and models')

    parser.add_argument('--scheduler', action='store_true', default=True, help='Apply LR scheduler: step')
    parser.add_argument('-sche_step','--LR_sche_step', type=int, default=1, help='Stepsize for LR scheduler') # For StepLR
    parser.add_argument('-sche_lamb','--LR_sche_lamb', type=float, default=0.981, help='Lambda for LR scheduler') # For StepLR

    # ================ For torch.distributed.launch multi-process argument ================
    parser.add_argument('--local_rank', type=int, help='Required argument for torch.distributed.launch, similar as rank')

    # ================ Not Implemented Yet ================
    parser.add_argument('-w','--omega', default=0.5, type=float, help='omega value of the generalized consensus')

    # ================================================

    args = parser.parse_args()

    # Setting device to local rank (torch.distributed.launch)
    if args.use_cuda:
        devices = os.environ['CUDA_VISIBLE_DEVICES'].strip().split(',')
        print(devices,flush=True)
        per_device_ranks = int(wsize/len(devices)) + 1
        print('Device assignment: %s , %s'%(args.local_rank, int(args.local_rank/per_device_ranks)),flush=True)
        torch.cuda.set_device(int(args.local_rank/per_device_ranks))


    torch.manual_seed(123)
    device = torch.device("cuda" if args.use_cuda else "cpu")

    # with open('Connectivity/%s_%s.json'%(wsize,args.experiment), 'r') as f:
    with open('New_connectivity/%s_%s.json'%(wsize,args.experiment), 'r') as f:
        cdict = json.load(f)

    neighbors = [i[0] for i in enumerate(cdict['connectivity'][rank]) if i[1]>=0] # will include all rank, but required 

    # Create save folder
    folder_name = os.path.join("log",str(wsize)+'_agents', args.data_dist, args.data, args.model,str(cdict['graph_type']),args.opt+'_'+str(args.momentum))
    if rank == 0:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

    argdict = {
        'dist':dist,
        'use_cuda':args.use_cuda,
        'data':args.data,
        'model_arch':args.model,
        'experiment':args.experiment,
        'graph_type':cdict['graph_type'],
        'data_dist':args.data_dist,
        'server_rank':server_rank,
        'epochs':args.epochs,
        'batch_size':args.batch_size,
        'optimizer':args.opt,
        'device':device,
        'wsize':wsize,
        'num_workers':wsize,
        'neighbors': neighbors,
        'pi':cdict['pi'][rank],
        'rank':rank,
        'lr':args.lr,
        'scheduler':args.scheduler,
        'LR_sche_step':args.LR_sche_step,
        'LR_sche_lamb':args.LR_sche_lamb,
        'momentum':args.momentum,
        'verbose':args.verbosity,
        'log_interval':args.log_interval,
        'log_folder':folder_name
        }
    print(f"Rank {rank}:",cdict['pi'][rank])
    #############################################################################################################################
    #                                                 workers' setup code
    ############################################################################################################################# 
    dataloader = data.LoadData(**argdict)
    testdataloader, data_dim = data.LoadTestData(**argdict)
    model, opt, criterion, LR_scheduler = models.LoadModel(data_dim, **argdict) # workers and servers all have a model!
    dist.barrier()
    trainer = Collab(dataloader, testdataloader, model, opt, criterion, LR_scheduler, **argdict)
    trainer.train()
