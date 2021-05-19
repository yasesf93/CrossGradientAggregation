## Created by: Aditya Balu
import torch
from torch.optim import Optimizer
from collections import defaultdict
import numpy as np

class CDSGD(Optimizer):
    def __init__(self, params, kwargs, lr=0.01, momentum = None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
        super(CDSGD, self).__init__(params, defaults)

        # Extract required var.
        self.kwargs = kwargs
        self.pi = kwargs.get('pi')
        self.device = kwargs.get('device')

        self.local_neigh = np.argwhere(np.asarray(self.pi) != 0.0).ravel() # Find connected agents
        self.pi = [self.pi[i] for i in self.local_neigh] # simplified pi -- with only connected agents
        

    def __setstate__(self, state):
        super(CDSGD, self).__setstate__(state)


    # def step(self, opt_kwargs, closure=None):
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        # # Extract var. added thru Collab()
        dist = self.kwargs['dist']
        distgroup = self.kwargs['distgroup']
        neighbors = self.kwargs['neighbors']


        loss = None
        if closure is not None:
            loss = closure()
        if not isinstance(self.state, defaultdict):
            self.state = defaultdict(dict)
        for i, group in enumerate(self.param_groups):## Update rule
            for j, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                con_buf = [torch.zeros(p.data.size()).to(self.device) for _ in range(len(neighbors))] # Parameters placeholder
                # print('pdata',torch.norm(p.data), torch.norm(d_p))
                dist.all_gather(con_buf, p.data, group=distgroup) # Gather parameters from workers to con_buf

                buf = torch.zeros(p.data.size()).to(self.device)

                # Extract connected agents data only
                con_buf = [con_buf[i] for i in self.local_neigh]

                for pival, con_buf_agent in zip(self.pi, con_buf):
                    buf.add_(other=con_buf_agent, alpha=pival)

                p.data = buf.add_(other=d_p, alpha=-group['lr'])

        return loss
