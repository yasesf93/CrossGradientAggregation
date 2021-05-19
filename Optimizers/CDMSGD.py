import torch
from torch.optim import Optimizer
from collections import defaultdict
import numpy as np

class CDMSGD(Optimizer):
	def __init__(self, params, kwargs, lr=0.01, momentum=0.95, dampening=0,
				 weight_decay=0):
		if not 0.0 <= lr:
			raise ValueError("Invalid learning rate: {}".format(lr))
		if not 0.0 <= momentum:
			raise ValueError("Invalid momentum value: {}".format(momentum))
		if not 0.0 <= weight_decay:
			raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

		defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
						weight_decay=weight_decay)
		if (momentum <= 0 or dampening != 0):
			raise ValueError("Nesterov momentum requires a momentum and zero dampening")
		super(CDMSGD, self).__init__(params, defaults)

		# Extract required var.
		self.kwargs = kwargs
		self.pi = kwargs.get('pi')
		self.device = kwargs.get('device')

		self.local_neigh = np.argwhere(np.asarray(self.pi) != 0.0).ravel() # Find connected agents
		self.pi = [self.pi[i] for i in self.local_neigh] # simplified pi -- with only connected agents


	def __setstate__(self, state):
		super(CDMSGD, self).__setstate__(state)
		for group in self.param_groups:
			group.setdefault('nesterov', True)

	# def step(self, closure=None):
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
			weight_decay = group['weight_decay']
			momentum = group['momentum']
			dampening = group['dampening']

			for j, p in enumerate(group['params']):
				if p.grad is None:
					continue
				d_p = p.grad.data

				con_buf = [torch.zeros(p.data.size()).to(self.device) for _ in range(len(neighbors))] # Parameters placeholder
				dist.all_gather(con_buf, p.data, group=distgroup) # Gather parameters from workers to con_buf
				buf = torch.zeros(p.data.size()).to(self.device)

				# Extract connected agents data only
				con_buf = [con_buf[i] for i in self.local_neigh]

				for pival, con_buf_agent in zip(self.pi, con_buf):
					buf.add_(other=con_buf_agent, alpha=pival)

				param_state = self.state[p]
				if 'momentum_buffer' not in param_state:
					m_buf = param_state['momentum_buffer'] = torch.zeros(p.data.size()).to(self.device)
					m_buf.mul_(momentum).add_(d_p)
				else:
					m_buf = param_state['momentum_buffer']
					m_buf.mul_(momentum).add_(other=d_p, alpha=1-dampening)

				d_p.add_(other=m_buf, alpha=momentum)
				p.data = buf.add_(other=d_p, alpha=-group['lr'])


		return loss