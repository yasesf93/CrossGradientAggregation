'''
arXiv PDF: https://arxiv.org/pdf/1910.12308v1.pdf
Github: https://github.com/ICLR-PopSGD/PopSGD/blob/master/worker.py
'''

import torch
from torch.optim import Optimizer
import numpy as np


class SwarmSGD(Optimizer):
	def __init__(self, params, kwargs, lr=0.01, momentum=0, dampening=0,
				 weight_decay=0, nesterov=False):
		if lr < 0.0:
			raise ValueError("Invalid learning rate: {}".format(lr))
		if momentum < 0.0:
			raise ValueError("Invalid momentum value: {}".format(momentum))
		if weight_decay < 0.0:
			raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

		defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
						weight_decay=weight_decay, nesterov=nesterov)
		if nesterov and (momentum <= 0 or dampening != 0):
			raise ValueError("Nesterov momentum requires a momentum and zero dampening")
		super(SwarmSGD, self).__init__(params, defaults)


		# Extract required var.
		self.kwargs = kwargs
		self.pi = kwargs.get('pi')
		self.device = kwargs.get('device')

		self.local_neigh = np.argwhere(np.asarray(self.pi) != 0.0).ravel() # Find connected agents
		self.pi = [self.pi[i] for i in self.local_neigh] # simplified pi -- with only connected agents




	def __setstate__(self, state):
		super(SwarmSGD, self).__setstate__(state)
		for group in self.param_groups:
			group.setdefault('nesterov', False)

	@torch.no_grad()
	def step(self, closure=None):
		"""Performs a single optimization step.

		Args:
			closure (callable, optional): A closure that reevaluates the model
				and returns the loss.
		"""

		# # Extract var. added thru Collab()
		dist = self.kwargs['dist']
		distgroup = self.kwargs['distgroup']
		neighbors = self.kwargs['neighbors']
		selected_workers = self.kwargs['selected_workers']

		loss = None
		if closure is not None:
			with torch.enable_grad():
				loss = closure()

		for group in self.param_groups:
			weight_decay = group['weight_decay']
			momentum = group['momentum']
			dampening = group['dampening']
			nesterov = group['nesterov']

			for p in group['params']:
				# Only perform update on selected workers
				if dist.get_rank() in selected_workers:
					if p.grad is None:
						continue
					d_p = p.grad
					if weight_decay != 0:
						d_p = d_p.add(p, alpha=weight_decay)
					if momentum != 0:
						param_state = self.state[p]
						if 'momentum_buffer' not in param_state:
							buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
						else:
							buf = param_state['momentum_buffer']
							buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
						if nesterov:
							d_p = d_p.add(buf, alpha=momentum)
						else:
							d_p = buf

					p.add_(d_p, alpha=-group['lr'])

				else:
					pass

				# Gather all the parameters from all agents
				con_buf = [torch.zeros(p.data.size()).to(self.device) for _ in range(dist.get_world_size())] # Parameters placeholder
				dist.all_gather(con_buf, p.data, group=distgroup) # Gather parameters from workers to con_buf

				# Perform averaging with the other selected agent 
				if dist.get_rank() in selected_workers:
					# print("Rank before",dist.get_rank(),":",torch.norm(p.data))
					p.data = (con_buf[selected_workers[0]]+con_buf[selected_workers[1]])/2
					# print("Rank after",dist.get_rank(),":",torch.norm(p.data))
				# A

		return loss


'''
Pseudo:
Local update for selected worker
all_gather()
for selected worker,
average and update params
'''