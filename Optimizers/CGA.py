###### Gradient Episodic Memory ######
import torch
from torch.optim import Optimizer
from copy import deepcopy
from copy import copy
from collections import defaultdict
import numpy as np
import quadprog
# from Optimizers.SGA import ProjSGA, gradavg

'''
"Simplified" pi and cross_grad contains only info. for local neighborhood (including self)
'''

class CGA(Optimizer):
	def __init__(self, params, kwargs, lr=0.01, momentum=0.95, dampening=0, nesterov=False):
		if not 0.0 <= lr:
			raise ValueError("Invalid learning rate: {}".format(lr))
		if not 0.0 <= momentum:
			raise ValueError("Invalid momentum value: {}".format(momentum))

		print('CGA initialized with momentum of %s'%(momentum))

		defaults = dict(lr=lr, momentum=momentum, dampening=dampening, nesterov=nesterov)
		super(CGA, self).__init__(params, defaults)

		# For ProjSGA()
		self.eps = 1e-12
		self.margin = 0.5

		self.kwargs = kwargs
		self.pi = self.kwargs.get('pi')
		self.device = self.kwargs.get('device')

		self.local_neigh = np.argwhere(np.asarray(self.pi) != 0.0).ravel() # Find connected agents
		self.pi = [self.pi[i] for i in self.local_neigh] # simplified pi -- with only connected agents

		self.old_v = [[] for _ in range(len(self.param_groups[0]['params']))]



	def __setstate__(self, state):
		super(CGA, self).__setstate__(state)
		for group in self.param_groups:
			group.setdefault('nesterov', False)


	def step(self, closure=None):
		"""Performs a single optimization step.
		Arguments:
			closure (callable, optional): A closure that reevaluates the model
				and returns the loss.
		"""

		dist = self.kwargs['dist']
		distgroup = self.kwargs['distgroup']
		neighbors = self.kwargs['neighbors']

		cross_grad = self.kwargs['cross_grad']


		# Find this worker's index in the "simplified" pi and cross_grad 
		cross_grad_len = [len(i) for i in cross_grad]
		worker_index = np.where(np.asarray(cross_grad_len) == 0)[0][0]

		loss = None
		if closure is not None:
			loss = closure()
		if not isinstance(self.state, defaultdict):
			self.state = defaultdict(dict)

		for i, group in enumerate(self.param_groups): # always 1
			momentum = group['momentum']
			dampening = group['dampening']

			for j, p in enumerate(group['params']):
				if p.grad is None:
					continue
				d_p = p.grad.data
				assert (torch.isfinite(p.data).any()), "p.data before update Norm2 is %s"%(p.data.norm(2))
				assert (torch.isfinite(d_p).any()), "d_p Norm2 is %s"%(d_p.norm(2))

				gradsize = d_p.size()
				g_tilda = torch.zeros((len(self.local_neigh), *gradsize)).to(self.device)

				empty_counter = 0
				for i in range(len(g_tilda)):
					if len(cross_grad[i]) == 0: # empty for own agent's grad. -- grab it from d_p directly
						g_tilda[i] = d_p
						empty_counter += 1
					else:
						g_tilda[i] = cross_grad[i][j] # i-th worker's cross grad. for j-th layer

				assert empty_counter==1, "No empty list in cross_grad! cross_test is probably buggy!"


				################################### SGA implementation #################################
				g_tilda = self.ProjSGA(g_tilda, gradsize, j, worker_index)     #old_v_batch updated


				############################ CDMSGD (but with nesterov=False) ##############################
				con_buf = [torch.zeros(p.data.size(), dtype=p.data.dtype).to(self.device) for _ in range(len(neighbors))] # Parameters placeholder

				dist.all_gather(con_buf, p.data, group=distgroup) # Gather parameters from workers to con_buf
				buf = torch.zeros(p.data.size()).to(self.device)

				# Extract connected agents data only
				con_buf = [con_buf[i] for i in self.local_neigh]

				for pival, con_buf_agent in zip(self.pi, con_buf):
					buf.add_(other=con_buf_agent, alpha=pival)

				param_state = self.state[p]
				if 'momentum_buffer' not in param_state:
					m_buf = param_state['momentum_buffer'] = torch.zeros(p.data.size()).to(self.device)
					m_buf.mul_(momentum).add_(g_tilda)
				else:
					m_buf = param_state['momentum_buffer']
					m_buf.mul_(momentum).add_(other=g_tilda, alpha=1-dampening)

				g_tilda.add_(other=m_buf, alpha=momentum)
				p.data = buf.add_(other=g_tilda, alpha=-group['lr'])

				assert (torch.isfinite(p.data).any()),"p.data Norm2 after update is %s"%(p.data.norm(2))

		return loss


	def ProjSGA(self, grad, gradsize, model_layer, worker_index):
		except_triggered = False
		neigh_size = len(self.local_neigh)
		grad_flat = torch.zeros(*grad[0].flatten().size(), neigh_size).to(self.device)
		for i in range(neigh_size):
			grad_flat[:,i] = grad[i].flatten()
		################## new (graph topology) ###############################
		grad_flat = grad_flat.cpu().double().numpy()
		grad_flat = np.transpose(grad_flat)
		for i in range(neigh_size):
			grad_flat[i,:] = self.pi[i]*grad_flat[i,:]
		grad_np = grad_flat[worker_index,:]
		grad_flat = np.delete(grad_flat, (worker_index), axis=0) #memory rows
		memory_np = grad_flat[~np.all(grad_flat==0,axis=1)] #non zerow rows

		t = memory_np.shape[0]
		p = np.dot(memory_np, memory_np.transpose())
		p = 0.5*(p+p.transpose()) + self.eps*np.eye(t)
		q = -1*np.dot(memory_np, grad_np)
		G = np.eye(t)
		h = np.zeros(t) + self.margin
		try:
			v = quadprog.solve_qp(p, q, G,h)[0]
			self.old_v[model_layer] = v
		except ValueError:
			except_triggered = True
			print('Handling ValueError')
			#v = np.ones(t)*0.5							#arbitrary v
			# v = self.old_v[worker_index][model_layer][:]					#old_v_batch
			v = self.old_v[model_layer]					#old_v_batch
			print('v',v)
		x = np.dot(v,memory_np)+grad_np
		grad = torch.Tensor(x).view(*gradsize).to(self.device)
		if except_triggered:
			print('grad:', grad.norm(2))
		return grad


