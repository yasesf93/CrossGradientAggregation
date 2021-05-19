###### Gradient Episodic Memory ######
import torch
from torch.optim import Optimizer
from copy import deepcopy
from copy import copy
from collections import defaultdict
import numpy as np
import quadprog

class SGA(Optimizer):
	def __init__(self, params, lr=0.01, momentum=0.0, dampening=0,
				 weight_decay=0, nesterov=False, omega=0.5):
		if not 0.0 <= lr:
			raise ValueError("Invalid learning rate: {}".format(lr))
		if not 0.0 <= momentum:
			raise ValueError("Invalid momentum value: {}".format(momentum))
		if not 0.0 <= weight_decay:
			raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
		print('Optimizer initialized with omega value of %s and momentum of %s'%(omega, momentum))
		defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
						weight_decay=weight_decay, nesterov=nesterov, omega=omega)
		if nesterov and (momentum <= 0 or dampening != 0):
			raise ValueError("Nesterov momentum requires a momentum and zero dampening")
		super(SGA, self).__init__(params, defaults)

		self.eps = 1e-12
		self.margin = 0.5


	def __setstate__(self, state):
		super(SGA, self).__setstate__(state)
		for group in self.param_groups:
			group.setdefault('nesterov', False)

	def step(self, dist, distgroup, neighbors, pi, device, batch_idx, i_ag, closure=None):
		"""Performs a single optimization step.
		Arguments:
			closure (callable, optional): A closure that reevaluates the model
				and returns the loss.
		"""
		loss = None
		if closure is not None:
			loss = closure()
		if not isinstance(self.state, defaultdict):
			self.state = defaultdict(dict)


		# check data_gradient passing:
		'''
		try:
			print("Agent %s optimizer data gradients: "%self.agent_id)
			print(np.shape(self.data_gradient))
			for grad in self.data_gradient:
				print(torch.norm(grad))
		except:
			print("First epoch")
		'''

		# self.old_v = []
		for i, group in enumerate(self.param_groups): # always 1
			weight_decay = group['weight_decay']
			momentum = group['momentum']
			dampening = group['dampening']
			nesterov = group['nesterov']
			for j, p in enumerate(group['params']):
				if p.grad is None:
					continue
				d_p = p.grad.data
				assert (torch.isfinite(p.data).any()), "p.data before update Norm2 is %s"%(p.data.norm(2))
				assert (torch.isfinite(d_p).any()), "d_p Norm2 is %s"%(d_p.norm(2))
				# n_ag = len(self.data_gradient)
				gradsize = d_p.size()
				# print('agent id', self.agent_id)
				# print('i', i_ag)
				#g_tilda[0] = d_p
				g_tilda = torch.zeros((neighbors, *gradsize)).cuda()
				g_tilda_mat = [torch.zeros(gradsize).to(device) for n in neighbors]
				dist.all_gather(g_tilda_mat, p.grad.data, group=distgroup)
				g_tilda[self.agent_id] = d_p
				# for pival, g_tilda_agent in zip(pi, g_tilda_mat):
				for ag, g_tilda_agent in zip (neighbors, g_tilda_mat):
					if ag != self.agent_id: # ignoring own param, own data from data_gradient, use p.grad.data
						g_tilda[ag] = g_tilda_agent 

				######################## gradient averaging combined with SGA implementation #################
				# if epoch%3 == 0: 
				# 	g_tilda,v = self.ProjSGA(g_tilda,gradsize, n_ag, j, batch_idx, old_v_one,i_ag)
				# else:
				# 	g_tilda = gradavg(g_tilda,n_ag)

				################################### SGA implementation #################################
				#g_tilda,v = self.ProjSGA(g_tilda,gradsize, n_ag, j, batch_idx, old_v_ep,i_ag)     #old_v_epoch
				# g_tilda,v = self.ProjSGA(g_tilda,gradsize, n_ag, j, batch_idx, old_v_one,i_ag, n_ag)	   #old_v_batch
				g_tilda = ProjSGA(pi, g_tilda,gradsize, neighbors, j, batch_idx, i_ag)	   #old_v_batch updated
				# self.old_v.append(v)

				################################### gradient averaging implementation #################################
				#g_tilda = gradavg(g_tilda, n_ag)


				p.data.add_(other=g_tilda, alpha=-group['lr'])
				assert (torch.isfinite(p.data).any()),"p.data Norm2 after update is %s"%(p.data.norm(2))

		return loss

	def set_agent_param_groups(self, agent_param_groups):
		self.agent_param_groups = agent_param_groups

	def set_agent_states(self, agent_states):
		self.agent_states = agent_states

	def set_data_grad_and_loss(self, data_gradient, data_loss): # self.data_gradient ==> list of tensors, data_loss = list of float
		self.data_gradient = data_gradient
		self.data_loss = data_loss


	
################################### SGA function #################################

def ProjSGA(pi, grad,gradsize, neighbors, j, batch_idx,i_ag):	# old_v_batch updated as general func for easier import across other SGA-variant

	except_triggered = False
	grad_flat = torch.zeros(*grad[0].flatten().size(), neighbors).cuda()
	for i in range (neigbors):
		grad_flat[:,i] = grad[i].flatten()
	################## new (graph topology) ###############################
	grad_flat = grad_flat.cpu().double().numpy()
	grad_flat = np.transpose(grad_flat)
	for d, pival in zip (neighbors, pi):
		# grad_flat[d,:] = self.pi[i_ag][d]*grad_flat[d,:]
		grad_flat[d,:] = pival*grad_flat[d,:]
	grad_np = grad_flat[i_ag,:]
	grad_flat = np.delete(grad_flat, (i_ag), axis =0) #memory rows
	memory_np = grad_flat[~np.all(grad_flat==0,axis=1)] #non zerow rows

	t = memory_np.shape[0]
	p = np.dot(memory_np, memory_np.transpose())
	p = 0.5*(p+p.transpose()) + eps*np.eye(t)
	q = -1*np.dot(memory_np, grad_np)
	G = np.eye(t)
	h = np.zeros(t) + margin
	try:
		v = quadprog.solve_qp(p, q, G,h)[0]
	except ValueError:
		except_triggered = True
		print('handling ValueError')
		#v = np.ones(t)*0.5							#arbitrary v
		#v = old_v_ep[batch_idx][i_ag][j][:]		#old_v_epoch
		v = old_v_one[i_ag][j][:]					#old_v_batch
		print('v',v)
	x = np.dot(v,memory_np)+grad_np
	grad = torch.Tensor(x).view(*gradsize).cuda()
	if except_triggered:
		print('grad:', grad.norm(2))
	return grad


def gradavg(grad, n):
	grad = grad.sum(0)
	grad /= n
	return grad