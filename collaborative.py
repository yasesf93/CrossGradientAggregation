import torch
from trainer import Trainer
import numpy as np
from copy import deepcopy
from random import randint, choice

class Collab(Trainer):
	"""docstring for Collab"""
	def __init__(self, dataloader, testdataloader, model, opt, criterion, LR_scheduler, **kwargs):
		super(Collab, self).__init__(dataloader, testdataloader, model, opt, criterion, LR_scheduler, **kwargs)
		dist = kwargs['dist']
		self.server_rank = kwargs.get('server_rank', -1)
		assert self.server_rank == -1
		self.num_workers = kwargs.get('num_workers', -1)
		assert self.wsize == self.num_workers, 'num_workers is same as wsize'
		assert self.num_workers > 0, 'num_workers must be greater than zero'
		self.workers = list(range(self.num_workers))
		self.neighbors = kwargs.get('neighbors')
		assert len(self.neighbors) <= self.wsize
		self.distgroup = dist.new_group(ranks=self.neighbors) # should only consist of connected neigh.

		self.verbose = kwargs.get("verbose")

		# Add to optimizer kwargs (args not present in argdict)
		self.optimizer.kwargs['dist'] = self.dist
		self.optimizer.kwargs['distgroup'] = self.distgroup
		self.optimizer.kwargs['neighbors'] = self.neighbors

		self.hand_shake()

	def hand_shake(self):
		''' original '''
		numdata = torch.tensor(len(self.dataloader))
		self.numdata = [torch.tensor(0) for _ in range(self.wsize)]
		self.dist.all_gather(self.numdata, numdata)

		'''
		numdata = torch.tensor(self.kwargs.get('rank'))
		print("Gathering...")
		# Testing all_gather() with intercepting ranks in "group" with 4 workers: works if group does not have intercepting ranks
		if self.kwargs.get('rank') == 0:
			
			self.numdata = [torch.tensor(10) for _ in range(2)]
			self.dist.all_gather(self.numdata, numdata,group=self.dist.new_group(ranks=[0,1]))
			
			# self.numdata = [torch.tensor(10) for _ in range(3)]
			# self.dist.all_gather(self.numdata, numdata,group=self.dist.new_group(ranks=[0,1,2]))

		elif self.kwargs.get('rank') == 1:

			self.numdata = [torch.tensor(10) for _ in range(2)]
			self.dist.all_gather(self.numdata, numdata,group=self.dist.new_group(ranks=[0,1]))

			# self.numdata = [torch.tensor(10) for _ in range(3)]
			# self.dist.all_gather(self.numdata, numdata,group=self.dist.new_group(ranks=[0,1,2]))

		elif self.kwargs.get('rank') == 2:

			self.numdata = [torch.tensor(10) for _ in range(2)]
			self.dist.all_gather(self.numdata, numdata,group=self.dist.new_group(ranks=[2,3]))

			# self.numdata = [torch.tensor(10) for _ in range(3)]
			# self.dist.all_gather(self.numdata, numdata,group=self.dist.new_group(ranks=[0,1,2]))

		elif self.kwargs.get('rank') == 3:

			self.numdata = [torch.tensor(10) for _ in range(2)]
			self.dist.all_gather(self.numdata, numdata,group=self.dist.new_group(ranks=[2,3]))

			# self.numdata = [torch.tensor(10) for _ in range(1)]
			# self.dist.all_gather(self.numdata, numdata,group=self.dist.new_group(ranks=[3]))
		'''


		'''
		# Testing gather(): Can only specify gather for 1 rank
		# self.numdata = [torch.tensor(10) for _ in range(len(self.neighbors))]
		self.numdata = [torch.tensor(10) for _ in range(3)]

		if self.kwargs.get('rank') == 0:
			self.dist.gather(numdata, self.numdata, dst=0)
			# self.dist.gather(numdata, self.numdata, dst=0, group=self.distgroup)
			# self.dist.gather(numdata, self.numdata, dst=0, group=self.distgroup, async_op=True)
		else:
			self.dist.gather(numdata)

		# self.dist.gather(numdata, self.numdata, dst=self.kwargs.get('rank'), group=self.distgroup, async_op=True)
		# self.dist.gather(numdata, self.numdata, dst=self.kwargs.get('rank'), group=self.distgroup)
		'''

		'''
		# Testing scatter(): Can only specify scatter for 1 rank
		self.numdata = [torch.tensor(10) for _ in range(len(self.neighbors))]
		# self.dist.scatter(numdata,self.numdata,group=self.distgroup)#,async_op=True)
		'''

		# print(self.kwargs.get('rank'),':',self.numdata)
		# A


	def train_epoch(self, epoch_id):
		# worker code
		epoch_loss = 0.0
		for batch_idx, (data, target) in enumerate(self.dataloader):
			if self.verbose >= 1:
				if self.rank ==0:
					print(f"\rBatch: {batch_idx+1}/{len(self.dataloader)}",end="")

			if self.kwargs.get('optimizer') in ['CGA','CompCGA']:
				if batch_idx == len(self.dataloader)-1: # For the last batch,
					data, target = self.batchsize_check(data, target)
				self.perform_cross_test(data, target)


			elif self.kwargs.get('optimizer') == 'SwarmSGD':
				self.find_worker_pair()

			# initialize optimizer
			self.optimizer.zero_grad()
			# compute the gradient first or else gradient will be None.
			output = self.model(data.to(self.device))
			loss = self.criterion(output, target.to(self.device))
			loss.backward()
			epoch_loss += loss.item()
			# update 
			self.optimizer.step()
		if self.verbose >= 1:
			if self.rank ==0:
				print("\n")
		if self.scheduler is not None:
			self.scheduler.step()

		self.worker_train_loss_hist.append(epoch_loss/len(self.dataloader))

		# temp. tensor to be collected by worker 0 in compute_globals()
		self.global_train_loss = torch.tensor(epoch_loss/len(self.dataloader)) 



	def test_epoch(self, epoch_id):
		data_loaders = [self.dataloader, self.testdataloader]
		mode = ["train", "test"]
		temp_loss = []
		temp_acc = []

		for i, data_loader in enumerate(data_loaders):
			self.model.eval()
			loss = 0
			correct = 0

			with torch.no_grad():
				for inputs, target in data_loader:
					inputs, target = inputs.to(self.device), target.to(self.device)
					output = self.model(inputs)
					loss += self.criterion(output, target).item() # sum up batch loss
					pred = output.data.max(1)[1] # get the index of the max log-probability
					correct += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

			loss /= len(data_loader)
			acc = 100. * correct / len(data_loader.dataset) # check if data_loader.dataset works
			if self.verbose >= 2:
				print(f'Agent {self.dist.get_rank()} {mode[i]} set: Average loss: {loss:.4f}, Accuracy: {correct}/{len(data_loader.dataset)} ({acc:.0f}%)')

			temp_loss.append(loss)
			temp_acc.append(acc)

		self.worker_trainloader_loss_hist.append(temp_loss[0])
		self.worker_testloader_loss_hist.append(temp_loss[1])

		self.worker_trainloader_acc_hist.append(temp_acc[0])
		self.worker_testloader_acc_hist.append(temp_acc[1])

		# temp. tensor to be collected by worker 0 in compute_globals()
		self.global_trainloader_loss = torch.tensor(temp_loss[0])
		self.global_testloader_loss = torch.tensor(temp_loss[1])
		self.global_trainloader_acc = torch.tensor(temp_acc[0])
		self.global_testloader_acc = torch.tensor(temp_acc[1])



	def find_worker_pair(self):
		# self.optimizer.kwargs['selected_workers'] = [torch.zeros(1).to(self.device) for _ in range(2)]
		self.agent_1 = torch.tensor(self.kwargs.get('wsize')+1).to(self.device) # selected agent 1 placeholder
		self.agent_2 = torch.tensor(self.kwargs.get('wsize')+1).to(self.device) # selected agent 2 placeholder

		if self.rank==0:
			agent_1 = torch.tensor(randint(0,self.kwargs.get('wsize')-1)).to(self.device) # First selected agent
			agent_1 = [agent_1 for _ in range(self.kwargs.get('wsize'))] # List of agent_1 to be scattered to all agents
			# print("agent_1:",agent_1)
			self.dist.scatter(self.agent_1, agent_1)
		else:
			self.dist.scatter(self.agent_1)

		# if self.rank==0:
		# 	print("Rank:",self.rank,self.agent_1)

		if self.rank==self.agent_1:
			# print("Rank:",self.rank,"It's me!")
			neigh_1 = list(deepcopy(self.optimizer.local_neigh))
			# print("====== 1 =======> ",neigh_1)
			neigh_1.remove(self.agent_1) # same as neigh_1.remove(optimizers[agent_1].agent_id)
			# print("====== 2 =======> ",neigh_1)
			agent_2 = torch.tensor(choice(neigh_1))
			agent_2 = [agent_2 for _ in range(self.kwargs.get('wsize'))] # List of agent_2 to be scattered to all agents
			# print("agent_2:",agent_2)

			self.dist.scatter(self.agent_2, agent_2, src=self.agent_1)
		else:
			self.dist.scatter(self.agent_2, src=self.agent_1)

		# Add this optimizer kwargs
		self.optimizer.kwargs['selected_workers'] = [self.agent_1, self.agent_2]
		# print("Rank",self.rank,":",self.optimizer.kwargs['selected_workers'])
		
		if self.verbose >= 2:
			if self.rank==0:
				print("\nSelected workers:",self.optimizer.kwargs['selected_workers'])






	def compute_globals(self):
		# Computes global loss, accuracy and training time

		data = [
			self.global_train_loss, 
			self.global_trainloader_loss, 
			self.global_testloader_loss, 
			self.global_trainloader_acc, 
			self.global_testloader_acc,
			torch.tensor(self.exec_time)
			]

		# Global data placeholder
		if self.rank == 0:
			temp_g_data = []

		for TENSOR in data:
			# reduce -- sum all agent loss to worker 0
			torch.distributed.reduce(TENSOR, dst=0)

			if self.rank == 0:
				# div by num agents(wsize) to get global loss (only on worker 0)
				TENSOR = torch.div(TENSOR, self.wsize)
				
				# Collect computed global data
				temp_g_data.append(TENSOR.item())

		# Store global data
		if self.rank == 0:
			self.global_train_loss_hist.append(temp_g_data[0])
			self.global_trainloader_loss_hist.append(temp_g_data[1])
			self.global_testloader_loss_hist.append(temp_g_data[2])

			self.global_trainloader_acc_hist.append(temp_g_data[3])
			self.global_testloader_acc_hist.append(temp_g_data[4])

			self.global_avg_train_time_hist.append(temp_g_data[5])



	def batchsize_check(self, data, target):

		# Ensure all workers have same num. of data and target for all_gather()
		target_len = torch.tensor(len(target)) # Determine worker's current batch len
		all_target_len = [torch.tensor(0, dtype=target_len.dtype).to(self.device) for _ in range(len(self.neighbors))] # Placeholder

		self.dist.all_gather(all_target_len, target_len, group=self.distgroup) # Gather ALL worker's current batch len
		min_len = min(all_target_len) # Find the smallest one

		# Trim batch to smallest len
		data = data[:min_len.item()]
		target = target[:min_len.item()]

		return data, target



	def perform_cross_test(self, data, target):
		'''
		Pseudo:
		- Gathers all data and target from all agents
		- Filter and extract connected agents data and target (excluding self, but included a empty placeholder too)
		- Perform cross test
		'''

		# Initiate placeholder to gather all agents current batch of data
		all_workers_cur_data   = [torch.zeros(data.size(), dtype=data[0].dtype).to(self.device) for _ in range(len(self.neighbors))]
		all_workers_cur_target = [torch.zeros(target.size(), dtype=target[0].dtype).to(self.device) for _ in range(len(self.neighbors))]

		# Gathers data and target from workers
		self.dist.all_gather(all_workers_cur_data, data, group=self.distgroup)
		self.dist.all_gather(all_workers_cur_target, target, group=self.distgroup)

		# Extract connected agents data and target only
		local_neigh = np.argwhere(np.asarray(self.kwargs.get('pi')) != 0.0).ravel() # Find connected agents

		# Initialize cross gradient placeholder -- including a slot for own grad. too
		self.optimizer.kwargs['cross_grad'] = [[] for _ in range(len(local_neigh))]
		# self.optimizer.kwargs['cross_loss'] = [[] for _ in range(len(local_neigh))] # not used for now

		# Remove worker own index/rank in local neighborhood -- our own data will be used in normal training later
		idx = np.delete(local_neigh, np.where(local_neigh == self.rank)) 

		all_workers_cur_data = [all_workers_cur_data[i] for i in idx]
		all_workers_cur_target = [all_workers_cur_target[i] for i in idx]

		# Cross test with data from other connected agents (excluding own data)
		for i, (data_, target_) in enumerate(zip(all_workers_cur_data, all_workers_cur_target)):
			cross_grad, cross_loss = self.cross_test(data_, target_)

			# Store grad at respective neigh. rank index
			self.optimizer.kwargs['cross_grad'][np.where(np.asarray(local_neigh) == idx[i])[0][0]] = cross_grad
			# self.self.optimizer.kwargs['cross_grad'][np.where(np.asarray(local_neigh) == idx[i])[0][0]] = cross_grad



	def cross_test(self, data, target):
		'''
		Performs test/eval with self.optimizer(own parameter) on input data and target
		'''
		self.model.eval()

		self.optimizer.zero_grad()
		output = self.model(data)
		loss = self.criterion(output, target)
		loss.backward()
		batch_loss = loss.item() # minibatch loss

		mb_gradient = []
		for i in range(len(self.optimizer.param_groups[0]["params"])): # For each model layer,
			mb_gradient.append(deepcopy(self.optimizer.param_groups[0]["params"][i].grad.data))
		
		return mb_gradient, batch_loss