import random, math, sys
import torch
from torch.utils.data import DataLoader#, Sampler, DistributedSampler
import numpy as np

class Partition(object):
	
	def __init__(self, data, index):
		self.data = data
		self.index = index
		
	def __len__(self):
		return len(self.index)
	
	def __getitem__(self, index):
		data_idx = self.index[index]
		return self.data[data_idx]
	
	
class DataPartition(object):
	
	def __init__(self, dataset, **kwargs):
		self.data = dataset
		self.num_workers = kwargs.get('num_workers')
		self.dataset_name = kwargs.get('data')
		self.data_dist = kwargs.get('data_dist', 'iid')
		self.dist = kwargs['dist']

		# split inputs and labels
		self.labels = self.data.targets
		if self.dataset_name == 'MNIST':
			self.labels = self.labels.numpy()
			self.labels = np.asarray(self.labels)
		elif self.dataset_name == 'CIFAR10' or self.dataset_name == 'CIFAR100' or self.dataset_name == 'stl10':
			self.labels = np.asarray(self.labels)

		#identify number of classes
		classes, class_counts = np.unique(self.labels, return_counts=True)
		self.num_classes = len(classes)
		min_cls_cnt = min(class_counts) # used in non-iid cases to ensure equal data distribution

		
		#sort data by class
		self.class_list = [[] for nb in range(self.num_classes)]
		for class_ in range(self.num_classes):
			self.class_list[class_] = np.argwhere(self.labels == class_).ravel()

		#initialize worker index
		self.worker_index = [[] for worker in range(self.num_workers)]

		#Partition into desired distribution
		#iid
		if self.data_dist == 'iid':
			#distribute data amongst workers (card dealer style)
			sample_per_class = [len(class_)//self.num_workers for class_ in self.class_list] # To account for imbalanced classes like MNIST
			start_id = [0 for _ in range(self.num_classes)] # start ID for each class -- Same reason as above

			#iterate through each worker
			for rank_ in range(self.num_workers):
				# iterate through each class
				for class_ in range(self.num_classes):
					temp_index = self.class_list[class_][start_id[class_]:start_id[class_]+sample_per_class[class_]] # Extract data from class
					self.worker_index[rank_].extend(temp_index) # Assign data to worker
					start_id[class_] += sample_per_class[class_] # Updates start ID for each class

		#non-iid
		elif self.data_dist == 'non-iid':
			self.class_list = [class_[:min_cls_cnt] for class_ in self.class_list] # Trim samples to min. 
			
			# Classes > workers
			if self.num_classes > self.num_workers:
				
				# Determine acceptable number of worker
				possible_num_workers = []
				for i in range(3,self.num_classes):
					if self.num_classes%i==0:
						possible_num_workers.append(i)
				assert self.num_classes % self.num_workers == 0, f"For {self.dataset_name} classes > workers, pls choose num. workers from: {possible_num_workers}" # To ensure each worker have same num of classes
				
				for i in range(self.num_classes):
					temp_index = self.class_list[i] # Assigning whole class, but ensure each class have same num. samples
					self.worker_index[i%self.num_workers].extend(temp_index)
			
			# Classes = workers
			elif self.num_classes == self.num_workers:
				for i in range(self.num_classes):
					temp_index = self.class_list[i]
					self.worker_index[i%self.num_workers].extend(temp_index)
			
			# Classes < workers
			else:

				# Check if each worker can have same amount of data
				assert self.num_workers%self.num_classes == 0, f"For {self.dataset_name} workers > classes, pls choose num. workers from multiples of {self.num_classes}" # To ensure each worker have same num of classes

				start_id = [0 for _ in range(self.num_classes)] # start ID for each class
				class_worker_ratio = self.num_classes/self.num_workers # % of data per class for one worker
				assert class_worker_ratio<=0.5, "Warning: class_worker_ratio > 0.5 -- will cause heavy imbalance in number of data across different workers."

				# Compute sample per class to be assigned to single worker
				sample_per_class = int(np.floor(len(self.class_list[0])*class_worker_ratio)) # Classes in self.class_list should have all similar len

				for i in range(self.num_workers):
					class_ = i%self.num_classes # Determine assigned class for this worker
					temp_index = self.class_list[class_][start_id[class_]:start_id[class_]+sample_per_class] # Extract data from class
					self.worker_index[i].extend(temp_index) # Assign data to worker
					start_id[class_] += sample_per_class # Updates start ID for each class

	
	def get_(self, rank_id):
		"""
		This function takes in a rank id and returns its share of minibatch
		"""
		return Partition(self.data, self.worker_index[rank_id])

						  
		 

def get_partition_dataloader(dataset, **kwargs):
	"""
	Partition the whole dataset into smaller sets for each rank.
	
	@param dataset: complete the dataset. We will 1) split it evenly to every worker and 2) build dataloaders on splitted dataset
	@batch_size: batch_size for the data loader! i.e. every worker will load this many samples 
	@param num_workers: the number of workers. Used to decide partition ratios
	@param myrank: the rank of this particular process
	@**kwargs partition: the partition ratio for dividing the data
	rvalue: training set for this particular rank
	"""
	data_dist = kwargs.get('data_dist', 'iid')
	batch_size = kwargs.get('batch_size', 128)
	num_workers = kwargs.get('num_workers')
	dataset_name = kwargs.get('data')
	myrank = kwargs.get('rank')
	if num_workers is None:
		raise ValueError('provide number of workers')

	partitioner = DataPartition(dataset, **kwargs)  # partitioner is in charge of producing shuffled id lists
	if kwargs.get('server_rank', -1) != -1:
		curr_rank_dataset = partitioner.get_(myrank-1)  # get the data partitioned for current rank, 0 is the server so -1
	else:
		curr_rank_dataset = partitioner.get_(myrank)  # get the data partitioned for current rank, 0 is the server so -1
	# build a dataloader based on the partitioned dataset for current rank
	train_set = DataLoader(curr_rank_dataset, batch_size=batch_size, shuffle=True)
	return train_set                          
 