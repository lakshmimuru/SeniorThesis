#Author:Bhishma Dedhia

import numpy as np 
import torch
import random


def low_rank(data, pct_to_keep, cont_dim):

	#todo: roshini meeting




class PreTrainDataLoader:

	def __init__(self, seed, dir_path, device, config, lowrank_approx = False, pct_to_keep = 50):

		
		torch.manual_seed(seed)
		np.random.seed(seed)
		random.seed(seed)
		self.device = device
		self.config = config
		self.K = config.K
		self.feature_dim = config.feature_dim
		self.pre_int_len = config.pre_int_len
		self.post_int_len = config.post_int_len
		self.cont_dim = config.cont_dim
		self.data_init = np.load(dir_path+'data.npy')
		self.mask = np.load(dir_path+'mask.npy')
		self.data_init[mask] = 0
		if lowrank_approx:
			self.data = low_rank(self.data_init,pct_to_keep, self.cont_dim)
		else:
			self.data = self.data_init
		self.seqs = config.seq_range
		self.exclude_ids = exclude_ids
		self.seq_pool = np.asarray([i for i in range(self.seqs) if i not in self.exclude_ids])
		self.time_range = self.data.shape[1]
		self.time_ids = np.asarray(self.data.shape[1])



	def get_batch(self, batch_size):

		
		pre_int_seqs = torch.zeros(batch_size,self.K,self.pre_int_len,self.feature_dim)
		post_int_seqs = torch.zeros(batch_size,self.K,self.post_int_len,self.feature_dim)		
		timestamps_preint = torch.zeros(batch_size,self.K,self.pre_int_len)
		timestamps_postint =  torch.zeros(batch_size,self.K,self.post_int_len)
		seqids_preint = torch.zeros(batch_size,self.K,self.pre_int_len)
		seqids_postint = torch.zeros(batch_size,self.K,self.post_int_len)
		target_masks_preint = torch.zeros(batch_size,self.K,self.pre_int_len)
		target_masks_postint = torch.zeros(batch_size,self.K,self.post_int_len)
		attention_masks_preint = torch.zeros(batch_size,self.K,self.pre_int_len)
		attention_masks_postint = torch.zeros(batch_size,self.K,self.post_int_len)
		targets_cont = torch.zeros(batch_size,self.post_int_len,1)
		
		if self.discrete_dim>0:
			targets_discrete = torch.zeros(batch_size,self.post_int_len,self.discrete_dim) 
		else:
			targets_discrete = None

		for i in range(batch_size):

			seq_ids = random.sample(self.seq_pool,self.K)
			interv_time = np.random.randint(self.time_range-self.post_int_len)
			pre_int_seq = self.data[seq_ids,interv_time - self.pre_int_len:interv_time]
			post_int_seq = self.data[seq_ids,interv_time:interv_time+self.post_int_len]
			timestamp_preint = np.repeat(self.time_ids[interv_time -\
								self.pre_int_len:interv_time].reshape(1,-1),self.K,axis=0)
			timestamp_postint = np.repeat(self.time_ids[interv_time:\
								interv_time+self.post_int_len].reshape(1,-1),self.K,axis=0)
			
			attention_mask_preint = np.ones(timestamp_preint.shape)
			attention_mask_postint = np.ones(timestamp_postint.shape)

			if pre_int_seq.shape[1]<self.pre_int_len:
				pre_int_seq = np.concatenate((np.zeros((self.K,self.pre_int_len-pre_int_seq.shape[1]\
							,self.feature_dim)),pre_int_seq),axis=1)

				timestamp_preint = np.concatenate((np.zeros((self.K,self.pre_int_len-pre_int_seq.shape[1])),timestamp_preint),axis=1)
				attention_mask_preint = np.concatenate((np.zeros((self.K,self.pre_int_len-pre_int_seq.shape[1])),attention_mask_preint),axis=1)

			if post_int_seq.shape[1]<self.post_int_len:
				post_int_seq = np.concatenate((np.zeros((self.K,self.post_int_len-post_int_seq.shape[1]\
							,self.feature_dim)),post_int_seq),axis=1)

				timestamp_postint = np.concatenate((np.zeros((self.K,self.post_int_len-post_int_seq.shape[1])),timestamp_postint),axis=1)
				attention_mask_postint = np.concatenate((np.zeros((self.K,self.post_int_len-post_int_seq.shape[1])),attention_mask_postint),axis=1)


			seqid_pre_int = np.repeat(np.asarray(seq_ids).reshape(-1,1),self.pre_int_len,axis=1)
			seqid_post_int = np.repeat(np.asarray(seq_ids).reshape(-1,1),self.post_int_len,axis=1)
			target_mask_preint = torch.zeros(self.K,self.pre_int_len)
			target_mask_preint[self.K-1]  = 1
			target_mask_postint = torch.zeros(self.K,self.post_int_len)
			target_mask_postint[self.K-1] = 1
			target_cont = post_int_seq[self.K-1,:,0].reshape(-1,1)

			pre_int_seqs[i] = torch.from_numpy(pre_int_seq)
			post_int_seqs[i] = torch.from_numpy(post_int_seq)
			timestamps_preint[i] = torch.from_numpy(timestamp_preint)
			timestamps_postint[i] = torch.from_numpy(timestamp_postint)
			seqids_preint[i] = torch.from_numpy(seqid_pre_int)
			seqids_postint[i] = torch.from_numpy(seqids_postint)
			target_masks_preint[i] = torch.from_numpy(target_mask_preint)
			target_mask_postint[i] = torch.from_numpy(target_mask_postint)
			attention_masks_preint[i] = torch.from_numpy(attention_mask_preint)
			attention_masks_postint[i] = torch.from_numpy(attention_mask_postint)
			targets_cont[i] = torch.from_numpy(target_cont)

			if targets_discrete is not None:
				target_discrete = post_int_seq[self.K-1,:,self.cont_dim:]
				targets_discrete[i] = torch.from_numpy(target_discrete)

		pre_int_seqs = pre_int_seqs.to(dtype=torch.float32,device=self.device)
		post_int_seqs = post_int_seqs.to(dtype=torch.float32, device=self.device)
		timestamps_preint = timestamps_preint.to(dtype=torch.long, device=self.device)
		timestamps_postint = timestamps_postint.to(dtype=torch.long, device=self.device)
		seqids_preint = seqids_preint.to(type=torch.long, device=self.device)
		seqids_postint = seqids_postint.to(type=torch.long, device = self.device)
		target_masks_preint = target_masks_preint.to(type=torch.long, device = self.device)
		target_masks_postint = target_masks_postint.to(type=torch.long, device=self.device)
		attention_masks_preint = attention_masks_preint.to(type=torch.long, device=self.device)
		attention_masks_postint = attention_masks_postint.to(type=torch.long, device=self.device)
		targets_cont = targets_cont.to(type=torch.float32,device=self.device)

		if targets_discrete is not None:
			targets_discrete = targets_discrete.to(type=torch.longdevice=self.device)

		return pre_int_seqs, post_int_seqs, timestamps_preint, 
				timestamps_postint, seqids_preint, seqids_postint,
				target_masks_preint, target_masks_postint, attention_masks_preint,
				attention_masks_postint, targets_cont, targets_discrete







