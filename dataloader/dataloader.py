#Author:Bhishma Dedhia 

import numpy as np 
import torch
import random


def low_rank(data, sing_to_keep):

	data_flat = data.reshape(len(data),-1) 
	u, s, v = np.linalg.svd(data_flat.astype(float))

	#k = int(np.rint(len(s)*pct_to_keep/100))
	u_approx = u[:, :sing_to_keep]
	s_approx = np.diag(s[:sing_to_keep])
	v_approx = v[:sing_to_keep, :]

	approx_flat = np.dot(u_approx, np.dot(s_approx, v_approx))
	approx = approx_flat.reshape(data.shape[0], data.shape[1], data.shape[2])
	print('Low rank conversion done')
	return approx




class PreTrainDataLoader:

	def __init__(self, seed, dir_path, device, config, target_id, lowrank_approx = False, sing_to_keep =3):

		#exlude_ids is 
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
		self.discrete_dim = config.discrete_dim
		self.target_id = target_id
		self.data_init = np.float32(np.load(dir_path+'data.npy',allow_pickle=True))
		self.mask = np.load(dir_path+'mask.npy',allow_pickle=True).astype(bool)
		self.data_init[self.mask] = 0
		target_data = self.data_init[target_id] 
		red_data = np.delete(self.data_init,self.target_id,0)
		if lowrank_approx:	
			red_data[:,:,:self.cont_dim] = low_rank(red_data[:,:,:self.cont_dim],sing_to_keep)
			#fraction adjust estimator
			data_min = np.amin(red_data.reshape(-1,self.feature_dim),0)[:self.cont_dim]
			data_max = np.amax(red_data.reshape(-1,self.feature_dim),0)[:self.cont_dim]
			self.data = np.insert(red_data,target_id,target_data,0)	
			self.data[:,:,:self.cont_dim] = (self.data[:,:,:self.cont_dim] - data_min)/(data_max - data_min)

		else:
			data_min = np.amin(red_data.reshape(-1,self.feature_dim),0)
			data_max = np.amax(red_data.reshape(-1,self.feature_dim),0)
			self.data = self.data_init
			self.data[:,:,:self.cont_dim] = (self.data[:,:,:self.cont_dim] - data_min)/(data_max - data_min)	
		self.seqs = config.seq_range
		self.seq_pool = [i for i in range(self.data.shape[0]) if i!=self.target_id]
		self.time_range = config.time_range
		self.time_ids = np.arange(self.time_range)




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
			interv_time = np.random.randint(self.pre_int_len, self.time_range-self.post_int_len)
			pre_int_seq = self.data[seq_ids,interv_time - self.pre_int_len:interv_time]
			post_int_seq = self.data[seq_ids,interv_time:interv_time+self.post_int_len]
			timestamp_preint = np.repeat(self.time_ids[interv_time - self.pre_int_len:interv_time].reshape(1,-1),self.K,axis=0)

			timestamp_postint = np.repeat(self.time_ids[interv_time:\
								interv_time+self.post_int_len].reshape(1,-1),self.K,axis=0)
			#print(timestamp_postint.shape)
			attention_mask_preint = np.ones(timestamp_preint.shape)
			attention_mask_postint = np.ones(timestamp_postint.shape)

			if pre_int_seq.shape[1]<self.pre_int_len:
				seqlen = pre_int_seq.shape[1]
				pre_int_seq = np.concatenate((np.zeros((self.K,self.pre_int_len-seqlen\
							,self.feature_dim)),pre_int_seq),axis=1)

				timestamp_preint = np.concatenate((np.zeros((self.K,self.pre_int_len-seqlen)),timestamp_preint),axis=1)
				attention_mask_preint = np.concatenate((np.zeros((self.K,self.pre_int_len-seqlen)),attention_mask_preint),axis=1)

			if post_int_seq.shape[1]<self.post_int_len:
				seqlen = post_int_seq.shape[1]
				post_int_seq = np.concatenate((np.zeros((self.K,self.post_int_len-seqlen\
							,self.feature_dim)),post_int_seq),axis=1)

				timestamp_postint = np.concatenate((np.zeros((self.K,self.post_int_len-seqlen)),timestamp_postint),axis=1)
				attention_mask_postint = np.concatenate((np.zeros((self.K,self.post_int_len-seqlen)),attention_mask_postint),axis=1)

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
			seqids_postint[i] = torch.from_numpy(seqid_post_int)
			target_masks_preint[i] = target_mask_preint
			target_masks_postint[i] = target_mask_postint
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
		seqids_preint = seqids_preint.to(dtype=torch.long, device=self.device)
		seqids_postint = seqids_postint.to(dtype=torch.long, device = self.device)
		target_masks_preint = target_masks_preint.to(dtype=torch.long, device = self.device)
		target_masks_postint = target_masks_postint.to(dtype=torch.long, device=self.device)
		attention_masks_preint = attention_masks_preint.to(dtype=torch.long, device=self.device)
		attention_masks_postint = attention_masks_postint.to(dtype=torch.long, device=self.device)
		targets_cont = targets_cont.to(dtype=torch.float32,device=self.device)

		if targets_discrete is not None:
			targets_discrete = targets_discrete.to(dtype=torch.long, device=self.device)

		return pre_int_seqs, post_int_seqs, timestamps_preint,\
				timestamps_postint, seqids_preint, seqids_postint,\
				target_masks_preint, target_masks_postint, attention_masks_preint,\
				attention_masks_postint, targets_cont, targets_discrete


class FinetuneDataLoader(object):

	def __init__(self, seed, dir_path, device, config, target_id, interv_time, lowrank_approx = False, sing_to_keep = 3):

		torch.manual_seed(seed)
		np.random.seed(seed)
		random.seed(seed)
		self.device = device
		self.config = config
		self.K = config.K
		self.interv_time = interv_time
		self.feature_dim = config.feature_dim
		self.pre_int_len = config.pre_int_len
		self.post_int_len = config.post_int_len
		self.cont_dim = config.cont_dim
		self.discrete_dim = config.discrete_dim
		self.data_init = np.float32(np.load(dir_path+'data.npy',allow_pickle=True))
		self.mask = np.load(dir_path+'mask.npy',allow_pickle=True).astype(bool)
		self.data_init[self.mask] = 0
		self.target_data = self.data_init[target_id]
		red_data = np.delete(self.data_init,target_id,0)
		if lowrank_approx:
			red_data[:,:,:self.cont_dim] = low_rank(red_data[:,:,:self.cont_dim],sing_to_keep)
			data_min = np.amin(red_data.reshape(-1,self.feature_dim),0)[:self.cont_dim]
			data_max = np.amax(red_data.reshape(-1,self.feature_dim),0)[:self.cont_dim]
			self.data = np.concatenate((red_data,self.target_data.reshape(1,-1,self.feature_dim)),0)
			self.data[:,:,:self.cont_dim] = (self.data[:,:,:self.cont_dim] - data_min)/(data_max - data_min)

		else:
			data_min = np.amin(red_data.reshape(-1,self.feature_dim),0)
			data_max = np.amax(red_data.reshape(-1,self.feature_dim),0)
			self.data = np.concatenate((red_data,self.target_data.reshape(1,-1,self.feature_dim)),0)
			self.data[:,:,:self.cont_dim] = (self.data[:,:,:self.cont_dim] - data_min)/(data_max - data_min)
		self.data[self.K-1,:,1:self.cont_dim] = 0
		self.seqs = config.seq_range
		self.target_id = target_id
		self.seq_pool = [i for i in range(self.seqs) if i!=self.target_id]
		self.time_range = config.time_range
		self.time_ids = np.arange(self.time_range)

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
		seq_ids = np.asarray(self.seq_pool+[self.target_id])
		
		if self.discrete_dim>0:
			targets_discrete = torch.zeros(batch_size,self.post_int_len,self.discrete_dim) 
		else:
			targets_discrete = None

		for i in range(batch_size):

			interv_time = np.random.randint(self.pre_int_len,self.interv_time-1)
			pre_int_seq = self.data[:,interv_time - self.pre_int_len:interv_time]
			post_int_lim = min(self.interv_time,interv_time+self.post_int_len)
			post_int_seq = self.data[:,interv_time:post_int_lim]
			timestamp_preint = np.repeat(self.time_ids[interv_time - self.pre_int_len:interv_time].reshape(1,-1),self.K,axis=0)

			timestamp_postint = np.repeat(self.time_ids[interv_time:\
								post_int_lim].reshape(1,-1),self.K,axis=0)
			
			attention_mask_preint = np.ones(timestamp_preint.shape)
			attention_mask_postint = np.ones(timestamp_postint.shape)

			if pre_int_seq.shape[1]<self.pre_int_len:
				seqlen = pre_int_seq.shape[1]
				pre_int_seq = np.concatenate((np.zeros((self.K,self.pre_int_len-seqlen\
							,self.feature_dim)),pre_int_seq),axis=1)

				timestamp_preint = np.concatenate((np.zeros((self.K,self.pre_int_len-seqlen)),timestamp_preint),axis=1)
				attention_mask_preint = np.concatenate((np.zeros((self.K,self.pre_int_len-seqlen)),attention_mask_preint),axis=1)

			if post_int_seq.shape[1]<self.post_int_len:
				seqlen = post_int_seq.shape[1]
				post_int_seq = np.concatenate((np.zeros((self.K,self.post_int_len-seqlen\
							,self.feature_dim)),post_int_seq),axis=1)

				timestamp_postint = np.concatenate((np.zeros((self.K,self.post_int_len-seqlen)),timestamp_postint),axis=1)
				attention_mask_postint = np.concatenate((np.zeros((self.K,self.post_int_len-seqlen)),attention_mask_postint),axis=1)


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
			seqids_postint[i] = torch.from_numpy(seqid_post_int)
			target_masks_preint[i] = target_mask_preint
			target_masks_postint[i] = target_mask_postint
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
		seqids_preint = seqids_preint.to(dtype=torch.long, device=self.device)
		seqids_postint = seqids_postint.to(dtype=torch.long, device = self.device)
		target_masks_preint = target_masks_preint.to(dtype=torch.long, device = self.device)
		target_masks_postint = target_masks_postint.to(dtype=torch.long, device=self.device)
		attention_masks_preint = attention_masks_preint.to(dtype=torch.long, device=self.device)
		attention_masks_postint = attention_masks_postint.to(dtype=torch.long, device=self.device)
		targets_cont = targets_cont.to(dtype=torch.float32,device=self.device)

		if targets_discrete is not None:
			targets_discrete = targets_discrete.to(type=torch.long, device=self.device)

		return pre_int_seqs, post_int_seqs, timestamps_preint,\
				timestamps_postint, seqids_preint, seqids_postint,\
				target_masks_preint, target_masks_postint, attention_masks_preint,\
				attention_masks_postint, targets_cont, targets_discrete








