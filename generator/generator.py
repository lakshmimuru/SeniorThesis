#Author:Bhishma Dedhia

import torch
import torch.nn as nn
import os
import sys
import numpy as np
import argparse
import yaml
sys.path.append('../models/')
sys.path.append('../dataloader/')
sys.path.append('../transformers/src/')
from dataloader import low_rank
from bert2bert import Bert2BertSynCtrl
from transformers import BertConfig
from trainer import Trainer 
from dataloader import FinetuneDataLoader


class Generator:

	def __init__(self,
				model,
				device,
				dir_path,
				target_id,
				interv_time,
				lowrank_approx
				):

		self.model = model
		self.config = model.config
		self.device = device
		self.interv_time = interv_time
		self.K = self.config.K
		self.feature_dim = self.config.feature_dim
		self.pre_int_len = self.config.pre_int_len
		self.post_int_len = self.config.post_int_len
		self.cont_dim = self.config.cont_dim
		self.discrete_dim = self.config.discrete_dim
		self.target_id = target_id
		self.data_init = np.float32(np.load(dir_path+'data.npy',allow_pickle=True))
		self.mask = np.load(dir_path+'mask.npy',allow_pickle=True).astype(bool)
		self.data_init[self.mask] = 0
		self.target_data = self.data_init[target_id] 
		red_data = np.delete(self.data_init,target_id,0)
		if lowrank_approx:	
			red_data[:,:,:self.cont_dim] = low_rank(red_data[:,:,:self.cont_dim],pct_to_keep)
			self.data_min = np.amin(red_data.reshape(-1,self.feature_dim),0)[:self.cont_dim]
			self.data_max = np.amax(red_data.reshape(-1,self.feature_dim),0)[:self.cont_dim]
			self.data = red_data
			self.data[:,:,:self.cont_dim] = (red_data[:,:,:self.cont_dim] - data_min)/(data_max - data_min)

		else:
			self.data_min = np.amin(red_data.reshape(-1,self.feature_dim),0)[:self.cont_dim]
			self.data_max = np.amax(red_data.reshape(-1,self.feature_dim),0)[:self.cont_dim]
			self.data = red_data
			self.data[:,:,:self.cont_dim] = (red_data[:,:,:self.cont_dim] - self.data_min)/(self.data_max - self.data_min)	

		self.target_data[:,:self.cont_dim] = (self.target_data[:,:self.cont_dim]- self.data_min)/(self.data_max - self.data_min)
		self.target_data[:,1:self.cont_dim] = 0
		self.seqs = self.config.seq_range
		self.time_range = self.config.time_range		
		self.seq_pool = [i for i in range(self.seqs) if i!=self.target_id]
		self.seq_ids = np.asarray(self.seq_pool+[self.target_id])
		self.time_ids = np.arange(self.time_range)

	def generate_post_int(self,interv_time):

		pre_int_seq_donor = self.data[:,max(interv_time - self.pre_int_len,0):interv_time]
		pre_int_seq_target = np.expand_dims(self.target_data[max(interv_time - self.pre_int_len,0):interv_time],0)
		pre_int_seq = np.concatenate((pre_int_seq_donor,pre_int_seq_target),0)
		post_int_seq_donor = self.data[:,interv_time:interv_time+self.post_int_len]
		post_int_seq_target = np.zeros((1,post_int_seq_donor.shape[1],self.feature_dim))
		post_int_seq = np.concatenate((post_int_seq_donor,post_int_seq_target),0)
		seqid_pre_int = np.repeat(np.asarray(self.seq_ids).reshape(-1,1),self.pre_int_len,axis=1)
		seqid_post_int = np.repeat(np.asarray(self.seq_ids).reshape(-1,1),self.post_int_len,axis=1)
		timestamp_preint = np.repeat(self.time_ids[max(interv_time - self.pre_int_len,0)\
								:interv_time].reshape(1,-1),self.K,axis=0)
		post_int_lim = min(self.time_range,interv_time+self.post_int_len)
		timestamp_postint = np.repeat(self.time_ids[interv_time:\
								post_int_lim].reshape(1,-1),self.K,axis=0)
		
		attention_mask_preint = np.ones(timestamp_preint.shape)
		attention_mask_postint = np.ones(timestamp_postint.shape)

		if pre_int_seq.shape[1]<self.pre_int_len:
			seqlen = pre_int_seq.shape[1]
			pre_int_seq = np.concatenate([np.zeros((self.K,self.pre_int_len-seqlen\
						,self.feature_dim)),pre_int_seq],axis=1)
			timestamp_preint = np.concatenate([np.zeros((self.K,self.pre_int_len-seqlen)),timestamp_preint],axis=1)
			attention_mask_preint = np.concatenate([np.zeros((self.K,self.pre_int_len-seqlen)),attention_mask_preint],axis=1)

		if post_int_seq.shape[1]<self.post_int_len:
			seqlen = post_int_seq.shape[1]
			post_int_seq = np.concatenate([np.zeros((self.K,self.post_int_len-seqlen\
						,self.feature_dim)),post_int_seq],axis=1)
			timestamp_postint = np.concatenate([np.zeros((self.K,self.post_int_len-seqlen)),timestamp_postint],axis=1)
			attention_mask_postint = np.concatenate([np.zeros((self.K,self.post_int_len-seqlen)),attention_mask_postint],axis=1)

		target_mask_preint = torch.zeros(self.K,self.pre_int_len)
		target_mask_preint[self.K-1]  = 1
		target_mask_postint = torch.zeros(self.K,self.post_int_len)
		target_mask_postint[self.K-1] = 1

		pre_int_seq = torch.unsqueeze(torch.from_numpy(pre_int_seq).to(dtype = torch.float32,device=self.device),0)
		post_int_seq = torch.unsqueeze(torch.from_numpy(post_int_seq).to(dtype = torch.float32,device=self.device),0)
		seqid_pre_int = torch.unsqueeze((torch.from_numpy(seqid_pre_int)).to(dtype = torch.long,device=self.device),0)
		seqid_post_int = torch.unsqueeze(torch.from_numpy(seqid_post_int).to(dtype = torch.long,device=self.device),0)
		timestamp_preint = torch.unsqueeze(torch.from_numpy(timestamp_preint).to(dtype = torch.long,device=self.device),0)
		timestamp_postint = torch.unsqueeze(torch.from_numpy(timestamp_postint).to(dtype = torch.long,device=self.device),0)
		target_mask_preint = torch.unsqueeze(target_mask_preint.to(dtype=torch.long,device=self.device),0)
		target_mask_postint = torch.unsqueeze(target_mask_postint.to(dtype=torch.long,device=self.device),0)
		attention_mask_preint = torch.unsqueeze(torch.from_numpy(attention_mask_preint).to(dtype = torch.long,device=self.device),0)
		attention_mask_postint = torch.unsqueeze(torch.from_numpy(attention_mask_postint).to(dtype = torch.long,device=self.device),0)
		
		for i in range(min(self.post_int_len,self.time_range-interv_time)):



			cont_target, disc_target = self.model.generate_post_int(pre_int_seq, post_int_seq,\
																	timestamp_preint, timestamp_postint,\
																	seqid_pre_int,seqid_post_int,\
																	target_mask_preint,target_mask_postint,\
																	attention_mask_preint,attention_mask_postint,\
																	i)
			if disc_target is not None:
				disc_target = disc_target.to(dtype=float32)
				post_int_seq[0,self.K-1,i,0] = cont_target
				post_int_seq[0,self.K-1,i,self.cont_dim:] = disc_target

			else:
				post_int_seq[0,self.K-1,i,0] = cont_target

		self.target_data[interv_time:post_int_lim,0] = post_int_seq[0,self.K-1,:post_int_lim - interv_time,0].cpu().numpy()
		self.target_data[interv_time:post_int_lim,self.cont_dim:] = post_int_seq[0,self.K-1,:post_int_lim - interv_time,self.cont_dim:].cpu().numpy()


	def sliding_window_generate(self):

		self.model.eval()

		for i in range(self.interv_time,self.time_range,self.post_int_len):
			self.generate_post_int(i)

		self.target_data[:,:self.cont_dim] = self.target_data[:,:self.cont_dim]*(self.data_max - self.data_min)+self.data_min

		return self.target_data[:,0]







  

