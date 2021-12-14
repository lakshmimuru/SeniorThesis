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


class Trainer:

    def __init__(self,
				model,
				output_file,
				dir_path,
				target_id,
				interv_time,
				lowrank_approx,
				):

		self.model = model
		self.config = model.config
		self.interv_time = interv_time
		self.K = self.config.K
		self.feature_dim = self.config.feature_dim
		self.pre_int_len = self.config.pre_int_len
		self.post_int_len = self.config.post_int_len
		self.cont_dim = self.config.cont_dim
		self.discrete_dim = self.config.discrete_dim
		#process data
		self.mask = np.load(dir_path+'mask.npy',allow_pickle=True)
		self.data_init[self.mask] = 0
		self.target_data = self.data_init[target_id] 
		red_data = np.delete(self.data_init,self.target_id,0)
		if lowrank_approx:	
			red_data[:,:,:self.cont_dim] = low_rank(red_data[:,:,:self.cont_dim],pct_to_keep)
			#fraction adjust estimator
			self.data_min = np.amin(red_data.reshape(-1,self.feature_min),0)[:self.cont_dim]
			self.data_max = np.amax(red_data.reshape(-1,self.feature_min),0)[:self.cont_dim]
			self.data[:,:,:self.cont_dim] = (red_data[:,:,:self.cont_dim] - data_min)/(data_max - data_min)

		else:
			data_min = np.amin(red_data.reshape(-1,self.feature_min),0)[:self.cont_dim]
			data_max = np.amax(red_data.reshape(-1,self.feature_min),0)[:self.cont_dim]
			self.data = self.data_init
			self.data[:,:,:self.cont_dim] = (red_data[:,:,:self.cont_dim] - data_min)/(data_max - data_min)	

		self.target_data[:,:cont_dim] = (self.target_data[:,:cont_dim]- data_min)/(data_max - data_min)
		self.seqs = config.seq_range
		self.time_range = config.time_range
		self.target_id = target_id
		self.seq_pool = [i for i in range(self.seqs) if i!=self.target_id]
		self.seq_ids = np.asarray(self.seq_pool+[self.target_id])
		self.time_ids = np.arange(self.time_range)

    def generate_post(self,interv_time):

		pre_int_seq_donor = self.data[:,interv_time - self.pre_int_len:interv_time]
		pre_int_seq_target = self.target_data[interv_time - self.pre_int_len:interv_time]
		pre_int_seq = np.concatenate((pre_int_seq_donor,pre_int_seq_target),0).reshape(1,-1,self.feature_dim)
		post_int_seq_donor = self.data[:,interv_time:interv_time+self.post_int_len]
		post_int_seq_target = np.zeros((post_int_seq_donor.shape[1],self.feature_dim))
		post_int_seq = np.concatenate((post_int_seq_donor,post_int_seq_target),0).reshape(1,-1,self.feature_dim)
		seqid_pre_int = np.repeat(np.asarray(self.seq_ids).reshape(-1,1),self.pre_int_len,axis=1)
		seqid_post_int = np.repeat(np.asarray(self.seq_ids).reshape(-1,1),self.post_int_len,axis=1)
		timestamp_preint = np.repeat(self.time_ids[interv_time - self.pre_int_len\
								:interv_time].reshape(1,-1),self.K,axis=0)
		timestamp_postint = np.repeat(self.time_ids[interv_time:\
								post_int_lim].reshape(1,-1),self.K,axis=0)
		
		attention_mask_preint = np.ones(timestamp_preint.shape)
		attention_mask_postint = np.ones(timestamp_postint.shape)



		if post_int_seq.shape[1]<self.post_int_len:
			post_int_seq = np.concatenate((np.zeros((self.K,self.post_int_len-post_int_seq.shape[1]\
						,self.feature_dim)),post_int_seq),axis=1)
			timestamp_postint = np.concatenate((np.zeros((self.K,self.post_int_len-post_int_seq.shape[1])),timestamp_postint),axis=1)
			attention_mask_postint = np.concatenate((np.zeros((self.K,self.post_int_len-post_int_seq.shape[1])),attention_mask_postint),axis=1)


		target_mask_preint = torch.zeros(self.K,self.pre_int_len)
		target_mask_preint[self.K-1]  = 1
		target_mask_postint = torch.zeros(self.K,self.post_int_len)
		target_mask_postint[self.K-1] = 1

		
		for i in range(self.post_int_len):



	def sliding_window_geneate()







  

