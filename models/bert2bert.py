#Author: Bhishma Dedhia

import torch
import sys
sys.path.append('../transformers/src/')
from transformers import BertConfig, BertModelSynCtrl, EncoderDecoderModel
import numpy as np
import torch.nn as nn


class Bert2BertSynCtrl(nn.Module):
	'''
	This model is an encoder decoder model that enoodes in pre-intervention sequences and generates the control from post-intervention data of donor states
	'''

	def __init__(self, K, pre_int_len, post_int_len, hidden_dim, time_range, seq_range):

		super().__init__()


		self.K = K
		self.hidden_dim = hidden_dim
		self.pre_int_len = pre_int_len
		self.post_int_len = post_int_len
		self.encoder_config = BertConfig(K=K,max_position_embeddings=K*pre_int_len,hidden_size=hidden_dim)
		self.decoder_config = BertConfig(K=K,max_position_embeddings=K*post_int_len,is_decoder=True,add_cross_attention=True,hidden_size=hidden_dim)
		self.encoder_model = BertModelSynCtrl(self.encoder_config)
		self.decoder_model = BertModelSynCtrl(self.decoder_config)
		self.Bert2BertSynCtrl = EncoderDecoderModel(encoder=self.encoder_model,decoder=self.decoder_model)
		self.embed_timestep = nn.Embedding(time_range,self.hidden_dim)
		self.embed_seq = nn.Embedding(seq_range,self.hidden_dim)
		self.embed_target = nn.Embedding(2,self.hidden_dim)
		self.embed_ln = nn.LayerNorm(self.hidden_dim)
		self.predict_target = nn.LayerNorm(self.hidden_dim,1) #more than 1 prediction head here!


	def forward(self, pre_int, post_int, timesteps, seq_ids, target_mask, attention_mask, target):



