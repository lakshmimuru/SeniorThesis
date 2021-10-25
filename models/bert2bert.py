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

	def __init__(self, K, pre_int_len, post_int_len, feature_dim, hidden_dim, time_range, seq_range):

		'''
		Initialize a syn ctrl encoder decoder model.
		K = Total control units
		pre_int_len = Total pre-int slice used
		post_int_len = Total post-int slice predicted
		hidden_dim = hidden_dim used in txf
		time_range = maximum time stamp processed by the model
		seq_range = Total sequences in dataset

		'''

		super().__init__()


		self.K = K
		self.hidden_dim = hidden_dim
		self.feature_dim = feature_dim
		self.pre_int_len = pre_int_len
		self.post_int_len = post_int_len
		self.encoder_config = BertConfig(K=K,max_position_embeddings=K*pre_int_len,hidden_size=hidden_dim)
		self.decoder_config = BertConfig(K=K,max_position_embeddings=K*post_int_len,is_decoder=True,add_cross_attention=True,hidden_size=hidden_dim)
		self.encoder_model = BertModelSynCtrl(self.encoder_config)
		self.decoder_model = BertModelSynCtrl(self.decoder_config)
		self.Bert2BertSynCtrl = EncoderDecoderModel(encoder=self.encoder_model,decoder=self.decoder_model)
		self.embed_features = nn.Linear(self.feature_dim,self.hidden_dim)
		self.embed_timestep = nn.Embedding(time_range,self.hidden_dim)
		self.embed_seq = nn.Embedding(seq_range,self.hidden_dim)
		self.embed_target = nn.Embedding(2,self.hidden_dim)
		self.embed_ln = nn.LayerNorm(self.hidden_dim)
		self.predict_target = nn.Linear(self.hidden_dim,1) #Todo: more than 1 prediction head incase of mutliple targets


	def forward(self, 
				pre_int_seq, 
				post_int_seq,
				timestamps_pre_int, 
				timestamps_post_int, 
				seq_ids_pre_int, 
				seq_ids_post_int, 
				target_pre_int, 
				target_post_int,
				attention_mask_pre_int, 
				attention_mask_post_int,
				):

	'''
	pre_int_seq = pre_intervention batch Shape: BS*max_len_encoder*feature_dim
	post_int_seq = post_intervention batch Shape: BS*max_len_decoder*feature_dim
	timesteps_pre_int = timestamp IDs of preint Shape: BS*max_len_encoder
	timesteps_posts_int = timestamp IDs of postint Shape: BS*max_len_decoder
	seq_id_pre_int = seq_ids of pre_int Shape: BS*max_len_encoder
	seq_ids_post_int = seq_ids of pre_int Shape: BS*max_len_ decoder
	target_pre_int = target pre-int seq or not Shape: BS*max_len_encoder
	target_post_int = target post-int seq or not Shape: BS*max_len_decoder
	attention_mask_pre_int = encoder attention mask Shape: BS*max_len_encoder
	attention_mask_post_int = decoder attention mask Shape: BS*max_len_decoder
	'''

	pre_int_embedding = self.embed_features(pre_int_seq)
	post_int_embedding = self.embed_features(post_int_seq)
	pre_int_time_embedding = self.embed_timestep(timestamps_pre_int)
	post_int_time_embedding = self.embed_timestep(timestamps_post_int)
	pre_seq_embedding = self.embed_seq(seq_ids_pre_int)
	post_seq_embedding = self.embed_seq(seq_ids_post_int)
	pre_target_embedding = self.embed_target(target_pre_int)
	post_target_embedding = self.embed_target(target_post_int)
	pre_int_embedding = pre_int_embedding + pre_int_time_embedding + pre_seq_embedding + pre_target_embedding
	post_int_embedding = post_int_embedding + post_int_time_embedding + post_seq_embedding + post_target_embedding

	outputs = self.Bert2BertSynCtrl(inputs_embeds=pre_int_embedding,
									decoder_inputs_embeds=post_int_embedding,
									attention_mask=attention_mask_pre_int,
									decoder_attention_mask=attention_mask_post_int,
									)

	decoder_outputs = output.decoder_last_hidden_state






