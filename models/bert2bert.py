#Author: Bhishma Dedhia

import torch
import sys
sys.path.append('../transformers/src/')
from transformers import BertConfig, EncoderDecoderModel
from transformers.models.bert.modeling_bert_syn_ctrl import BertModelSynCtrl
import numpy as np
import torch.nn as nn
import copy


class Bert2BertSynCtrl(nn.Module):
	'''
	This model is an encoder decoder model that enoodes in pre-intervention sequences and generates the control from post-intervention data of donor states
	'''

	def __init__(self, config, random_seed):

		'''
		Initialize a syn ctrl encoder decoder model.
		'''

		super().__init__()

		self.config = config
		self.K = config.K 
		self.hidden_dim = config.hidden_size
		self.feature_dim = config.feature_dim
		self.pre_int_len = config.pre_int_len
		self.post_int_len = config.post_int_len
		self.cont_dim = config.cont_dim#no. of cont features
		self.discrete_dim = config.discrete_dim#no. of discrete features
		self.classes = config.classes #List of no. of classes for each discrete feature
		self.encoder_config =  copy.deepcopy(config) #BertConfig(K=config.K,max_position_embeddings=K*pre_int_len,hidden_size=self.hidden_dim)
		self.encoder_config.max_position_embeddings = 0#K*pre_int_len*embeddings
		self.decoder_config = copy.deepcopy(config) #BertConfig(K=config.K,max_position_embeddings=K*post_int_len,is_decoder=True,add_cross_attention=True,hidden_size=self.hidden_dim)
		self.decoder_config.max_position_embeddings = 0#K*post_int_len*embeddings
		self.decoder_config.is_decoder = True
		self.decoder_config.add_cross_attention = True
		self.encoder_model = BertModelSynCtrl(self.encoder_config)
		self.decoder_model = BertModelSynCtrl(self.decoder_config)
		self.Bert2BertSynCtrl = EncoderDecoderModel(encoder=self.encoder_model,decoder=self.decoder_model)
		self.embed_features = nn.Linear(self.feature_dim,self.hidden_dim)
		self.embed_timestep = nn.Embedding(config.time_range,self.hidden_dim)
		self.embed_seq = nn.Embedding(config.seq_range,self.hidden_dim)
		self.embed_target = nn.Embedding(2,self.hidden_dim)
		self.embed_ln = nn.LayerNorm(self.hidden_dim)
		self.predict_cont_target = nn.Linear(self.hidden_dim,1) 
		if self.discrete_dim>0:
			self.predict_discrete_target = [nn.Linear(self.hidden_dim, num_class) for num_class in self.classes]
		else:
			self.predict_discrete_target = None

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
				attention_mask_post_int
				):

		'''
		Foward pass through the model 

		pre_int_seq = pre_intervention batch Shape: BS*K*pre_int_len*feature_dim
		post_int_seq = post_intervention batch Shape: BS*K*post_int_len*feature_dim
		timesteps_pre_int = timestamp IDs of preint Shape: BS*K*pre_int_len
		timesteps_posts_int = timestamp IDs of postint Shape: BS*K*post_int_len
		seq_id_pre_int = seq_ids of pre_int Shape: BS*K*pre_int_len
		seq_ids_post_int = seq_ids of pre_int Shape: BS*K*post_int_len
		target_pre_int = target pre-int seq or not Shape: BS*K*pre_int_len
		target_post_int = target post-int seq or not Shape: BS*K*post_int_len
		attention_mask_pre_int = encoder attention mask Shape: BS*K*pre_int_len
		attention_mask_post_int = decoder attention mask Shape: BS*K*post_int_len
		'''
		batch_size, pre_int_seq_len = pre_int_seq.shape[0], pre_int_seq.shape[2]
		post_int_seq_len = post_int_seq.shape[2]

		pre_int_seq = pre_int_seq.permute(0,2,1,3).reshape(batch_size,self.K*pre_int_seq_len,-1)
		post_int_seq = post_int_seq.permute(0,2,1,3).reshape(batch_size,self.K*post_int_seq_len,-1)
		pre_int_embedding = self.embed_features(pre_int_seq)
		post_int_embedding = self.embed_features(post_int_seq)

		pre_int_time_embedding = self.embed_timestep(timestamps_pre_int.permute(0,2,1).reshape(batch_size,self.K*pre_int_seq_len))
		post_int_time_embedding = self.embed_timestep(timestamps_post_int.permute(0,2,1).reshape(batch_size,self.K*post_int_seq_len))

		pre_seq_embedding = self.embed_seq(seq_ids_pre_int.permute(0,2,1).reshape(batch_size,self.K*pre_int_seq_len))
		post_seq_embedding = self.embed_seq(seq_ids_post_int.permute(0,2,1).reshape(batch_size,self.K*post_int_seq_len))
		pre_target_embedding = self.embed_target(target_pre_int.permute(0,2,1).reshape(batch_size,self.K*pre_int_seq_len))
		post_target_embedding = self.embed_target(target_post_int.permute(0,2,1).reshape(batch_size,self.K*post_int_seq_len))
		
		pre_int_embedding = pre_int_embedding + pre_int_time_embedding + pre_seq_embedding + pre_target_embedding
		post_int_embedding = post_int_embedding + post_int_time_embedding + post_seq_embedding + post_target_embedding

		attention_mask_pre_int = attention_mask_pre_int.permute(0,2,1).reshape(batch_size,self.K*pre_int_seq_len)
		attention_mask_post_int = attention_mask_post_int.permute(0,2,1).reshape(batch_size,self.K*post_int_seq_len)

		outputs = self.Bert2BertSynCtrl(inputs_embeds=pre_int_embedding,
										decoder_inputs_embeds=post_int_embedding,
										attention_mask=attention_mask_pre_int,
										decoder_attention_mask=attention_mask_post_int,
										)

		last_hidden_state = outputs.decoder_hidden_states[-1]
		decoder_outputs = last_hidden_state.reshape(batch_size,self.post_int_len,self.K,self.hidden_dim).permute(0,2,1,3)
		cont_target_preds = self.predict_cont_target(decoder_outputs)[:,self.K-2] #predict target seq using the hidden state from last donor (any donor hidden unit can be used)
		if self.predict_discrete_target is not None:
			discrete_target_preds = [self.predict_discrete_target[i](decoder_outputs)[:,self.K-2] 
												for i in range(self.discrete_dim)]
		else:
			discrete_target_preds = None
		
		return cont_target_preds, discrete_target_preds

	def forward_attention(self, 
				pre_int_seq, 
				post_int_seq,
				timestamps_pre_int, 
				timestamps_post_int, 
				seq_ids_pre_int, 
				seq_ids_post_int, 
				target_pre_int, 
				target_post_int,
				attention_mask_pre_int, 
				attention_mask_post_int
				):

		'''
		Foward pass through the model 

		pre_int_seq = pre_intervention batch Shape: BS*K*pre_int_len*feature_dim
		post_int_seq = post_intervention batch Shape: BS*K*post_int_len*feature_dim
		timesteps_pre_int = timestamp IDs of preint Shape: BS*K*pre_int_len
		timesteps_posts_int = timestamp IDs of postint Shape: BS*K*post_int_len
		seq_id_pre_int = seq_ids of pre_int Shape: BS*K*pre_int_len
		seq_ids_post_int = seq_ids of pre_int Shape: BS*K*post_int_len
		target_pre_int = target pre-int seq or not Shape: BS*K*pre_int_len
		target_post_int = target post-int seq or not Shape: BS*K*post_int_len
		attention_mask_pre_int = encoder attention mask Shape: BS*K*pre_int_len
		attention_mask_post_int = decoder attention mask Shape: BS*K*post_int_len
		'''
		batch_size, pre_int_seq_len = pre_int_seq.shape[0], pre_int_seq.shape[2]
		post_int_seq_len = post_int_seq.shape[2]

		pre_int_seq = pre_int_seq.permute(0,2,1,3).reshape(batch_size,self.K*pre_int_seq_len,-1)
		post_int_seq = post_int_seq.permute(0,2,1,3).reshape(batch_size,self.K*post_int_seq_len,-1)
		pre_int_embedding = self.embed_features(pre_int_seq)
		post_int_embedding = self.embed_features(post_int_seq)

		pre_int_time_embedding = self.embed_timestep(timestamps_pre_int.permute(0,2,1).reshape(batch_size,self.K*pre_int_seq_len))
		post_int_time_embedding = self.embed_timestep(timestamps_post_int.permute(0,2,1).reshape(batch_size,self.K*post_int_seq_len))

		pre_seq_embedding = self.embed_seq(seq_ids_pre_int.permute(0,2,1).reshape(batch_size,self.K*pre_int_seq_len))
		post_seq_embedding = self.embed_seq(seq_ids_post_int.permute(0,2,1).reshape(batch_size,self.K*post_int_seq_len))
		pre_target_embedding = self.embed_target(target_pre_int.permute(0,2,1).reshape(batch_size,self.K*pre_int_seq_len))
		post_target_embedding = self.embed_target(target_post_int.permute(0,2,1).reshape(batch_size,self.K*post_int_seq_len))
		
		pre_int_embedding = pre_int_embedding + pre_int_time_embedding + pre_seq_embedding + pre_target_embedding
		post_int_embedding = post_int_embedding + post_int_time_embedding + post_seq_embedding + post_target_embedding

		attention_mask_pre_int = attention_mask_pre_int.permute(0,2,1).reshape(batch_size,self.K*pre_int_seq_len)
		attention_mask_post_int = attention_mask_post_int.permute(0,2,1).reshape(batch_size,self.K*post_int_seq_len)

		outputs = self.Bert2BertSynCtrl(inputs_embeds=pre_int_embedding,
										decoder_inputs_embeds=post_int_embedding,
										attention_mask=attention_mask_pre_int,
										decoder_attention_mask=attention_mask_post_int,
										output_attentions = True,
										)

		last_hidden_state = outputs.decoder_hidden_states[-1]

		decoder_outputs = last_hidden_state.reshape(batch_size,self.post_int_len,self.K,self.hidden_dim).permute(0,2,1,3)
		cont_target_preds = self.predict_cont_target(decoder_outputs)[:,self.K-2] 
		if self.predict_discrete_target is not None:
			discrete_target_preds = [self.predict_discret_target[i](decoder_outputs)[:,self.K-2] 
												for i in range(self.discrete_dim)]
		else:
			discrete_target_preds = None

		last_attention = outputs.decoder_attentions[-1]

		
		return cont_target_preds, discrete_target_preds, last_attention


	def generate_post_int(self, 
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
				pred_index):

		'''
		Generate next time step's post-int target prediction
		'''

		with torch.no_grad():
			outputs_cont, outputs_discrete = self.forward(pre_int_seq, 
								post_int_seq,
								timestamps_pre_int, 
								timestamps_post_int, 
								seq_ids_pre_int, 
								seq_ids_post_int, 
								target_pre_int, 
								target_post_int,
								attention_mask_pre_int, 
								attention_mask_post_int)

		if outputs_discrete is not None:
			outputs_discrete = torch.stack([torch.argmax(pred[0,pred_index]).reshape(1,-1) for pred in outputs_discrete],dim=1).reshape(-1)
			
		return outputs_cont[0, pred_index], outputs_discrete 

	def return_attention(self, 
				pre_int_seq, 
				post_int_seq,
				timestamps_pre_int, 
				timestamps_post_int, 
				seq_ids_pre_int, 
				seq_ids_post_int, 
				target_pre_int, 
				target_post_int,
				attention_mask_pre_int, 
				attention_mask_post_int):

		'''
		Generate next time step's post-int target prediction
		'''

		with torch.no_grad():
			outputs_cont, outputs_discrete, attention = self.forward_attention(pre_int_seq, 
								post_int_seq,
								timestamps_pre_int, 
								timestamps_post_int, 
								seq_ids_pre_int, 
								seq_ids_post_int, 
								target_pre_int, 
								target_post_int,
								attention_mask_pre_int, 
								attention_mask_post_int)


		if outputs_discrete is not None:
			outputs_discrete = torch.stack([torch.argmax(pred[0,0]).reshape(1,-1) for pred in outputs_discrete],dim=1).reshape(-1)
			
		
		return outputs_cont[0, 0], outputs_discrete, attention[0,0,self.K-2,:self.K-1]






















