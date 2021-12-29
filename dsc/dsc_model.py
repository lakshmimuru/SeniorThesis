#Author 

import torch
import torch.nn as nn
import os
import sys
import numpy as np
import argparse
import yaml
sys.path.append('../models/')
sys.path.append('../training/')
sys.path.append('../dataloader/')
sys.path.append('../transformers/src/')
sys.path.append('../generator/')

import numpy as np
from bert2bert import Bert2BertSynCtrl
from transformers import BertConfig
from trainer import Trainer 
from dataloader import PreTrainDataLoader, FinetuneDataLoader
from generator import Generator

class DSCModel(object):

	def __init__(self,
				model,
				config,
				op_dir,
				target_id,
				random_seed,
				datapath,
				device,
				lowrank=False,
				classes=None):

		self.model = model
		self.config = config
		self.op_dir = op_dir
		self.target_id = target_id
		self.random_seed = random_seed
		self.datapath = datapath
		self.device = device
		self.lowrank = lowrank
		self.classes = classes

		self.config_model = BertConfig(hidden_size = config['hidden_size'],
							num_hidden_layers = config['n_layers'],
							num_attention_heads = config['n_heads'],
							intermediate_size = 4*config['hidden_size'],
							vocab_size = 0,
							max_position_embeddings = 0,
							output_hidden_states = True,
							)

		self.config_model.add_syn_ctrl_config(K=config['K'],
									pre_int_len=config['pre_int_len'],
									post_int_len=config['post_int_len'],
									feature_dim=config['feature_dim'],
									time_range=config['time_range'],
									seq_range=config['seq_range'],
									cont_dim=config['cont_dim'],
									discrete_dim=config['discrete_dim'],
									classes = classes)

		self.model = Bert2BertSynCtrl(self.config_model, random_seed)
		self.model = self.model.to(device)
		if not(os.path.exists(op_dir)):
			os.mkdir(op_dir)


	def fit(self, interv_time, checkpoint_pretrain = None):

		self.pretrain(checkpoint_pretrain)

		if self.model.Bert2BertSynCtrl.config.encoder.K == self.config['K']:

			print('Modifying K')
			self.model.config.K+=1
			self.model.K+=1
			self.model.Bert2BertSynCtrl.encoder.config.K+=1
			self.model.Bert2BertSynCtrl.decoder.config.K+=1

		self.finetune(interv_time)

	def predict(self, interv_time):

		generator = Generator(self.model,
						self.device,
						self.datapath,
						self.target_id,
						interv_time,
						self.lowrank)

		target_data =  generator.sliding_window_generate()

		return target_data


	def pretrain(self, checkpoint_pretrain, num_iters=1e1):


		dataloader_pretrain = PreTrainDataLoader(self.random_seed,
									self.datapath,
									self.device,
									self.config_model,
									self.target_id,
									lowrank_approx = self.lowrank)

		optimizer_pretrain = torch.optim.AdamW(self.model.parameters(),
									lr=eval(self.config['lr']),
									weight_decay=eval(self.config['weight_decay']),
									)

		warmup_steps = self.config['warmup_steps']
		scheduler = torch.optim.lr_scheduler.LambdaLR(
					optimizer_pretrain,
					lambda steps: min((steps+1)/warmup_steps,1))
		batch_size = self.config['batch_size']
		op_path_pretrain = self.op_dir + 'pretrain/'
		if not(os.path.exists(op_path_pretrain)):
			os.mkdir(op_path_pretrain)

		trainer_pretrain = Trainer(self.model,
						optimizer_pretrain,
						dataloader_pretrain,
						op_path_pretrain,
						batch_size,
						scheduler
						)

		print('Pretraining model on donor units')

		self.model = trainer_pretrain.train(int(num_iters),checkpoint_pretrain)


	def finetune(self, interv_time, num_iters=1e1):

		dataloader_finetune = FinetuneDataLoader(self.random_seed,
									self.datapath,
									self.device,
									self.config_model,
									self.target_id,
									interv_time,
									lowrank_approx = self.lowrank)

		optimizer_finetune = torch.optim.AdamW(self.model.parameters(),
								lr=eval(self.config['lr']),
								weight_decay=eval(self.config['weight_decay']),
									)
		batch_size = self.config['batch_size']
		op_path_finetune = self.op_dir + 'finetune/'
		if not(os.path.exists(op_path_finetune)):
			os.mkdir(op_path_finetune)
		trainer = Trainer(self.model,
							optimizer_finetune,
							dataloader_finetune,
							op_path_finetune,
							batch_size
							)

		print('Fitting model on target unit')

		self.model = trainer.train(int(num_iters))

	def load_model_from_checkpoint(self,modelpath):

		cp = torch.load(modelpath)
		state_dict = cp['model_state_dict']
		self.model.load_state_dict(state_dict)













