#Author:Bhishma Dedhia

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
from bert2bert import Bert2BertSynCtrl
from transformers import BertConfig
from trainer import Trainer 
from dataloader import PreTrainDataLoader

def run_pretraining(args, num_iters=5e5):

	device = torch.device('cuda:0' if torch.cuda.is_available else "cpu")	

	if not(os.path.exists(args.op_path)):
		os.mkdir(args.op_path)


	config = yaml.load(open(args.config,'r'),Loader=yaml.FullLoader)
	
	if args.target_index is not None:
		target_id = args.target_index
		classes = None 
	
	elif exp_name == 'basque':
		target_id = 16#basque county has index 16
		classes = None 

	elif exp_name == 'germany':
		target_id = 6# west germany has index 6
		classes = None 

	elif exp_name == 'prop99':
		target_id = 2 #california has index 2
		classes = None 

	elif exp_name == 'retail':
		target_id = 
		classes = None

	#elif exp_name == 'cricket':

	config_model = BertConfig(hidden_size = config['hidden_size'],
							num_hidden_layers = config['n_layers'],
							num_attention_heads = config['n_heads'],
							intermediate_size = 4*config['hidden_size'],
							vocab_size = 0,
							max_position_embeddings = 0,
							output_hidden_states = True,
							)

	config_model.add_syn_ctrl_config(K=config['K'],
									pre_int_len=config['pre_int_len'],
									post_int_len=config['post_int_len'],
									feature_dim=config['feature_dim'],
									time_range=config['time_range'],
									seq_range=config['seq_range'],
									cont_dim=config['cont_dim'],
									discrete_dim=config['discrete_dim'],
									classes = classes)

	if args.data_transform is not None:
		lowrank = True

	else:
		lowrank = False
	model = Bert2BertSynCtrl(config_model, args.random_seed)
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
	model = model.to(device)
	dataloader = PreTrainDataLoader(args.random_seed,
									args.datapath,
									device,
									config_model,
									target_id,
									lowrank_approx = lowrank)

	optimizer = torch.optim.AdamW(model.parameters(),
								lr=eval(config['lr']),
								weight_decay=eval(config['weight_decay']),
								)
	warmup_steps = config['warmup_steps']
	scheduler = torch.optim.lr_scheduler.LambdaLR(
				optimizer,
				lambda steps: min((steps+1)/warmup_steps,1))
	batch_size = config['batch_size']
	trainer = Trainer(model,
						optimizer,
						dataloader,
						args.op_path,
						batch_size,
						scheduler
						)

	print(f'Running training with batch_size {batch_size}')

	trainer.train(int(num_iters),args.checkpoint)



def main():

	'''Pretrains Bert2Bert synthetic ctrl txf'''
	parser = argparse.ArgumentParser(description='Experiment parameters',
									formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--exp_name',type=str,default='lunarlander')
	parser.add_argument('--datapath',type=str,default='')
	parser.add_argument('--config',type=str, default='')
	parser.add_argument('--op_path',type=str,default='')
	parser.add_argument('--random_seed',type=int,default=0)
	parser.add_argument('--target_index',type=int,default=0)
	parser.add_argument('--checkpoint',type=str,default=None)
	parser.add_argument('--data_transform',type=str,default=None)
	args = parser.parse_args()
	run_pretraining(args)

if __name__ == "__main__":
    main()
