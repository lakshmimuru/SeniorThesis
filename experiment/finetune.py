#Author:Bhishma Dedhia

import torch
import torch.nn as nn
import os
import sys
import numpy as np
import argparse
import yaml
from collections import OrderedDict
sys.path.append('../models/')
sys.path.append('../training/')
sys.path.append('../dataloader/')
sys.path.append('../transformers/src/')
from bert2bert import Bert2BertSynCtrl
from transformers import BertConfig
from trainer import Trainer 
from dataloader import FinetuneDataLoader

def run_finetuning(args, num_iters=1e4):

	device = torch.device('cuda:0' if torch.cuda.is_available else "cpu")	

	if not(os.path.exists(args.op_path)):
		os.mkdir(args.op_path)


	config = yaml.load(open(args.config,'r'),Loader=yaml.FullLoader)
	
	if args.target_index is not None:
		target_id = args.target_index
		interv_time = args.interv_time
		classes = None 
	
	elif args.exp_name == 'basque':
		target_id = 16#basque county has index 16
		interv_time = 15
		classes = None 

	elif args.exp_name == 'germany':
		target_id = 6# west germany has index 6
		interv_time = 30
		classes = None 

	elif args.exp_name == 'prop99':
		target_id = 2 #california has index 2
		interv_time = 18
		classes = None 


	#elif exp_name == 'retail':

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
	model = model.to(device)
	cp = torch.load(args.modelpath+'model.pth')
	state_dict = cp['model_state_dict']
	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		name = k
		if k[:7] == 'module.':
			name = k[7:] # remove 'module.' of dataparallel
		new_state_dict[name]=v
	model.load_state_dict(new_state_dict)
	dataloader = FinetuneDataLoader(args.random_seed,
									args.datapath,
									device,
									config_model,
									target_id,
									interv_time,
									lowrank_approx = lowrank)

	optimizer = torch.optim.AdamW(model.parameters(),
								lr=eval(config['lr']),
								weight_decay=eval(config['weight_decay']),
								)
	batch_size = config['batch_size']
	trainer = Trainer(model,
						optimizer,
						dataloader,
						args.op_path,
						batch_size
						)

	print(f'Running finetuning with batch_size {batch_size}')

	trainer.train(int(num_iters),args.checkpoint)


def main():

	'''Pretrains Bert2Bert synthetic ctrl txf'''
	parser = argparse.ArgumentParser(description='Experiment parameters',
									formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--exp_name',type=str,default='basque')
	parser.add_argument('--datapath',type=str,default='')
	parser.add_argument('--modelpath',type=str,default = '')
	parser.add_argument('--config',type=str, default='')
	parser.add_argument('--op_path',type=str,default='')
	parser.add_argument('--random_seed',type=int,default=0)
	parser.add_argument('--target_index',type=int,default=None)
	parser.add_argument('--interv_time',type=int,default=None)
	parser.add_argument('--checkpoint',type=str,default=None)
	parser.add_argument('--data_transform',type=str,default=None)
	args = parser.parse_args()
	run_finetuning(args)

if __name__ == "__main__":
    main()
