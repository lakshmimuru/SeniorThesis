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
sys.path.append('../generator/')
sys.path.append('../transformers/src/')
from bert2bert import Bert2BertSynCtrl
from transformers import BertConfig
from trainer import Trainer 
from generator import Generator

def generate(args):

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
	model.load_state_dict(state_dict)
	generator = Generator(model,
						device,
						args.datapath,
						target_id,
						interv_time,
						lowrank)

	target_data =  generator.sliding_window_generate()
	np.save(args.op_path+'control.npy',target_data)

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
	parser.add_argument('--data_transform',type=str,default=None)
	args = parser.parse_args()
	generate(args)

if __name__ == "__main__":
    main()
