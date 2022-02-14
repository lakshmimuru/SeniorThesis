#Author:Bhishma Dedhia

import torch
import torch.nn as nn
import os
import sys
import numpy as np
import argparse
import yaml
sys.path.append('../dsc/')
from dsc_model import DSCModel
from bert2bert import Bert2BertSynCtrl
from transformers import BertConfig

def fitandpredict(args):

	print(f'Running experiment {args.exp_name}')

	device = torch.device('cuda:0' if torch.cuda.is_available else "cpu")	

	if not(os.path.exists(args.op_path)):
		os.mkdir(args.op_path)

	config = yaml.load(open(args.config,'r'),Loader=yaml.FullLoader)
	
	if args.target_index is not None:
		target_id = args.target_index
		interv_time = args.interv_time
		classes = None 
	
	elif args.exp_name == 'basque':
		target_id = 16#basque country has index 16
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

	elif args.exp_name == 'synthetic':
		target_id = 0 #synthetic target has index 0
		interv_time = 1600
		classes = None 


	if args.lowrank is not None:
		lowrank = True
	else:
		lowrank = False


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

	model = Bert2BertSynCtrl(config_model, args.random_seed)
	model = model.to(device)
	dscmodel = DSCModel(model,
	                    config,
	                    args.op_path,
	                    target_id,
	                    args.random_seed,
	                    args.datapath,
	                    device,
	                    lowrank = lowrank,
	                    classes= classes)

	dscmodel.fit(interv_time)
	op = dscmodel.predict(interv_time)
	np.save(args.op_path+'target.npy',op)

def main():

	'''Pretrain finetune and predict '''
	parser = argparse.ArgumentParser(description='Experiment parameters',
									formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--exp_name',type=str,default='basque')
	parser.add_argument('--datapath',type=str,default='')
	parser.add_argument('--config',type=str, default='')
	parser.add_argument('--op_path',type=str,default='')
	parser.add_argument('--random_seed',type=int,default=0)
	parser.add_argument('--target_index',type=int,default=None)
	parser.add_argument('--interv_time',type=int,default=None)
	parser.add_argument('--lowrank',type=str,default=None)
	args = parser.parse_args()
	fitandpredict(args)

if __name__ == "__main__":
    main()
