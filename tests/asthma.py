import torch
import torch.nn as nn
import os
import sys
import numpy as np
import argparse
import yaml
from matplotlib import pyplot as plt
sys.path.append('../dsc/')
sys.path.append("../..")
sys.path.append("../../tslib/src/")
sys.path.append("../../tslib/")
sys.path.append(os.getcwd())
# sys.path.append('/Users/lakshmimurugappan/Desktop/Princeton/Senior Thesis/deep_synthetic_ctrl/transformers')
from dsc.dsc_model import DSCModel
from models.bert2bert import Bert2BertSynCtrl
from transformers import BertConfig
#from transformers import tsUtils
import pandas as pd

# Pretraining Asthma dataset
datapath = f'../datasets/asthma_placebo/'
#config_path = '../exp_configs/asthma/config.yaml'
config_path= "/Users/lakshmimurugappan/Desktop/Princeton/Senior Thesis/deep_synthetic_ctrl/exp_configs/asthma/config.yaml"
config = yaml.load(open(config_path,'r'),Loader=yaml.FullLoader)
op_path = f'/Users/lakshmimurugappan/Desktop/Princeton/Senior Thesis/deep_synthetic_ctrl/logs_dir/asthma_control/'
if not(os.path.exists(op_path)):
    os.mkdir(op_path)
random_seed = 0
target_index = 0
lowrank = True
device = torch.device('cuda:0' if torch.cuda.is_available else "cpu")
classes = None
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

model = Bert2BertSynCtrl(config_model, random_seed)
model = model.to(device)
dscmodel = DSCModel(model,
                    config,
                    op_path,
                    target_index,
                    random_seed,
                    datapath,
                    device,
                    lowrank = True,
                    classes=None)
dscmodel.pretrain(checkpoint_pretrain = None)

#Finetuning Asthma dataset

interv_times = [7,15]

for interv_time in interv_times:
    datapath = f'../datasets/asthma_placebo/'
    config_path = '../exp_configs/asthma/config.yaml'
    config = yaml.load(open(config_path,'r'),Loader=yaml.FullLoader)
    model_path = f'../logs_dir/asthma_control/pretrain/model.pth'
    op_path = f'../logs_dir/asthma_control_{interv_time}/'
    if not(os.path.exists(op_path)):
        os.mkdir(op_path)
    random_seed = 0
    target_index = 0
    lowrank = True
    device = torch.device('cuda:0' if torch.cuda.is_available else "cpu")
    classes = None
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
    model = Bert2BertSynCtrl(config_model, random_seed)
    model = model.to(device)
    dscmodel = DSCModel(model,
                        config,
                        op_path,
                        target_index,
                        random_seed,
                        datapath,
                        device,
                        lowrank = True,
                        classes=None)
    dscmodel.load_model_from_checkpoint(model_path)
    dscmodel.fit(interv_time,pretrain=False)


