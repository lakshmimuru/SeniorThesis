import numpy as np
import torch
from torch import nn
import random

data_init = np.float32(np.load('/Users/lakshmimurugappan/Desktop/Princeton/Senior Thesis/deep_synthetic_ctrl/datasets/asthma_placebo/data.npy',allow_pickle=True))
mask = np.load('/Users/lakshmimurugappan/Desktop/Princeton/Senior Thesis/deep_synthetic_ctrl/datasets/asthma_placebo/mask.npy',allow_pickle=True).astype(bool)
data_init[mask] = 0

def low_rank(data, sing_to_keep):

	data_flat = data.reshape(len(data),-1) 
	u, s, v = np.linalg.svd(data_flat.astype(float))

	#k = int(np.rint(len(s)*pct_to_keep/100))
	u_approx = u[:, :sing_to_keep]
	s_approx = np.diag(s[:sing_to_keep])
	v_approx = v[:sing_to_keep, :]

	approx_flat = np.dot(u_approx, np.dot(s_approx, v_approx))
	approx = approx_flat.reshape(data.shape[0], data.shape[1], data.shape[2])
	print('Low rank conversion done')
	return approx

topk=None
pre_int_len= 2
post_int_len= 2
K=274
weights=None

if topk is not None:
			get_indices = np.argpartition(weights,-topk)[-topk:] +1
			red_data = data_init[get_indices]

else:
	        red_data = np.delete(data_init,5,0)

red_data[:,:,:16] = low_rank(red_data[:,:,:16],55)
target_id = 5
target_data = data_init[target_id] 
data = np.insert(red_data,target_id,target_data,0)
seq_pool = [i for i in range(data.shape[0]) if i!=target_id]
seq_ids = random.sample(seq_pool,K)
time_range=20

interv_time = np.random.randint(pre_int_len, time_range-post_int_len)

pre_int_seq = data[seq_ids,interv_time - pre_int_len:interv_time]


def missing(pre_int_seq, mask_ratio):
		# create random array of floats in equal dimension to input sequence (pre_int_seq)
		rand = torch.rand(pre_int_seq.shape)
		# where the random array is less than the mask ratio, we set true
		missing_mask = rand < mask_ratio
		mask_token=0
		pre_int_seq[missing_mask]=mask_token
		return missing_mask, pre_int_seq
