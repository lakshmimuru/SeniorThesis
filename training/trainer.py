#Author:Bhishma Dedhia


import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import time
import torch.nn as nn



def  return_dataloader(filename, batch_size, train = True):

	if train == True:
		filename = filename+'train/'
	else:
		filename = filename + 'eval/'

	pre_int_data = torch.from_numpy(np.load(filename+'pre_int_data.npz',mmap_mode = 'r')).type(torch.FloatTensor)
	post_int_data = torch.from_numpy(np.load(filename+'post_int_data.npz',mmap_mode = 'r')).type(torch.FloatTensor)
	pre_int_timestamps = torch.from_numpy(np.load(filename+'pre_int_timestamps.npz',mmap_mode = 'r')).type(torch.LongTensor)
	post_int_timestamps = torch.from_numpy(np.load(filename+'post_int_timestamps.npz',mmap_mode = 'r')).type(torch.LongTensor)
	pre_int_seq_ids = torch.from_numpy(np.load(filename+'pre_int_seq_ids.npz',mmap_mode = 'r')).type(torch.LongTensor)
	post_int_seq_ids = torch.from_numpy(np.load(filename+'post_int_seq_ids.npz',mmap_mode='r')).type(torch.LongTensor)
	pre_int_target_mask = torch.from_numpy(np.load(filename+'pre_int_target_mask.npz',mmap_mode = 'r')).type(torch.LongTensor)
	post_int_target_mask = torch.from_numpy(np.load(filename+'post_int_target_mask.npz',mmap_mode = 'r')).type(torch.LongTensor)
	pre_int_attention_mask = torch.from_numpy(np.load(filename+'pre_int_attention_mask.npz',mmap_mode = 'r')).type(torch.LongTensor)
	post_int_attention_mask = torch.from_numpy(np.load(filename+'post_int_attention_mask.npz',mmap_mode = 'r')).type(torch.LongTensor)
	post_int_target_cont = torch.from_numpy(np.load(filename+'post_int_cont_targets.npz',mmap_mode = 'r')).type(torch.FloatTensor)
	post_int_target_discrete = torch.from_numpy(np.load(filename+'post_int_discrete_targets.npz',mmap_mode = 'r')).type(torch.LongTensor)

	dataset = TensorDataset(pre_int_data, post_int_data, pre_int_timestamps,
							post_int_timestamps, pre_int_seq_ids, post_int_seq_ids, 
							pre_int_target_mask, post_int_target_mask, pre_int_attention_mask,
							post_int_attention_mask, post_int_target_cont, post_int_target_discrete)

	dataloader = DataLoader(dataset, batch_size=batch_size)

	return dataloader


class PreTrainer:

    def __init__(self,
				model,
				config,
				optimizer, 
				datapath_train, 
				datapath_eval, 
				output_file,
				device,
				scheduler=None):
        
        self.model = model
        if torch.cuda.device_count() > 1:
  			model = nn.DataParallel(model)
  		self.config = config
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.scheduler = scheduler
        self.train_loader = return_dataloader(datapath_train, batch_size = self.batch_size, train = True)
        self.eval_loader = return_dataloader(datapath_eval, batch_size = self.batch_size, train = False)
        self.loss_fn_cont = nn.MSELoss()
        self.loss_fn_discrete = nn.BCEWithLogitsLoss()
        self.output_file= output_file
        self.device = device


    def train(self, epochs, log_freq = 100, load_from_checkpoint = None):

    	self.model.train()
    	log = dict()
    	start_epoch = 0
    	if load_from_checkpoint is not None:
    		#load model, scheduler, optimizer, start epoch
    		checkpoint = torch.load(checkpoint)
    		self.model.load_state_dict(checkpoint['model_state_dict'])
    		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    		start_epoch = checkpoint['epoch']+1
    		log = checkpoint['logs']
    		if self.scheduler is not None:
    			self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    	for epoch in range(start_epoch,start_epoch+epochs):
    		log_i = dict()
    		for i_batch, (pre_int_seq, post_int_seq, timestamps_pre_int, timestamps_post_int, seq_ids_pre_int, 
				seq_ids_post_int, target_pre_int, target_post_int, attention_mask_pre_int, attention_mask_post_int, 
				target_cont, target_discrete)  in enumerate(self.train_loader):
    			
    			'''
    			pre_int_seq = pre_int_seq.to(self.device)
    			post_int_seq = post_int_seq.to(self.device)
    			timestamps_pre_int = timestamps_pre_int.to(self.device)
    			timestamps_post_int = timestamps_post_int.to(self.device)
    			seq_ids_pre_int = seq_ids_pre_int.to(self.device)
				seq_ids_post_int = seq_ids_post_int.to(self.device)
				target_pre_int =  target_pre_int.to(self.device)
				target_post_int =  target_post_int.to(self.device)
				attention_mask_pre_int = attention_mask_pre_int.to(self.device)
				attention_mask_post_int =  attention_mask_post_int.to(self.device)
				target_cont =  target_cont.to(self.device)
				target_discrete = target_discrete.to(self.device)
				'''

				cont_pred, discrete_pred = self.model(pre_int_seq,
													  post_int_seq,
													  timestamps_pre_int, 
													  timestamps_post_int, 
													  seq_ids_pre_int, 
													  seq_ids_post_int, 
													  target_pre_int, 
													  target_post_int, 
													  attention_mask_pre_int, 
													  attention_mask_post_int,)

				loss_cont = self.loss_fn_cont(cont_pred,target_cont)
				loss_discrete = 0
				if discrete_pred is not None:
					for i in range(discrete_pred):
						loss_discrete = loss_discrete + self.loss_fn_discrete(discrete_pred[i],target_discrete[:,:,i])
				loss = loss_cont+loss_discrete
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				
				if self.scheduler is not None:
                	self.scheduler.step()

                if ibatch%log_freq == 0:


                	#log, print and save models/optimizer/scheduler
                	self.model.eval()
                	eval_losses = []

                	for i_batch_eval,(pre_int_seq, post_int_seq, timestamps_pre_int, timestamps_post_int, seq_ids_pre_int, 
						seq_ids_post_int, target_pre_int, target_post_int, attention_mask_pre_int, attention_mask_post_int, 
						target_cont, target_discrete)  in enumerate(self.eval_loader):

                		'''
                		pre_int_seq = pre_int_seq.to(self.device)
		    			post_int_seq = post_int_seq.to(self.device)
		    			timestamps_pre_int = timestamps_pre_int.to(self.device)
		    			timestamps_post_int = timestamps_post_int.to(self.device)
		    			seq_ids_pre_int = seq_ids_pre_int.to(self.device)
						seq_ids_post_int = seq_ids_post_int.to(self.device)
						target_pre_int =  target_pre_int.to(self.device)
						target_post_int =  target_post_int.to(self.device)
						attention_mask_pre_int = attention_mask_pre_int.to(self.device)
						attention_mask_post_int =  attention_mask_post_int.to(self.device)
						target_cont =  target_cont.to(self.device)
						target_discrete = target_discrete.to(self.device)
						'''

                		
						cont_pred, discrete_pred = self.model(pre_int_seq,
															  post_int_seq,
															  timestamps_pre_int, 
															  timestamps_post_int, 
															  seq_ids_pre_int, 
															  seq_ids_post_int, 
															  target_pre_int, 
															  target_post_int, 
															  attention_mask_pre_int, 
															  attention_mask_post_int,)

						loss_cont = self.loss_fn_cont(cont_pred,target_cont)
						loss_discrete = 0
						if discrete_pred is not None:
							for i in range(discrete_pred):
								loss_discrete = loss_discrete + self.loss_fn_discrete(discrete_pred[i],target_discrete[:,:,i])
						eval_loss = loss_cont+loss_discrete
						eval_losses.append(eval_loss.cpu().item())

					log_i[str(i_batch)] = {'train':loss.cpu().item() , 'eval': sum(eval_losses)/len(eval_losses)}
					print('[%d/%d][%d/%d]\tTrain Loss: %.4f\tEval Loss:%.4f'% (epoch, epochs,i_batch, 
																			len(self.train_loader),
																			loss.cpu().item(),sum(eval_losses)/len(eval_losses)))
					log[str(epoch)] = log_i
					checkpoint = {'model_state_dict':self.model.state_dict(),
								'optimizer_state_dict':self.optimizer.state_dict(),
								'logs':log,
								'epoch':epoch
								}
					if self.scheduler is not None:
						checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

					torch.save(checkpoint,self.output_file+str('model.pt')
					self.model.train()


					


























       



   		