#Author:Bhishma Dedhia




import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import time
import torch.nn as nn



class Trainer:

    def __init__(self,
                model,
                optimizer, 
                dataloader,
                output_file,
                batch_size,
                scheduler=None,
                log_freq = 100):
        
        self.model = model
        self.config = model.config
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = dataloader
        self.loss_fn_cont = nn.MSELoss()
        self.loss_fn_discrete = nn.BCEWithLogitsLoss()
        self.output_file= output_file
        self.batch_size= batch_size
        self.log_freq = log_freq
  


    def train(self, num_iters, load_from_checkpoint = None):

        self.model.train()
        log = dict()
        start_iter = 0

        if load_from_checkpoint is not None:
            
            checkpoint = torch.load(checkpoint)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_iter = checkpoint['iters']+1
            log = checkpoint['logs']
            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        loss_accumulate = []

        for i in range(start_iter,start_iter+num_iters):
            
            pre_int_seq, post_int_seq, timestamps_pre_int, timestamps_post_int, seq_ids_pre_int,\
            seq_ids_post_int, target_pre_int, target_post_int, attention_mask_pre_int, attention_mask_post_int, \
            target_cont, target_discrete = self.train_loader.get_batch(self.batch_size)
            

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

            target_cont_mask = target_cont[attention_mask_post_int[:,self.config.K-1]]
            cont_pred_mask = cont_pred[attention_mask_post_int[:,self.config.K-1]]

            loss_cont = self.loss_fn_cont(cont_pred_mask,target_cont_mask)
            loss_discrete = 0
            if discrete_pred is not None:
                for i in range(discrete_pred):
                    discrete_pred_mask = discrete_pred[i][attention_mask_post_int[:,self.config.K-1]]
                    target_discrete_mask = target_discrete[:,:,i][attention_mask_post_int[:,self.config.K-1]]
                    loss_discrete = loss_discrete + self.loss_fn_discrete(discrete_pred_mask,target_discrete_mask)
            loss = loss_cont+loss_discrete
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_accumulate.append(loss.item())
            
            if self.scheduler is not None:
                self.scheduler.step()

            if i%self.log_freq == 0:

                loss_accumulate = np.array(loss_accumulate)
                log[str(i)] = {'train_loss_mean':np.mean(loss_accumulate),'train_loss_std':np.std(loss_accumulate)}
                print(f'Iteration:{i}\tLoss_mean:{np.mean(loss_accumulate)}\tLoss_std:{np.std(loss_accumulate)}')
                loss_accumulate = []

                checkpoint = {'model_state_dict':self.model.state_dict(),
                            'optimizer_state_dict':self.optimizer.state_dict(),
                            'logs':log,
                            'iters':i
                            }
                if self.scheduler is not None:
                    checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

                torch.save(checkpoint,self.output_file+'model.pth')

        return self.model
                
