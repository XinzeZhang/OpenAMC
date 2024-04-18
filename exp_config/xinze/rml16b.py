import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))

from task.TaskParser import get_parser
from task.base.TaskWrapper import Task
from task.base.TaskLoader import TaskDataset

import pickle
import numpy as np
import torch

from exp_config._baseSetting import AMC_Net_base, AWN_base, mcldnn_base, vtcnn2_base, dualnet_base, resnet_base, cldnn_base, pcnn_base
from data.RML import RML2016_10b_Data
# from exp_config._model_config.rml16b import amcnet, awn, dualnet, mcl, res

class Data(RML2016_10b_Data):
    def __init__(self, opts):
        '''Merge the input args to the self object'''
        super().__init__(opts)


# class amcnet(AMC_Net_base):
#     def task_modify(self):
#         self.hyper.extend_channel = 36
#         self.hyper.num_heads = 2
#         self.hyper.conv_chan_list = [36, 64, 128, 256]        
#         self.hyper.pretraining_file = 'data/RML2016.10b/pretrain_models/2016.10b_AMC_Net.pt'
        # self.hyper.pretraining_file = 'data/RML2016.10b/pretrain_models/2016.10b_AMC_Net.official.pt'
    
# class awn(AWN_base):
#     def task_modify(self):
#         self.hyper.num_level = 1
#         self.hyper.regu_details = 0.01
#         self.hyper.regu_approx = 0.01
#         self.hyper.kernel_size = 3
#         self.hyper.in_channels = 64
#         self.hyper.latent_dim = 320    
#         self.hyper.pretraining_file = 'data/RML2016.10b/pretrain_models/RML2016.10b_AWN.0826.best.pt'


class mcl(mcldnn_base):
    def task_modify(self):
        self.hyper.batch_size = 128
        self.hyper.epochs = 200
        self.hyper.patience = 20
        # self.hyper.gamma = 0.5502
        # self.hyper.lr = 0.0015
        self.hyper.milestone_step = 2
        self.hyper.pretraining_file = 'data/RML2016.10b/pretrain_models/RML2016.10b_mcl.best.pt'

# class dualnet(dualnet_base):
#     def task_modify(self):
#         self.hyper.batch_size = 128
#         self.hyper.epochs = 200
#         self.hyper.patience = 15
#         self.hyper.pretraining_allDatafile = ''
#         self.hyper.pretraining_file = 'data/RML2016.10b/pretrain_models/RML2016.10b_dualnet.best.pt'

class cldnn(cldnn_base):
    def task_modify(self):
        self.hyper.batch_size = 1024
        self.hyper.milestone_step = 1
        self.hyper.patience = 20
        self.hyper.pretraining_allDatafile = ''
        self.hyper.pretraining_file = 'data/RML2016.10b/pretrain_models/RML2016.10b_cldnn.best.pt'
    
# class res(resnet_base):
#     def task_modify(self):
#         self.hyper.pretraining_file = 'data/RML2016.10b/pretrain_models/RML2016.10b_res.best.pt'
#         # self.hyper.batch_size = 1024
#         # self.hyper.milestone_step = 1
#         # self.hyper.patience = 20
#         # self.hyper.pretraining_allDatafile =''
#         # self.hyper.pretraining_file = ''
    
class vtcnn(vtcnn2_base):
    def task_modify(self):
        self.hyper.batch_size = 128
        self.hyper.patience = 20
                    
class pcnn(pcnn_base):
    def task_modify(self):
        self.hyper.batch_size = 128
        self.hyper.patience = 20
        
            
if __name__ == "__main__":
    args = get_parser()
    args.cuda = True
    
    args.exp_config = os.path.dirname(sys.argv[0]).replace(os.getcwd()+'/', '')
    args.exp_file = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    args.exp_name = 'Baselines'
    args.force_update = True
    # args.test = True
    # args.clean = True
    
    
    args.model = 'mcl'
    # args.tag = '0826'
    task = Task(args)
    task.conduct()

            
    