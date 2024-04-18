import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))

from task.TaskParser import get_parser
from task.base.TaskWrapper import Task
from task.base.TaskLoader import TaskDataset

import pickle
import numpy as np
import torch
import h5py

from exp_config._baseSetting import AMC_Net_base, AWN_base
from data.RML import RML2018_01a_Data as Data


class amcnet(AMC_Net_base):
    def task_modify(self):
        self.hyper.extend_channel = 36
        self.hyper.num_heads = 2
        self.hyper.conv_chan_list = [36, 64, 128, 256]        
        self.hyper.pretraining_file = ''

class awn(AWN_base):    
    def task_modify(self):
        self.hyper.num_level = 4
        self.hyper.regu_details = 0.01
        self.hyper.regu_approx = 0.01
        self.hyper.kernel_size = 3
        self.hyper.in_channels = 64
        self.hyper.latent_dim = 320    
        self.hyper.pretraining_file = 'data/RML2018.01a/pretrain_models/2018.01a_AWN.pt'    
    
if __name__ == "__main__":
    args = get_parser()
    args.cuda = True
    
    args.exp_config = os.path.dirname(sys.argv[0]).replace(os.getcwd()+'/', '')
    args.exp_file = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    args.exp_name = 'paper.test'
    
    args.test = True
    args.clean = False
    args.model = 'amcnet'
    
    
    task = Task(args)
    task.conduct()
            
    