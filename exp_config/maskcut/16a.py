import os
import sys
sys.path.append(os.path.join(os.path.dirname(
    __file__), os.path.pardir, os.path.pardir))
from models.base._baseSetting import AMC_Net_base, AWN_base, mcldnn_base, vtcnn2_base, dualnet_base, resnet_base, cldnn_base, pcnn_base
from ray import tune
import torch
import numpy as np
import pickle
from task.base.TaskLoader import Opt 
from task.TaskParser import get_parser
from data.RML201610a import RML2016_10a_Data
from task.maskcut.maskLoader import maskcut_Dataset
from task.maskcut.maskWrapper import maskcutTask

class Data(maskcut_Dataset, RML2016_10a_Data):
    def __init__(self, opts):
        super().__init__(opts)
        
    def pack_info(self):
        '''The info attributes will be automatically set to the model_opts.hyper
        '''
        self.info = Opt()
        self.info.merge(self, ['data_name', 'batch_size', 'sig_len', 'num_classes', 'post_data_file'])            
        for i in range(self.sig_len):
            self.info.dict['inputMask_{}'.format(i)] = 1



class amcnet(AMC_Net_base):
    def task_modify(self):
        self.hyper.extend_channel = 36
        self.hyper.latent_dim = 512
        self.hyper.num_heads = 2
        self.hyper.conv_chan_list = [36, 64, 128, 256]
        self.hyper.patience = 20
        # self.hyper.pretraining_file = '/home/xinze/Documents/Github/OpenAMC/data/RML2016.10a/pretrain_models/RML2016.10a_AMC_Net.best.pt'
        
        self.tuner.resource = {"gpu": 0.5}
        # self.tuner.algo = 'pso'
        self.tuner.num_samples = 60
        self.tuner.using_sched = True
        self.tuner.min_training_iteration = 30
        self.tuner.max_training_iteration = 200
        self.tuner.odd_check = False
        
        self.tuning = Opt()
        self.tuning.merge({'inputMask_{}'.format(i) :  tune.choice([0,1]) for i in range(128)})
        
class awn(AWN_base):
    def task_modify(self):
        self.hyper.num_level = 1
        self.hyper.regu_details = 0.01
        self.hyper.regu_approx = 0.01
        self.hyper.kernel_size = 3
        self.hyper.in_channels = 64
        self.hyper.latent_dim = 320
        # self.hyper.pretraining_file = '/home/xinze/Documents/Github/OpenAMC/data/RML2016.10a/pretrain_models/RML2016.10a_AWN.best.pt'
        self.hyper.patience = 20

        self.tuner.resource = {"gpu": 0.5}
        self.tuner.num_samples = 90
        self.tuner.using_sched = True
        self.tuner.min_training_iteration = 30
        self.tuner.max_training_iteration = 200
        self.tuner.odd_check = True
        
        self.tuning = Opt()
        self.tuning.merge({'inputMask_{}'.format(i) :  tune.choice([0,1]) for i in range(128)})
        
class res(resnet_base):
    def task_modify(self):
        self.hyper.epochs = 100
        self.hyper.batch_size = 1024
        self.hyper.milestone_step = 1
        self.hyper.patience = 20
        # self.hyper.pretraining_file ='/home/xinze/Documents/Github/OpenAMC/data/RML2016.10a/pretrain_models/RML2016.10a_ResNet.best.pt'
        
        self.tuner.num_samples = 60
        self.tuner.resource = {"gpu": 0.33}
        self.tuner.using_sched = False
        self.tuner.min_training_iteration = 30
        self.tuner.max_training_iteration = 200
        self.tuner.odd_check = False
        
        self.tuning = Opt()
        self.tuning.merge({'inputMask_{}'.format(i) :  tune.choice([0,1]) for i in range(128)})
                
        
if __name__ == "__main__":
    args = get_parser()
    args.cuda = True
    # args.gid = 2

    args.exp_config = os.path.dirname(sys.argv[0]).replace(os.getcwd()+'/', '')
    args.exp_file = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    # args.exp_name = 'ICASSP24'
    # args.exp_name = 'cldnn.test'
    # args.exp_name = 'ICASSP24.subsampling'
    # args.force_update = True # if need to rerun the model fiting or reload the pretraining_file (if os.path.exit(hyper.pretraining_file) to get the results, uncomment this line.)

    args.test = True
    args.clean = True
    # args.model = 'awn'
    args.exp_name = f'ICASSP24.{args.model}.maskcut'
    
    
    task = maskcutTask(args)
    task.tuning()
    task.conduct()
