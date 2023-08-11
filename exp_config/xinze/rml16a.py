import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))

from task.TaskParser import get_parser
from task.TaskWrapper import Task
from task.TaskLoader import TaskDataset

import pickle
import numpy as np
import torch

from ray import tune

from models._baseSetting import AMC_Net_base, AWN_base, mcldnn_base, vtcnn2_base, dualnet_base

class Data(TaskDataset):
    def __init__(self, opts):
        '''Merge the input args to the self object'''
        super().__init__(opts)
    
    def rawdata_config(self) -> object:
        self.data_name = 'RML2016.10a'
        self.batch_size = 64
        self.sig_len = 128
        
        self.val_size = 0.2
        self.test_size = 0.2
        
        self.classes = {b'QAM16': 0, b'QAM64': 1, b'8PSK': 2, b'WBFM': 3, b'BPSK': 4,b'CPFSK': 5, b'AM-DSB': 6, b'GFSK': 7, b'PAM4': 8, b'QPSK': 9, b'AM-SSB': 10}
        
        self.post_data_file = 'data/RML2016.10a/RML2016.10a_dict.split.pt'
        
    def load_rawdata(self, logger = None):
        file_pointer = 'data/RML2016.10a/RML2016.10a_dict.pkl'
        
        if logger is not None:
            logger.info('*'*80 + '\n' +f'Loading raw file in the location: {file_pointer}')
        
        Signals = []
        Labels = []
        SNRs = []
        
        Set = pickle.load(open(file_pointer, 'rb'), encoding='bytes')
        snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Set.keys())))), [1, 0])
        for mod in mods:
            for snr in snrs:
                Signals.append(Set[(mod, snr)])
                for i in range(Set[(mod, snr)].shape[0]):
                    Labels.append(mod)
                    SNRs.append(snr)
                    
        Signals = np.vstack(Signals)
        Signals = torch.from_numpy(Signals.astype(np.float32))

        Labels = [self.classes[i] for i in Labels]  # mapping modulation formats(str) to int
        Labels = np.array(Labels, dtype=np.int64)
        Labels = torch.from_numpy(Labels)
        
        return Signals, Labels, SNRs, snrs, mods

class amcnet(AMC_Net_base):
    def task_modify(self):
        self.hyper.extend_channel = 36
        self.hyper.latent_dim = 512
        self.hyper.num_heads = 2
        self.hyper.conv_chan_list = [36, 64, 128, 256]        
        self.hyper.pretraining_file = 'data/RML2016.10a/pretrain_models/2016.10a_AMC_Net.best.pt'
    

class awn(AWN_base):
    def task_modify(self):
        self.hyper.num_level = 1
        self.hyper.regu_details = 0.01
        self.hyper.regu_approx = 0.01
        self.hyper.kernel_size = 3
        self.hyper.in_channels = 64
        self.hyper.latent_dim = 320    
        self.hyper.pretraining_file = 'exp_tempTest/RML2016.10a/icassp23/fit/awn/checkpoint/RML2016.10a_awn.best.pt'
    


class vtcnn(vtcnn2_base):
    def task_modify(self):
        self.hyper.epochs = 100
        self.hyper.patience = 10
        self.hyper.gamma = 0.5
        
        self.tuner.resource = {
            "cpu": 20,
            "gpu": 1  # set this for GPUs
        }

class awn2(awn):
    def ablation_modify(self):
        self.hyper.epochs = 100
        self.hyper.pretraining_file = ''
        self.tuner.num_samples = 2
        self.tuner.resource = {
            "cpu": 5,
            "gpu": 0.5  # set this for GPUs
        }

class dualnet(dualnet_base):
    def task_modify(self):
        self.hyper.epochs = 200
        self.hyper.patience = 15

class mcl(mcldnn_base):
    def task_modify(self):
        self.hyper.epochs = 200

        self.tuner.num_samples = 40
        self.tuner.training_iteration = 200
        self.tuner.resource = {
            "cpu": 10,
            "gpu": 1  # set this for GPUs
        }
        
        self.tuner.points_to_evaluate=[{
            'lr':0.001,
            'gamma':0.8,
            'patience':60,
            'milestone_step':5,
            'batch_size': 400
        }]
        self.tuner.using_sched = False
        self.tuning.lr = tune.loguniform(1e-4, 1e-2)
        self.tuning.gamma = tune.uniform(0.5,0.99)
        self.tuning.milestone_step = tune.qrandint(2,10,1)
        self.tuning.patience = tune.qrandint(5,100,5)
        self.tuning.batch_size = tune.choice([64, 128, 192, 256, 320, 384, 400])

if __name__ == "__main__":
    args = get_parser()
    args.cuda = True
    
    args.exp_config = os.path.dirname(sys.argv[0]).replace(os.getcwd()+'/', '')
    args.exp_file = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    # args.exp_name = 'icassp23'
    args.exp_name = 'tuning.mcl'
    # args.gid = 0
    
    args.test = True
    args.clean = True
    # args.model = 'awn2'
    
    
    task = Task(args)
    task.tuning()
    task.conduct()
            
    