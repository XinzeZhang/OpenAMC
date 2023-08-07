import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))

from task.TaskParser import get_parser
from task.TaskWrapper import Task
from task.TaskLoader import TaskDataset

import pickle
import numpy as np
import torch
import h5py

from models._Setting import AMC_Net_base, AWN_base



class Data(TaskDataset):
    def __init__(self, opts):
        '''Merge the input args to the self object'''
        super().__init__(opts)
    
    def rawdata_config(self) -> object:
        self.data_name = 'RML2018.01a'
        self.batch_size = 512
        self.sig_len = 1024
        
        self.val_size = 0.2
        self.test_size = 0.2
        
        self.num_classes = 24
        self.classes = {b'00K': 0, b'4ASK': 1, b'8ASK': 2, b'BPSK': 3, b'QPSK': 4, b'8PSK': 5, b'16PSK': 6, b'32PSK': 7, b'16APSK': 8, b'32APSK': 9,b'64APSK': 10, b'128APSK': 11, b'16QAM': 12, b'32QAM': 13, b'64QAM': 14, b'128QAM': 15, b'256QAM': 16, b'AM-SSB-WC': 17, b'AM-SSB-SC': 18,b'AM-DSB-WC': 19, b'AM-DSB-SC': 20, b'FM': 21, b'GMSK': 22, b'OQPSK': 23}
        self.post_data_file = 'data/RML2018.01a/RML2018.01a_dict.split.pt'
        
    def load_rawdata(self, logger = None):
        file_pointer = 'data/RML2018.01a/GOLD_XYZ_OSC.0001_1024.hdf5'
        
        if logger is not None:
            logger.info('*'*80 + '\n' +f'Loading raw file in the location: {file_pointer}')
        
        Signals, Labels, SNRs  = [], [], []
        
        f = h5py.File(file_pointer)
        Signals, Labels, SNRs  = f['X'][:], f['Y'][:], f['Z'][:]
        f.close()

        Signals = torch.from_numpy(Signals.astype(np.float32))
        Signals = Signals.permute(0, 2, 1)  # X:(2555904, 2, 1024)

        SNRs = SNRs.tolist()
        snrs = list(np.unique(SNRs))
        mods = list(self.classes.keys())

        Labels = np.argwhere(Labels == 1)[:, 1]
        Labels = np.array(Labels, dtype=np.int64)
        Labels = torch.from_numpy(Labels)
        
        return Signals, Labels, SNRs, snrs, mods

class amcnet(AMC_Net_base):
    def task_modify(self):
        self.hyper.extend_channel = 36
        self.hyper.latent_dim = 512
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
    args.model = 'awn'
    
    
    task = Task(args)
    task.conduct()
    task.evaluate(force_update=True)
            
    