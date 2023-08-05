import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))

from task.TaskParser import get_parser
from task.TaskWrapper import Task
from task.TaskLoader import TaskDataset

import pickle
import numpy as np
import torch

from models._Setting import AMC_Net_base

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
        
        self.num_classes = 11
        self.classes = {b'QAM16': 0, b'QAM64': 1, b'8PSK': 2, b'WBFM': 3, b'BPSK': 4,b'CPFSK': 5, b'AM-DSB': 6, b'GFSK': 7, b'PAM4': 8, b'QPSK': 9, b'AM-SSB': 10}
        
        
        file_pointer = 'data/RML2016.10a/RML2016.10a_dict.pkl'
        
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
        
        self.Signals = Signals
        self.Labels = Labels
        self.SNRs= SNRs
        self.snrs = snrs
        self.mods = mods

    def pack_dataset(self):
        """
        Split the preprocessed data, and pack them to self.train_set, self.val_set, self.test_set, self.test_idx
        """
        # toDo: add if/else to auto load pre-split dataset
        
        return super().pack_dataset()

class amcnet(AMC_Net_base):
    def task_modify(self):
        self.hyper.extend_channel = 36
        self.hyper.latent_dim = 512
        self.hyper.num_heads = 2
        self.hyper.conv_chan_list = [36, 64, 128, 256]        
        self.pretraining_file = ''
        
if __name__ == "__main__":
    args = get_parser()
    args.cuda = True
    
    args.expdatafolder = 'exp_config/rml16a'
    args.exp_file= 'demo'
    args.exp_name = 'paper.test'
    
    args.test = True
    args.model = 'amcnet'
    
    
    task = Task(args)
    task.conduct()
    task.evaluate()
    