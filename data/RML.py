from task.base.TaskLoader import TaskDataset
import pickle
import numpy as np
import torch
import h5py

class RML2016_10a_Data(TaskDataset):
    '''Data config python file for RML2016.10a dataset
    '''
    def __init__(self, opts):
        '''Merge the input args to the self object'''
        super().__init__(opts)

    def rawdata_config(self) -> object:
        self.data_name = 'RML2016.10a'
        self.batch_size = 64
        self.sig_len = 128

        self.val_size = 0.2
        self.test_size = 0.2

        self.classes = {'QAM16': 0, 'QAM64': 1, '8PSK': 2, 'WBFM': 3, 'BPSK': 4,
                        'CPFSK': 5, 'AM-DSB': 6, 'GFSK': 7, 'PAM4': 8, 'QPSK': 9, 'AM-SSB': 10}

        self.post_data_file = 'data/RML2016.10a/RML2016.10a_dict.split.pt'

    def load_rawdata(self, logger=None):
        file_pointer = 'data/RML2016.10a/RML2016.10a_dict.pkl'

        if logger is not None:
            logger.info('*'*80 + '\n' +
                        f'Loading raw file in the location: {file_pointer}')

        Signals = []
        Labels = []
        SNRs = []

        Set = pickle.load(open(file_pointer, 'rb'), encoding='latin1')
        snrs, mods = map(lambda j: sorted(
            list(set(map(lambda x: x[j], Set.keys())))), [1, 0])
        for mod in mods:
            for snr in snrs:
                Signals.append(Set[(mod, snr)])
                for i in range(Set[(mod, snr)].shape[0]):
                    Labels.append(mod)
                    SNRs.append(snr)

        Signals = np.vstack(Signals)
        Signals = torch.from_numpy(Signals.astype(np.float32))

        # mapping modulation formats(str) to int
        Labels = [self.classes[i] for i in Labels]
        Labels = np.array(Labels, dtype=np.int64)
        Labels = torch.from_numpy(Labels)

        return Signals, Labels, SNRs, snrs, mods

class RML2016_10b_Data(TaskDataset):
    '''Data config python file for RML2016.10b dataset
    '''
    def __init__(self, opts):
        '''Merge the input args to the self object'''
        super().__init__(opts)

    def rawdata_config(self) -> object:
        self.data_name = 'RML2016.10b'
        self.batch_size = 128
        self.sig_len = 128

        self.val_size = 0.2
        self.test_size = 0.2

        self.classes = {'QAM16': 0, 'QAM64': 1, '8PSK': 2, 'WBFM': 3, 'BPSK': 4, 'CPFSK': 5, 'AM-DSB': 6, 'GFSK': 7, 'PAM4': 8, 'QPSK': 9}

        self.post_data_file = 'data/RML2016.10b/RML2016.10b_dict.split.pt'

    def load_rawdata(self, logger=None):
        file_pointer = 'data/RML2016.10b/RML2016.10b.dat'

        if logger is not None:
            logger.info('*'*80 + '\n' +
                        f'Loading raw file in the location: {file_pointer}')

        Signals = []
        Labels = []
        SNRs = []

        Set = pickle.load(open(file_pointer, 'rb'), encoding='latin1')
        snrs, mods = map(lambda j: sorted(
            list(set(map(lambda x: x[j], Set.keys())))), [1, 0])
        for mod in mods:
            for snr in snrs:
                Signals.append(Set[(mod, snr)])
                for i in range(Set[(mod, snr)].shape[0]):
                    Labels.append(mod)
                    SNRs.append(snr)

        Signals = np.vstack(Signals)
        Signals = torch.from_numpy(Signals.astype(np.float32))

        # mapping modulation formats(str) to int
        Labels = [self.classes[i] for i in Labels]
        Labels = np.array(Labels, dtype=np.int64)
        Labels = torch.from_numpy(Labels)

        return Signals, Labels, SNRs, snrs, mods
    

class RML2018_01a_Data(TaskDataset):
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