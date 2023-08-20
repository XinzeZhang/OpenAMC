from task.base.TaskLoader import TaskDataset
import pickle
import numpy as np
import torch

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

        self.classes = {b'QAM16': 0, b'QAM64': 1, b'8PSK': 2, b'WBFM': 3, b'BPSK': 4,
                        b'CPFSK': 5, b'AM-DSB': 6, b'GFSK': 7, b'PAM4': 8, b'QPSK': 9, b'AM-SSB': 10}

        self.post_data_file = 'data/RML2016.10a/RML2016.10a_dict.split.pt'

    def load_rawdata(self, logger=None):
        file_pointer = 'data/RML2016.10a/RML2016.10a_dict.pkl'

        if logger is not None:
            logger.info('*'*80 + '\n' +
                        f'Loading raw file in the location: {file_pointer}')

        Signals = []
        Labels = []
        SNRs = []

        Set = pickle.load(open(file_pointer, 'rb'), encoding='bytes')
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