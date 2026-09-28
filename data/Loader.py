from collections.abc import Mapping
import copy
import numpy as np
import torch
import torch.utils.data as Data
import os
from taskRecog.util import set_fitset
from taskRecog.util import Opt

class TaskDataset(Opt):
    """
    The TaskDataset class is designed to manage and preprocess signal datasets ,
    including loading raw data, splitting it into training, validation, and test sets,
    saving and loading preprocessed datasets , handling SNR slicing.
    """

    def __init__(self, args=None):
        super().__init__()


        self.Signals = []
        self.Labels = []
        self.SNR_values= []
        self.snr_envs = []
        self.mods = []

        self.num_workers = 0 # used in func: self.load_dataset
        self.post_data_file = ''

        self.rawdata_config()
        self.num_classes = len(self.classes.keys())
        # self.num_snrs = list(np.unique(self.snr_envs))

        # self.load_rawdata()
        # self.pack_dataset()
        self.using_snr = True if hasattr(args, 'using_snr') and args.using_snr else False
        self.pack_info()


    def pack_info(self):
        self.info = Opt()
        self.info.merge(self, ['data_name', 'batch_size', 'sig_len', 'num_classes', 'post_data_file', 'using_snr', 'snr_envs'])

    def rawdata_config(self,):
        """
        Preprocess the raw data, put the Signals, Labels, SNRs, snrs, and mods to the self.rawData, and update the dataset related parameters.

        Parameters
        ----------
        data_name: str
        batch_size: int
        sig_len: int
        val_size: float
        test_size: float
        num_classes: int
        classes: dict

        """
        self.data_name = 'None'
        self.val_size = 0.2
        self.test_size = 0.2
        self.batch_size = 64
        self.sig_len = 128
        self.num_classes = 0
        self.classes = {}


    def load_rawdata(self, logger = None):
        """
        Loading the rawdata, personalized by args.exp_file in the args.exp_config
        """
        pass

    def pack_dataset(self, logger = None):
        """
        Split the preprocessed data, and pack them to self.train_set, self.val_set, self.test_set, self.test_idx
        """
        if os.path.exists(self.info.post_data_file):  #processed data is already available , examples :RML2016.10a_dict.split.pt
            try:
                split_data = torch.load(self.info.post_data_file, weights_only=False)
                self.train_set =split_data['train_set']
                self.test_set = split_data['test_set']
                self.val_set = split_data['val_set']
                self.train_idx = split_data['train_idx']
                self.val_idx = split_data['val_idx']
                self.test_idx = split_data['test_idx']
                self.SNR_values = split_data['SNRs']
                if len(self.snr_envs) > 0 and self.snr_envs != split_data['snrs']:
                    raise ValueError('The dataset default snrs is not consistent with pre-processed datafile in the location: {}\nDefault is {}\nLoading {}'.format(self.info.post_data_file,self.snr_envs,split_data['snrs']))
                self.snr_envs = split_data['snrs']

                self.mods = split_data['mods']

                if logger is not None:
                    logger.info(f'Loading pre-split file in the location: {self.info.post_data_file}')

            except:
                raise ValueError(f'Error when loading pre-processed datafile in the location: {self.info.post_data_file}')

        else:
            Signals, Labels, SNRs, snrs, mods = self.load_rawdata(logger) #load rawdata , split and save processed data
            self.SNR_values= SNRs
            self.snr_envs = snrs
            self.mods = mods

            if logger is not None:
                logger.info(f'Using the random seed: {self.info.seed}')
                logger.info('Split the dataset to training set, validation set, and test set with the ration of {:.2f}, {:.2f}, and {:.2f}'.format(1- self.test_size - self.val_size, self.val_size, self.test_size))

            self.train_set, self.val_set, self.test_set, self.train_idx, self.val_idx, self.test_idx = self.dataset_Split(Signals=Signals, Labels=Labels, snrs=self.snr_envs, mods=self.mods, val_size=self.val_size,test_size=self.test_size)

            parent_dir = os.path.dirname(self.info.post_data_file)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

            torch.save({
            'train_set': self.train_set,
            'test_set': self.test_set,
            'val_set': self.val_set,
            'train_idx': self.train_idx,
            'val_idx': self.val_idx,
            'test_idx': self.test_idx,
            'SNRs': self.SNR_values,
            'snrs': self.snr_envs,
            'mods':  self.mods
            }, self.info.post_data_file
            )

        if self.using_snr:
            snr_values_tensor = torch.tensor(self.SNR_values)
            train_snr_values = snr_values_tensor[self.train_idx]
            val_snr_values = snr_values_tensor[self.val_idx]
            self.train_set = (self.train_set[0], self.train_set[1], train_snr_values)
            self.val_set = (self.val_set[0], self.val_set[1], val_snr_values)
            # test_snr_values = self.SNR_values[self.test_idx]
            # self.test_set = (self.test_set[0], self.test_set[1], test_snr_values)
            test_snr_values = snr_values_tensor[self.test_idx]
            self.test_set = (self.test_set[0], self.test_set[1], test_snr_values)
            #modify2026/02/01
            #modify2026/02/01

        self.info.update(self, check_subset=False, only_interset=True)
        return self.train_set, self.val_set, self.test_set, self.test_idx


    @staticmethod
    def dataset_Split(Signals, Labels, snrs, mods, val_size=0.2, test_size=0.2):
        '''
        Split the data into train, validation, and test sets
        ensuring the same number of samples for each (mod, snr) in each set
        '''
        global test_idx
        n_examples = Signals.shape[0]
        n_train = int(n_examples * (1 - val_size - test_size))

        train_idx = []
        test_idx = []
        val_idx = []

        Slices_list = np.linspace(0, n_examples, num=len(mods) * len(snrs) + 1)

        for k in range(0, Slices_list.shape[0] - 1):
            train_idx_subset = np.random.choice(
                range(int(Slices_list[k]), int(Slices_list[k + 1])), size=int(n_train / (len(mods) * len(snrs))),
                replace=False)
            Test_Val_idx_subset = list(
                set(range(int(Slices_list[k]), int(Slices_list[k + 1]))) - set(train_idx_subset))
            test_idx_subset = np.random.choice(Test_Val_idx_subset,
                                               size=int(round(
                                                   (n_examples - n_train) * test_size / (
                                                       (len(mods) * len(snrs)) * (test_size + val_size))) ),
                                               replace=False)
            val_idx_subset = list(
                set(Test_Val_idx_subset) - set(test_idx_subset))

            train_idx = np.hstack([train_idx, train_idx_subset])
            val_idx = np.hstack([val_idx, val_idx_subset])
            test_idx = np.hstack([test_idx, test_idx_subset])

        train_idx = train_idx.astype('int64')
        val_idx = val_idx.astype('int64')
        test_idx = test_idx.astype('int64')

        Signals_train = Signals[train_idx]
        Labels_train = Labels[train_idx]

        Signals_test = Signals[test_idx]
        Labels_test = Labels[test_idx]

        Signals_val = Signals[val_idx]
        Labels_val = Labels[val_idx]

        # logger.info(f"Signal_train.shape: {list(Signals_train.shape)}", )
        # logger.info(f"Signal_val.shape: {list(Signals_val.shape)}")
        # logger.info(f"Signal_test.shape: {list(Signals_test.shape)}")
        # logger.info('*' * 20)

        return (Signals_train, Labels_train), \
            (Signals_val, Labels_val), \
            (Signals_test, Labels_test), \
            train_idx, val_idx, test_idx


    def snr_slice(self, data_tag = 'val', snr = 0):
        '''Return: \n
        return signals labels which has the specific snr
        sig_i: tensor (N, 2, sig_len)
        lab_i: tensor (N,)
        idx_i: list []
        '''
        data_dict = dict(val = (self.val_set, self.val_idx),
                         train= (self.train_set, self.train_idx),
                         test = (self.test_set,self.test_idx))

        data_set, data_idx = data_dict[data_tag]
        from taskRecog.util import snr_slice
        sig_i, lab_i, idx_i = snr_slice(data_set,data_idx,self.SNR_values, snr)
        return sig_i, lab_i, idx_i


    def load_fitset(self, fit_batch_size = None):
        """
        Load the self.train_set and self.val_set for fitting the model.

        Parameters:
            fit_batch_size: int, optional, default is self.batch_size
            If fit_batch_size is None, it will use the default batch size defined in self.batch_size.
            If fit_batch_size is specified, it will use that value instead.

        Return:
            train_loader, val_loader
        """
        _fit_batch_size = fit_batch_size if fit_batch_size is not None else self.batch_size
        train_loader, val_loader = set_fitset(batch_size= _fit_batch_size, num_workers=self.num_workers, train_set=self.train_set, val_set=self.val_set)

        return train_loader, val_loader

    def load_testset(self, test_batch_size = 64):
        """
        Return two lists:
        Sample_list contains all signal batches corresponding to each specific SNR value
        Label_list contains all label batches corresponding to each specific SNR value
        shape like :  (num_snrs,num_of_batch,batch_size,2,sig_len)
        """
        Sample_list = []
        Label_list = []

        if 'num_snrs' not in self.dict:
            self.num_snrs = list(np.unique(self.snr_envs))

        for snr in self.num_snrs:
            sig_i, lab_i, _ = self.snr_slice('test', snr)
            # print(f"Loading test set for SNR={snr} with {sig_i.shape[0]} samples.")
            num_chunk = int(sig_i.shape[0] / test_batch_size)

            Sample = torch.chunk(sig_i, num_chunk, dim=0)
            Label = torch.chunk(lab_i, num_chunk, dim=0)

            Sample_list.append(Sample)
            Label_list.append(Label)

        return Sample_list, Label_list