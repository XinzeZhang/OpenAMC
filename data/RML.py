import __main__

from data.Loader import TaskDataset
import pickle
import numpy as np
import torch
import h5py
import pandas as pd
import scipy.io as scio

class RML2016_10a_Data(TaskDataset):
    '''O’shea, T.J., West, N., 2016. Radio machine learning dataset generation with GNU radio, in: Proceedings of the GNU Radio Conference.
    '''
    def rawdata_config(self) -> object:
        self.data_name = 'RML2016.10a'
        self.batch_size = 512
        self.sig_len = 128

        self.val_size = 0.2
        self.test_size = 0.2

        self.classes = {'QAM16': 0, 'QAM64': 1, '8PSK': 2, 'WBFM': 3, 'BPSK': 4,
                        'CPFSK': 5, 'AM-DSB': 6, 'GFSK': 7, 'PAM4': 8, 'QPSK': 9, 'AM-SSB': 10}
        self.snr_envs = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        self.snrs = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

        self.post_data_file = 'data/postdata/RML2016.10a_dict.split.pt'
        self.file_pointer = 'data/rawdata/RadioML/RML2016.10a_dict.pkl'

    def load_rawdata(self, logger=None):
        '''
        return all signals , all labels ,all snrs  ,unique snr list ,unique mod list
        '''
        if logger is not None:
            logger.info('*'*80 + '\n' +
                        f'Loading raw file in the location: {self.file_pointer}')

        Signals = []
        Labels = []
        SNRs = []

        Set = pickle.load(open(self.file_pointer, 'rb'), encoding='latin1')
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


class RML2022_01a_Data(RML2016_10a_Data):
    '''V. Sathyanarayanan, P. Gerstoft, and A. E. Gamal, “RML22: realistic dataset generation for wireless modulation classification,” Trans, Wirel., Comm, vol. 22, no. 11, pp. 7663–7675, 2023, doi: 10.1109/TWC.2023.3254490.
    '''
    def rawdata_config(self) -> object:
        self.data_name = 'RML2022.01a'
        self.batch_size = 512
        self.sig_len = 128

        self.val_size = 0.2
        self.test_size = 0.2

        self.classes = {'QAM16': 0, 'QAM64': 1, '8PSK': 2, 'WBFM': 3, 'BPSK': 4,
                        'CPFSK': 5, 'AM-DSB': 6, 'GFSK': 7, 'PAM4': 8, 'QPSK': 9, 'AM-SSB': 10}
        self.snr_envs = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        self.snrs = self.snr_envs

        self.post_data_file = 'data/postdata/RML2022.01a_dict.split.pt'
        self.file_pointer = 'data/rawdata/RadioML/RML22.01A.pkl'

class ACMR_Data(RML2016_10a_Data):
    """Tang, Z., Luo, C., Yin, Y., Luo, Y., 2026. ACMR: an automatic composite-modulation recognition dataset and baselines. IEEE Trans. Veh. Technol. 1–11. https://doi.org/10.1109/TVT.2026.3656598"""
    def rawdata_config(self) -> object:
        self.data_name = 'ACMR'
        self.batch_size = 512
        self.sig_len = 256

        self.val_size = 0.2
        self.test_size = 0.2

        # ['2FSK_PM', 'BPSK_FM', 'BPSK_PM', 'MSK_DSB', 'MSK_FM', 'OQPSK_FM', 'OQPSK_PM', 'PCM_DSB', 'PCM_FM', 'PCM_PM', 'QPSK_FM', 'QPSK_PM']
        self.classes = {'2FSK_PM': 0, 'BPSK_FM': 1, 'BPSK_PM': 2, 'MSK_DSB': 3, 'MSK_FM': 4, 'OQPSK_FM': 5, 'OQPSK_PM': 6, 'PCM_DSB': 7, 'PCM_FM': 8, 'PCM_PM': 9, 'QPSK_FM': 10, 'QPSK_PM': 11}
        self.snr_envs = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
        self.snrs = self.snr_envs

        self.post_data_file = 'data/postdata/ACMR_dict.split.pt'
        self.file_pointer = 'data/rawdata/RadioML/ACMR.pkl'


class RML2016_10b_Data(RML2016_10a_Data):
    '''O’shea, T.J., West, N., 2016. Radio machine learning dataset generation with GNU radio, in: Proceedings of the GNU Radio Conference.
    '''
    def rawdata_config(self) -> object:
        self.data_name = 'RML2016.10b'
        self.batch_size = 512
        self.sig_len = 128

        self.val_size = 0.2
        self.test_size = 0.2

        self.classes = {'QAM16': 0, 'QAM64': 1, '8PSK': 2, 'WBFM': 3, 'BPSK': 4, 'CPFSK': 5, 'AM-DSB': 6, 'GFSK': 7, 'PAM4': 8, 'QPSK': 9}
        self.snr_envs = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        self.snrs = self.snr_envs

        self.post_data_file = 'data/postdata/RML2016.10b_dict.split.pt'
        self.file_pointer = 'data/rawdata/RadioML/RML2016.10b.dat'


class RML2018_01a_Data(TaskDataset):
    '''O’Shea, T.J., Roy, T., Clancy, T.C., 2018. Over-the-air deep learning based radio signal classification. IEEE Journal of Selected Topics in Signal Processing 12, 168–179.
'''
    def rawdata_config(self) -> object:
        self.data_name = 'RML2018.01a'
        self.batch_size = 64
        self.sig_len = 1024

        self.val_size = 0.2
        self.test_size = 0.2

        self.num_classes = 24
        self.classes = {'00K': 0, '4ASK': 1, '8ASK': 2, 'BPSK': 3, 'QPSK': 4, '8PSK': 5, '16PSK': 6, '32PSK': 7, '16APSK': 8, '32APSK': 9,'64APSK': 10, '128APSK': 11, '16QAM': 12, '32QAM': 13, '64QAM': 14, '128QAM': 15, '256QAM': 16, 'AM-SSB-WC': 17, 'AM-SSB-SC': 18,'AM-DSB-WC': 19, 'AM-DSB-SC': 20, 'FM': 21, 'GMSK': 22, 'OQPSK': 23}
        self.post_data_file = 'data/postdata/RML2018.01a_dict.split.pt'

    def load_rawdata(self, logger = None):
        file_pointer = 'data/rawdata/RadioML/GOLD_XYZ_OSC.0001_1024.hdf5'

        if logger is not None:
            logger.info('*'*80 + '\n' +f'Loading raw file in the location: {file_pointer}')

        Signals, Labels, SNRs  = [], [], []

        f = h5py.File(file_pointer)
        Signals, Labels, SNRs  = f['X'][:], f['Y'][:], f['Z'][:]
        f.close()

        Signals = torch.from_numpy(Signals.astype(np.float32))
        Signals = Signals.permute(0, 2, 1)  # X:(2555904, 2, 1024)

        SNRs = SNRs.squeeze().tolist()
        snrs = list(np.unique(SNRs))
        mods = list(self.classes.keys())

        Labels = np.argwhere(Labels == 1)[:, 1]
        Labels = np.array(Labels, dtype=np.int64)
        Labels = torch.from_numpy(Labels)

        return Signals, Labels, SNRs, snrs, mods


class RML24_Data(TaskDataset):
    """Zhang, Y., Zang, B., Ji, H., Li, L., Li, S., Chen, L., 2026. Cognitive radio for satellite TT & C system: a general dataset using software-defined radio. Sci. Data. https://doi.org/10.1038/s41597-026-07182-7
"""
    def rawdata_config(self) -> object:
        self.data_name = 'RML24'
        self.batch_size = 64
        self.sig_len = 2048

        self.val_size = 0.2
        self.test_size = 0.2

        class_names=["BPSK","QPSK","8PSK","16QAM","32QAM","64QAM","GMSK","OQPSK","FQPSK","ARTM","SOQPSK","FM","PM", "16APSK","32APSK","BPSK_FM","BPSK_PM","QPSK_FM","QPSK_PM","FSK_PM","FQPSK_PM"] #"SOQPSK_PM" is not included in the dataset, so we only have 21 classes in total.
        self.num_classes = 21
        self.classes = {class_name: i for i, class_name in enumerate(class_names)}

        self.snr_envs = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        self.snrs = self.snr_envs

        self.post_data_file = f'data/postdata/{self.data_name}_dict.split.pt'

    def load_rawdata(self, logger = None):
        file_pointer = 'data/rawdata/RadioML/RML24_IQdata.h5'

        if logger is not None:
            logger.info('*'*80 + '\n' +f'Loading raw file in the location: {file_pointer}')

        Signals, Labels, SNRs  = [], [], []

        f = h5py.File(file_pointer)
        Signals, Labels, SNRs  = f['IQ_data'][:], f['class'][:], f['snr'][:]
        f.close()

        Signals = torch.from_numpy(Signals.astype(np.float32))
        Signals = Signals.permute(0, 2, 1)  # X:(2555904, 2, 1024)

        SNRs = SNRs.squeeze().tolist()
        snrs = list(np.unique(SNRs))
        mods = list(self.classes.keys())

        Labels = np.array(Labels, dtype=np.int64)
        Labels = torch.from_numpy(Labels)

        return Signals, Labels, SNRs, snrs, mods

    def dataset_Split(self, Signals, Labels, snrs, mods, val_size=0.2, test_size=0.2):
        snr_values = np.asarray(self.SNR_values)
        if snr_values.shape[0] != Signals.shape[0]:
            raise ValueError(
                f'SNR value count {snr_values.shape[0]} does not match signal count {Signals.shape[0]}.'
            )

        label_values = Labels.detach().cpu().numpy() if isinstance(Labels, torch.Tensor) else np.asarray(Labels)

        group_indices = []
        for label_idx in sorted(np.unique(label_values).tolist()):
            for snr in snrs:
                indices = np.flatnonzero((label_values == label_idx) & (snr_values == snr))
                if indices.size > 0:
                    group_indices.append(indices)

        if not group_indices:
            raise ValueError('No non-empty (label, snr) groups were found for dataset splitting.')

        common_test_count = min(int(np.floor(indices.size * test_size)) for indices in group_indices)

        train_idx = []
        val_idx = []
        test_idx = []

        for indices in group_indices:
            shuffled_indices = np.random.permutation(indices)

            test_count = min(common_test_count, shuffled_indices.size)
            val_count = min(int(np.floor(shuffled_indices.size * val_size)), shuffled_indices.size - test_count)

            test_idx.append(shuffled_indices[:test_count])
            val_idx.append(shuffled_indices[test_count:test_count + val_count])
            train_idx.append(shuffled_indices[test_count + val_count:])

        train_idx = np.concatenate(train_idx).astype(np.int64)
        val_idx = np.concatenate(val_idx).astype(np.int64)
        test_idx = np.concatenate(test_idx).astype(np.int64)

        Signals_train = Signals[train_idx]
        Labels_train = Labels[train_idx]

        Signals_val = Signals[val_idx]
        Labels_val = Labels[val_idx]

        Signals_test = Signals[test_idx]
        Labels_test = Labels[test_idx]

        return (Signals_train, Labels_train), \
            (Signals_val, Labels_val), \
            (Signals_test, Labels_test), \
            train_idx, val_idx, test_idx

class RML2016_04c_Data(RML2016_10a_Data):
    """O’shea, T.J., West, N., 2016. Radio machine learning dataset generation with GNU radio, in: Proceedings of the GNU Radio Conference.\n
    11个类 snr从-20到18有20个
    数据总量162060， 每个snr有8103个数据20*8103=162060。
    每个类的数据量是不相同的，([  8260, 14100, 14100, 24940, 24940, 24940, 12440,  6200,  4120,12440, 15580])  加起来162060
    """
    def rawdata_config(self) -> object:
        self.data_name = 'RML2016.04c'
        self.batch_size = 64
        self.sig_len = 128

        self.val_size = 0.2
        self.test_size = 0.2

        self.classes = {'8PSK':0, 'AM-DSB':1, 'AM-SSB':2, 'BPSK':3, 'CPFSK':4,
                        'GFSK':5, 'PAM4':6, 'QAM16':7, 'QAM64':8, 'QPSK':9, 'WBFM':10}
        self.snr_envs = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        self.snrs = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

        self.post_data_file = 'data/postdata/RML2016.04c_dict.split.pt'
        self.file_pointer = 'data/rawdata/RadioML/RML2016.04C.multisnr.pkl'


    @staticmethod
    def dataset_Split(Signals, Labels, snrs, mods, val_size=0.2, test_size=0.2):
        '''
        Split the data into train, validation, and test sets
        ensuring the same number of samples for each (mod, snr) in each set
        '''
        global test_idx
        n_examples = Signals.shape[0]
        train_size =1 - val_size - test_size

        train_idx = []
        test_idx = []
        val_idx = []

        unique_vals, counts = torch.unique(Labels, return_counts=True)
        Slices_list= [0]
        tmpsum = 0
        for i in range(0,len(unique_vals)):
            count_per_snr = int( counts[i]/len(snrs) )
            for j in range(0, len(snrs) ):
                tmpsum += count_per_snr
                Slices_list.append(tmpsum)
        # 确保最后一个值等于总样本数
        if Slices_list[-1] != len(Labels):
            Slices_list[-1] = len(Labels)
        Slices_list = np.array(Slices_list, dtype=np.int64)

        for k in range(0, Slices_list.shape[0] - 1):
            train_idx_subset = np.random.choice(
                range(int(Slices_list[k]), int(Slices_list[k + 1])), size= int((int(Slices_list[k + 1])-int(Slices_list[k]))*train_size) ,
                replace=False)
            Test_Val_idx_subset = list(
                set(range(int(Slices_list[k]), int(Slices_list[k + 1]))) - set(train_idx_subset))
            test_idx_subset = np.random.choice(Test_Val_idx_subset,
                                               size=int((int(Slices_list[k + 1])-int(Slices_list[k]))*test_size),
                                               replace=False)
            val_idx_subset = list(
                set(Test_Val_idx_subset) - set(test_idx_subset))

            train_idx = np.hstack([train_idx, train_idx_subset])
            val_idx = np.hstack([val_idx, val_idx_subset])
            test_idx = np.hstack([test_idx, test_idx_subset])
            # print(k,train_idx_subset.shape,len(val_idx_subset),test_idx_subset.shape)
            # print(train_idx.shape, val_idx.shape, test_idx.shape)
            #if((k+1)%20==0):
            #    print('tttt')


        train_idx = train_idx.astype('int64')
        val_idx = val_idx.astype('int64')
        test_idx = test_idx.astype('int64')
        #print(train_idx.shape,val_idx.shape,test_idx.shape)
        #print(Labels[train_idx].shape)
        #input('xxx')
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


class MIMO_Nt4Nr2_Data(TaskDataset):
    '''Zhang, F., Luo, C., Xu, J., Luo, Y., Zheng, F.-C., 2022. Deep learning based automatic modulation recognition: models, datasets, and challenges. Digital Signal Processing 129, 103650. https://doi.org/10.1016/j.dsp.2022.103650
    \n
    https://github.com/Richardzhangxx/AMR-Dataset-for-MIMO-system-with-precoding/tree/main
    \n
    Data config python file for MIMO_Nt4Nr2_Data dataset
    Nt4Nr2  Nt16Nr4  Nt64Nr16 三个文件夹 代表3种采集方式 每个文件夹下的数据量格式都相同
    每个文件夹下：
    单个类的数据量15500x2x128  6个类 31个snr (-10到20) 每个类下面的每个snr有500个数据  6x31x500=93000 =15500x6
    6个类的总数据量是93000x2x128
    注意，它这里lab.mat文件里取出来的lab值是类似于[1,0,0,0,0,0]的hot编码，不是单独的1，2，3，4，5，6这种
    '''
    def rawdata_config(self) -> object:
        self.data_name = 'MIMO.Nt4Nr2'
        self.batch_size = 512
        self.sig_len = 128

        self.val_size = 0.2
        self.test_size = 0.2

        self.classes = {'2PSK':0, 'QPSK':1, '8PSK':2, '16QAM':3, '64QAM':4, '128QAM':5}
        self.snr_envs = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        self.snrs = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

        self.post_data_file = 'data/postdata/MIMO.Nt4Nr2_dict.split.pt'

    def load_rawdata(self, logger=None):
        '''
        return all signals , all labels ,all snrs  ,unique snr list ,unique mod list
        '''
        folder = 'data/rawdata/MIMO/Nt4Nr2'
        tags=['2psk', 'qpsk', '8psk', '16qam', '64qam', '128qam']
        if logger is not None:
            logger.info('*'*80 + '\n' +
                        f'Loading raw file in the location: {folder}')
        signal_dict = {}
        for tag in tags:
            dataFile = folder + '/data' + tag + '.mat'
            data = scio.loadmat(dataFile)['data_save']
            labFile =  folder + '/label' + tag + '.mat'
            lab = scio.loadmat(labFile)['label_save']
            snrFile = folder + '/snr' + tag + '.mat'
            snr = scio.loadmat(snrFile)['snr_save']
            length = data.shape[0]
            for i in range(length):
                key = (tag.upper(), snr[i].item())
                if key not in signal_dict:
                    signal_dict[key] = []
                signal_dict[key].append(data[i])
        for key in signal_dict:
            signal_dict[key] = np.array(signal_dict[key])
        print(f"\n分组完成,共 {len(signal_dict)} 个组合")
        # 验证每个组合的数据量
        for key, data in signal_dict.items():
            if len(data) != 500:
                print(f"警告: {key} 只有 {len(data)} 个样本，不是500")

        Signals = []
        Labels = []
        SNRs = []
        Set = signal_dict

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

class MIMO_Nt16Nr4_Data(TaskDataset):
    '''Data config python file for MIMO_Nt16Nr4_Data dataset
    Nt4Nr2  Nt16Nr4  Nt64Nr16 三个文件夹 代表3种采集方式 每个文件夹下的数据量格式都相同
    每个文件夹下：
    单个类的数据量15500x2x128  6个类 31个snr (-10到20) 每个类下面的每个snr有500个数据  6x31x500=93000 =15500x6
    6个类的总数据量是93000x2x128
    注意，它这里lab.mat文件里取出来的lab值是类似于[1,0,0,0,0,0]的hot编码，不是单独的1，2，3，4，5，6这种
    '''
    def rawdata_config(self) -> object:
        self.data_name = 'MIMO.Nt16Nr4'
        self.batch_size = 512
        self.sig_len = 128

        self.val_size = 0.2
        self.test_size = 0.2

        self.classes = {'2PSK':0, 'QPSK':1, '8PSK':2, '16QAM':3, '64QAM':4, '128QAM':5}
        self.snr_envs = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        self.snrs = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

        self.post_data_file = 'data/postdata/MIMO.Nt16Nr4_dict.split.pt'

    def load_rawdata(self, logger=None):
        '''
        return all signals , all labels ,all snrs  ,unique snr list ,unique mod list
        '''
        folder = 'data/rawdata/MIMO/Nt16Nr4'
        tags=['2psk', 'qpsk', '8psk', '16qam', '64qam', '128qam']
        if logger is not None:
            logger.info('*'*80 + '\n' +
                        f'Loading raw file in the location: {folder}')
        signal_dict = {}
        for tag in tags:
            dataFile = folder + '/data' + tag + '.mat'
            data = scio.loadmat(dataFile)['data_save']
            labFile =  folder + '/label' + tag + '.mat'
            lab = scio.loadmat(labFile)['label_save']
            snrFile = folder + '/snr' + tag + '.mat'
            snr = scio.loadmat(snrFile)['snr_save']
            length = data.shape[0]
            for i in range(length):
                key = (tag.upper(), snr[i].item())
                if key not in signal_dict:
                    signal_dict[key] = []
                signal_dict[key].append(data[i])
        for key in signal_dict:
            signal_dict[key] = np.array(signal_dict[key])
        print(f"\n分组完成,共 {len(signal_dict)} 个组合")
        # 验证每个组合的数据量
        for key, data in signal_dict.items():
            if len(data) != 500:
                print(f"警告: {key} 只有 {len(data)} 个样本，不是500")

        Signals = []
        Labels = []
        SNRs = []
        Set = signal_dict

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

class MIMO_Nt64Nr16_Data(TaskDataset):
    '''Data config python file for MIMO_Nt64Nr16_Data dataset
    Nt4Nr2  Nt16Nr4  Nt64Nr16 三个文件夹 代表3种采集方式 每个文件夹下的数据量格式都相同
    每个文件夹下：
    单个类的数据量15500x2x128  6个类 31个snr (-10到20) 每个类下面的每个snr有500个数据  6x31x500=93000 =15500x6
    6个类的总数据量是93000x2x128
    注意，它这里lab.mat文件里取出来的lab值是类似于[1,0,0,0,0,0]的hot编码，不是单独的1，2，3，4，5，6这种
    '''
    def rawdata_config(self) -> object:
        self.data_name = 'MIMO.Nt64Nr16'
        self.batch_size = 512
        self.sig_len = 128

        self.val_size = 0.2
        self.test_size = 0.2

        self.classes = {'2PSK':0, 'QPSK':1, '8PSK':2, '16QAM':3, '64QAM':4, '128QAM':5}
        self.snr_envs = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        self.snrs = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

        self.post_data_file = 'data/postdata/MIMO.Nt64Nr16_dict.split.pt'

    def load_rawdata(self, logger=None):
        '''
        return all signals , all labels ,all snrs  ,unique snr list ,unique mod list
        '''
        folder = 'data/rawdata/MIMO/Nt64Nr16'
        tags=['2psk', 'qpsk', '8psk', '16qam', '64qam', '128qam']
        if logger is not None:
            logger.info('*'*80 + '\n' +
                        f'Loading raw file in the location: {folder}')
        signal_dict = {}
        for tag in tags:
            dataFile = folder + '/data' + tag + '.mat'
            data = scio.loadmat(dataFile)['data_save']
            labFile =  folder + '/label' + tag + '.mat'
            lab = scio.loadmat(labFile)['label_save']
            snrFile = folder + '/snr' + tag + '.mat'
            snr = scio.loadmat(snrFile)['snr_save']
            length = data.shape[0]
            for i in range(length):
                key = (tag.upper(), snr[i].item())
                if key not in signal_dict:
                    signal_dict[key] = []
                signal_dict[key].append(data[i])
        for key in signal_dict:
            signal_dict[key] = np.array(signal_dict[key])
        print(f"\n分组完成,共 {len(signal_dict)} 个组合")
        # 验证每个组合的数据量
        for key, data in signal_dict.items():
            if len(data) != 500:
                print(f"警告: {key} 只有 {len(data)} 个样本，不是500")

        Signals = []
        Labels = []
        SNRs = []
        Set = signal_dict

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



class Panoradio_HF_Data(TaskDataset):
    '''Scholl, S., 2019. Classification of radio signals and HF transmission modes with deep learning. CoRR abs/1906.4459.
    \n
    dataset_panoradio_hf.npy 一共172800个数据，每个数据2048长，里面的值是复数，所以最终的数据size是172800x2x2048
    18个类 8个snr  每个类的每个snr是1200个数据
    dataset_panoradio_hf_tags.csv
    "idx	 mode	 snr
    0	morse	25
    1	morse	25
    ...
    199	morse	25
    200	morse	20
    ...
    172798	fax	-10
    172799	fax	-10
    注意，它不是每个类直接按每个snr 1200这么放，而是每个snr放200，6次放完，再去下一个类。

    '''
    def rawdata_config(self) -> object:
        self.data_name = 'Panoradio.HF'
        self.batch_size = 128
        self.sig_len = 2048

        self.val_size = 0.2
        self.test_size = 0.2

        self.classes = {"MORSE": 0,"PSK31": 1,"PSK63": 2,"QPSK31": 3,"RTTY45_170": 4,"RTTY50_170": 5,
                "RTTY100_850": 6,"OLIVIA8_250": 7,"OLIVIA16_500": 8,"OLIVIA16_1000": 9,"OLIVIA32_1000": 10,
                "DOMINOEX11": 11,"MT63_1000": 12,"NAVTEX": 13,"USB": 14,"LSB": 15,"AM": 16,"FAX": 17      }
        self.snr_envs = [-10, -5, 0, 5, 10, 15, 20, 25]
        self.snrs     = [-10, -5, 0, 5, 10, 15, 20, 25]
        self.post_data_file = 'data/postdata/Panoradio.HF_dict.split.pt'

    def load_rawdata(self, logger=None):
        '''
        return all signals , all labels ,all snrs  ,unique snr list ,unique mod list
        '''
        #file_pointer = 'data/rawdata/Panoradio_HF/dataset_panoradio_hf.npy'
        #######
        np_file_pointer = 'data/rawdata/Panoradio_HF/dataset_panoradio_hf.npy'
        csv_file_pointer = 'data/rawdata/Panoradio_HF/dataset_panoradio_hf_tags.csv'
        signals = np.load(np_file_pointer, allow_pickle=True)
        # print(signals.shape)
        df = pd.read_csv(csv_file_pointer, skipinitialspace=True)
        # print(df.shape)
        # 2048里的每个数据是复数形式 形如np.complex128(-0.9683167539506656-0.7720366346859253j)
        I_channel = np.real(signals)
        Q_channel = np.imag(signals)
        signals_reshaped = np.stack([I_channel, Q_channel], axis=1)
        # print(signals_reshaped.shape)
        #按(mode, snr)分组数据
        signal_dict = {}
        # 遍历每一行数据
        for idx, row in df.iterrows():
            mode = row['mode'].upper()
            snr = row['snr']
            key = (mode, snr)  # 使用元组作为键

            # 获取对应的信号数据
            signal_sample = signals_reshaped[idx]  # 形状: (2, 1024)

            # 如果是第一次遇到这个组合，创建列表
            if key not in signal_dict:
                signal_dict[key] = []

            # 添加到对应组合的列表中
            signal_dict[key].append(signal_sample)
            # 将列表转换为numpy数组 (1200, 2, 1024)
        for key in signal_dict:
            signal_dict[key] = np.array(signal_dict[key])  # 形状: (1200, 2, 1024)
        print(f"\n分组完成,共 {len(signal_dict)} 个组合")
        # 验证每个组合的数据量
        for key, data in signal_dict.items():
            if len(data) != 1200:
                print(f"警告: {key} 只有 {len(data)} 个样本，不是1200")
        #######
        if logger is not None:
            logger.info('*'*80 + '\n' +
                        f'Loading raw file in the location: {np_file_pointer} {csv_file_pointer}')

        Signals = []
        Labels = []
        SNRs = []

        Set = signal_dict
        #Set = pickle.load(open(file_pointer, 'rb'), encoding='latin1')
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


class HisarMod2019_1_Data(TaskDataset):
    '''Tekbiyik, K., Ekti, A.R., Görçin, A., Karabulut-Kurt, G., Keçeci, C., 2020. Robust and fast automatic modulation classification with CNN under multipath fading channels, in: 91st IEEE Vehicular Technology Conference, VTC Spring 2020, Antwerp, Belgium, May 25-28, 2020. IEEE, pp. 1–6. https://doi.org/10.1109/VTC2020-Spring48590.2020.9128408
    \n
    分Train和Test两个文件夹。这里是统一收集后用我们自己的6：2：2再生成train、val、test集。
    原始的Train： 1024x2x520000  26个类 20种snr（-20到18） 每个类20000个数据 每个snr26000个数据  每个类下每个snr有1000个数据
    原始的Test： 1024x2x260000   26个类 20种snr（-20到18） 每个类10000个数据 每个snr13000个数据  每个类下每个snr有500个数据
    合在一起后，总共78万数据，26个类，20种snr，每个类30000，每个snr39000，每个类下每个snr1500。

    注意.csv文件第一行就是数据，不是表头，读取的时候要加参数避免。
    还有data.csv处理起来太慢太复杂（值是a+bi的str型），所以这里处理用了网上提供的.mat文件。

    '''
    def rawdata_config(self) -> object:
        self.data_name = 'HisarMod2019.1'
        self.batch_size = 256
        self.sig_len = 1024

        self.val_size = 0.2
        self.test_size = 0.2

        self.classes = {
        "BPSK":0,
        "QPSK":1,
        "8PSK":2,
        "16PSK":3,
        "32PSK":4,
        "64PSK":5,
        "4QAM":6,
        "8QAM":7,
        "16QAM":8,
        "32QAM":9,
        "64QAM":10,
        "128QAM":11,
        "256QAM":12,
        "2FSK":13,
        "4FSK":14,
        "8FSK":15,
        "16FSK":16,
        "4PAM":17,
        "8PAM":18,
        "16PAM":19,
        "AM-DSB":20,
        "AM-DSB-SC":21,
        "AM-USB":22,
        "AM-LSB":23,
        "FM":24,
        "PM":25
}

        self.snr_envs = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        self.snrs = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

        self.post_data_file = 'data/postdata/HisarMod2019.1_dict.split.pt'

    def load_rawdata(self, logger=None):
        '''
        return all signals , all labels ,all snrs  ,unique snr list ,unique mod list
        '''
        tags = {
        0 : "BPSK",
        10: "QPSK",
        20: "8PSK",
        30: "16PSK",
        40: "32PSK",
        50: "64PSK",
        1 : "4QAM",
        11: "8QAM",
        21: "16QAM",
        31: "32QAM",
        41: "64QAM",
        51: "128QAM",
        61: "256QAM",
        2 : "2FSK",
        12: "4FSK",
        22: "8FSK",
        32: "16FSK",
        3 : "4PAM",
        13: "8PAM",
        23: "16PAM",
        4 : "AM-DSB",
        14: "AM-DSB-SC",
        24: "AM-USB",
        34: "AM-LSB",
        44: "FM",
        54: "PM"
}
        if logger is not None:
            logger.info('*'*80 + '\n' +
                        f'Loading raw file in the location: train.mat test.mat')
        signal_dict ={}
        data = h5py.File("data/rawdata/HisarMod2019.1/Train/train.mat")
        train_data = data['data_save'][:].swapaxes(0,2)
        print(train_data.shape)
        print(type(train_data))

        data = h5py.File("data/rawdata/HisarMod2019.1/Test/test.mat")
        test_data = data['data_save'][:].swapaxes(0,2)
        print(test_data.shape)
        print(type(test_data))

        train_labels = pd.read_csv('data/rawdata/HisarMod2019.1/Train/train_labels.csv',header=None)
        train_labels=np.array(train_labels)
        test_labels = pd.read_csv('data/rawdata/HisarMod2019.1/Test/test_labels.csv',header=None)
        test_labels =np.array(test_labels)

        train_snr=pd.read_csv('data/rawdata/HisarMod2019.1/Train/train_snr.csv',header=None)
        train_snr=np.array(train_snr)

        test_snr=pd.read_csv('data/rawdata/HisarMod2019.1/Test/test_snr.csv',header=None)
        test_snr=np.array(test_snr)

        trainlen = train_data.shape[0]
        testlen = test_data.shape[0]

        for  i in range(trainlen):
            key = ( tags[train_labels[i].item()], train_snr[i].item())
            if key not in signal_dict:
                signal_dict[key] = []
            signal_dict[key].append(train_data[i])
        '''
        for key in signal_dict:
            signal_dict[key] = np.array(signal_dict[key])
        print(f"\n分组完成,共 {len(signal_dict)} 个组合")
        # 验证每个组合的数据量
        for key, data in signal_dict.items():
            if len(data) != 1000:
                print(f"警告: {key} 只有 {len(data)} 1000")
        '''
        #signal_dict ={}
        for  i in range(testlen):
            key = ( tags[test_labels[i].item()], test_snr[i].item())
            if key not in signal_dict:
                signal_dict[key] = []
            signal_dict[key].append(test_data[i])
        for key in signal_dict:
            signal_dict[key] = np.array(signal_dict[key])
        print(f"\n分组完成,共 {len(signal_dict)} 个组合")
        # 验证每个组合的数据量
        for key, data in signal_dict.items():
            if len(data) != 1500:
                print(f"警告: {key} 只有 {len(data)} 1500")


        Signals = []
        Labels = []
        SNRs = []

        Set = signal_dict
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

if __name__ == '__main__':
    pass
