from data.Loader import TaskDataset
import torch
import numpy as np
from taskRecog.util import chunk_list_nsub_dict
from copy import deepcopy

class SnrDataSet(TaskDataset):
    def __init__(self, args=None):
        super().__init__(args)

        self.num_classes = 20 if 'num_classes' not in args.__dict__ else args.num_classes
        self.info.num_classes = self.num_classes

    def target_replace(self, dataset, data_idx):
        Signals, Mods_labels =dataset

        # New_Labels = torch.empty_like(Labels)

        Cor_SNRs = map(lambda x: self.SNR_values[x], data_idx)
        Cor_SNRs = list(Cor_SNRs)

        New_Labels = [self.classes[snr] for snr in Cor_SNRs]
        New_Labels = np.array(New_Labels, dtype=np.int64)
        New_Labels = torch.from_numpy(New_Labels)
        return (Signals, New_Labels), Mods_labels


    def load_testset(self, test_batch_size = 64):

        Sample_list = []
        Label_list = []
        self.num_mods = len(self.mods)
        # self.test_mods
        Samples, Labels = self.test_set
        for i in range(self.num_mods):
            idx = torch.where(self.test_mods == i)
            sig_i = Samples[idx]
            lab_i = Labels[idx]

            num_chunk = int(sig_i.shape[0] / test_batch_size)

            Sample = torch.chunk(sig_i, num_chunk, dim=0)
            Label = torch.chunk(lab_i, num_chunk, dim=0)

            Sample_list.append(Sample)
            Label_list.append(Label)

        # if 'num_snrs' not in self.dict:
        #     self.num_snrs = list(np.unique(self.snr_envs))
        # for snr in self.num_snrs:
        #     sig_i, lab_i, _ = self.snr_slice('test', snr)

        #     num_chunk = int(sig_i.shape[0] / test_batch_size)

        #     Sample = torch.chunk(sig_i, num_chunk, dim=0)
        #     Label = torch.chunk(lab_i, num_chunk, dim=0)

        #     Sample_list.append(Sample)
        #     Label_list.append(Label)

        return Sample_list, Label_list


    def pack_dataset(self, logger=None):
        super().pack_dataset(logger)

        self.mods_dict = deepcopy(self.classes)
        self.mods_dict = {v:k for k,v in self.mods_dict.items()}

        self.classes = chunk_list_nsub_dict(self.num_classes, self.snr_envs, logger)

        self.train_set, _ = self.target_replace(self.train_set, self.train_idx)
        self.val_set, _ = self.target_replace(self.val_set, self.val_idx)
        self.test_set, self.test_mods = self.target_replace(self.test_set, self.test_idx)
