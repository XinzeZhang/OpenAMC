import os
import sys

# from sklearn.utils import gen_even_slices
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
import logging
from pathlib import Path
import shutil
from tqdm.auto import tqdm

import random

import numpy as np
import torch

import torch.utils.data as Data
from collections.abc import Mapping
import copy
from math import ceil


from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns




def logit_acc(logit, lab_batch):
    '''
    forward, calculate acc with logit
    '''
    pre_lab = torch.argmax(logit, 1)
    acc = torch.sum(pre_lab == lab_batch.data).double(
    ).item() / lab_batch.size(0)
    return acc

def check_mem(gid):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(gid)].split(',')
    return int(total),int(used)

def lock_mem(gid, ocm):
    cuda_exist = torch.cuda.is_available()
    if ocm > 0 and cuda_exist:
        if ocm >= 0.8:
            ocm = 0.8
        torch.cuda.empty_cache()
        total, used = check_mem(gid)
        block_mem = int((total - used) * ocm)
        with torch.no_grad():
            x = torch.ones((256,1024,block_mem)).cuda(device=torch.device('cuda:{}'.format(gid)))
            del x
            print('Total GPU Memory: {} MB; Used {} MB; OCM: {}; Reserved: {} MB'.format(total, used, ocm, block_mem))

def snr_slice(data_set, set_idx, SNRs = None, snr = 0):
    '''
    find the signals and labels where the SNR equals the specified value
    '''
    if len(data_set)==2 :
        Signals, Labels = data_set
    if len(data_set)==3 :
        Signals, Labels, _= data_set

    data_SNRs = map(lambda x: SNRs[x], set_idx)
    data_SNRs = list(data_SNRs)
    data_SNRs = np.array(data_SNRs).squeeze()
    idx_i = np.where(np.array(data_SNRs) == snr)
    sig_i = Signals[idx_i]
    lab_i = Labels[idx_i]

    # idx_i = idx_i[0]

    return sig_i, lab_i, idx_i

def fix_seed(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYHONHASHSEED'] = str(seed)
    np.random.seed(seed) # type: ignore
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False # type: ignore
    torch.backends.cudnn.deterministic = True # type: ignore

class temporary_seed:
    def __init__(self, seed):
        self.seed = seed
        self.backup = None

    def __enter__(self):
        self.backup = np.random.randint(2**32-1, dtype=np.uint32)
        fix_seed(self.seed)
        # np.random.seed(self.seed)

    def __exit__(self, *_):
        fix_seed(self.backup)
        # np.random.seed(self.backup)

def os_makedirs(folder_path):
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        return folder_path
    except FileExistsError:
        pass

def os_rmdirs(folder_path):
    try:
        dirPath = Path(folder_path)
        if dirPath.exists() and dirPath.is_dir():
            shutil.rmtree(dirPath)
    except FileExistsError:
        pass

def set_logger(log_path, log_name, level = 20, rewrite = True):
    '''Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `task_dir/train.log`.
    Example:
    logging.info('Starting training...')
    Args:
        log_path: (string) where to log
    '''
    rewrite = False
    logger = logging.Logger(log_name)
    logger.propagate = False
    logger.handlers.clear()
    if os.path.exists(log_path) and rewrite:
        os.remove(log_path) # os.remove can only delete a file with given file_path; os.rmdir() can delete a directory.
    log_file = Path(log_path)
    log_folder = log_file.parent
    os_makedirs(log_folder)
    log_file.touch(exist_ok=True)


    if level == 50:
        logger.setLevel(logging.CRITICAL)
    else:
        logger.setLevel(logging.INFO)

    fmt = logging.Formatter('[%(asctime)s] %(name)s: %(message)s', '%Y-%m-%d %H:%M:%S')

    class TqdmHandler(logging.StreamHandler):
        def __init__(self, formatter):
            logging.StreamHandler.__init__(self)
            self.setFormatter(formatter)

        def emit(self, record):
            msg = self.format(record)
            # Ensure logs are written on a clean line when tqdm/progress output is active.
            if record.levelno >= logging.ERROR:
                tqdm.write("")
            tqdm.write(msg, end="\n")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)# Log output to a file
    logger.addHandler(TqdmHandler(fmt)) # Log output to the terminal

    return logger

def set_dataloader(batch_size = None, num_workers = 0, data_set = None, shuffle = True):
    batch_size = data_set[0].size(0) if batch_size is None else min(batch_size, data_set[0].size(0))
    data = Data.TensorDataset(*data_set)
    data_loader = Data.DataLoader(
        dataset=data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return data_loader

def set_fitset(batch_size = 64, num_workers = 0, train_set = None, val_set = None):
    train_data = Data.TensorDataset(*train_set)
    val_data = Data.TensorDataset(*val_set)

    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = Data.DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, val_loader

class Opt(object):
    def __init__(self, init=None):
        super().__init__()

        if init is not None:
            self.merge(init)

    def merge(self, opts, ele_s=None):
        '''
        Only merge the key-value not in the current Opt.\n
        Using ele_s to select the element in opts to be merged.
        '''
        added_dict  = {}
        if isinstance(opts, Mapping):
            new = opts
        else:
            assert isinstance(opts, object)
            new = vars(opts)

        if ele_s is None:
            for key in new:
                if not key in self.dict:
                    self.copy_value(new,key)
                    added_dict[key] = new[key]
        else:
            for key in ele_s:
                if not key in self.dict:
                    self.copy_value(new,key)
                    added_dict[key] = new[key]

        return added_dict

    def update(self, opts, check_subset=True, only_interset = True):
        '''Update key-value for key in opts if key in self with only_interset = True.\n
        Using check_subset = True to check opts is subset of self.
        '''
        updated_dict = {}
        if isinstance(opts, Mapping):
            new = opts
        else:
            assert isinstance(opts, object)
            new = vars(opts)
        for key in new:
            if only_interset:
                if key in self.dict and self.dict[key] != new[key]:
                    self.copy_value(new,key)
                    updated_dict[key] = new[key]
            else:
                if check_subset:
                    if  key not in self.dict:
                        raise ValueError(
                            "Unknown config key '{}'".format(key))
                else:
                    self.copy_value(new,key)
                    updated_dict[key] = new[key]

        return updated_dict


    def copy_value(self, new, key):
        v = new[key]
        if isinstance(v, logging.Logger):
            self.dict[key] = v
        else:
            self.dict[key] = copy.copy(v)

    @property
    def dict(self):
        '''Gives dict-like access to Params instance by params.dict['learning_rate']'''
        return self.__dict__

def chunk_list_nsub(lst, n):
    size = ceil(len(lst) / n)
    return list(
        map(lambda x: lst[x * size:x * size + size],
        list(range(n)))
    )

def chunk_list_nsub_dict(num_classes, lst, logger):
    num_snrs = len(lst)
    if not  num_snrs % num_classes == 0:
        logger.warning(f'Unequal class with num_classes: {num_classes} for total {num_snrs} conditions.')

    class_dict = {}
    class_list = chunk_list_nsub(lst, num_classes)

    for i, period in enumerate(class_list):
        logger.info(f'{i}-th snr env class with snr conditions: {period}')

        for snr in period:
            class_dict[snr] = i
    return class_dict

def save_training_process(epochs_stats, plot_dir):
    if epochs_stats is not None and set(['val_loss','val_acc', 'train_loss', 'train_acc', 'lr_list']).issubset(epochs_stats.columns):
        os_makedirs(plot_dir)
        # fig1 = plt.figure(1)
        # plt.plot(epochs_stats.epoch, epochs_stats.lr_list)
        # plt.xlabel("epoch")
        # plt.ylabel("lr")
        # plt.title("learning rate")
        # plt.grid()
        # fig1.savefig(os.path.join(plot_dir, 'lr.png'), format='png', dpi=300)
        # plt.close()

        fig2 = plt.figure(figsize=(18, 4))
        plt.subplot(1, 3, 1)
        plt.plot(epochs_stats.epoch, epochs_stats.train_loss,
                "r-", label="Train loss")
        plt.plot(epochs_stats.epoch, epochs_stats.val_loss,
                "b-", label="Val loss")
        plt.legend()
        plt.grid()
        plt.xlabel("epoch")
        plt.ylabel("Loss")

        plt.subplot(1, 3, 2)
        plt.plot(epochs_stats.epoch, epochs_stats.train_acc,
                "r-", label="Train acc")
        plt.plot(epochs_stats.epoch, epochs_stats.val_acc,
                "b-", label="Val acc")
        plt.xlabel("epoch")
        plt.ylabel("acc")
        plt.legend()
        plt.grid()

        plt.subplot(1, 3, 3)
        plt.plot(epochs_stats.epoch, epochs_stats.lr_list, label = 'Learing rate')
        plt.xlabel("epoch")
        plt.ylabel("lr")
        plt.legend()
        plt.grid()

        fig2.savefig(os.path.join(plot_dir,'loss_acc.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

def save_confmat(Confmat_Set, num_snrs, classes, plot_dir ):
    '''
    for each snr value , draw confusion matrix heatmap and save
    '''
    os_makedirs(plot_dir)
    for i, snr in enumerate(num_snrs):
        fig = plt.figure()
        df_cm = pd.DataFrame(Confmat_Set[i],
                             index=classes,
                             columns=classes)
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
        heatmap.yaxis.set_ticklabels(
            heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
        heatmap.xaxis.set_ticklabels(
            heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        conf_mat_dir = os.path.join(plot_dir, 'conf_mat')
        os.makedirs(conf_mat_dir, exist_ok=True)
        fig.savefig(conf_mat_dir + '/' + f'ConfMat_{snr}dB.png', dpi=300, bbox_inches='tight')
        plt.close()

def save_snr_acc(Accuracy_list, Confmat_Set, num_snrs, data_name, class_names, plot_dir):
    '''
    Plot the accuracy-SNR curve overall
    Plot the accuracy-SNR curve for each mod
    return accuracy each mod for each snr : Accuracy_Mods  (num_snrs,num_mods)
    '''
    plt.plot(num_snrs, Accuracy_list)
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Overall Accuracy")
    plt.title(f"Overall Accuracy on {data_name} dataset")
    plt.yticks(np.linspace(0, 1, 11))
    plt.grid()
    # acc_dir = os.path.join(plot_dir, 'acc')
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(plot_dir + '/' + 'snr_acc.png', dpi=300)
    plt.close()

    Accuracy_Mods = np.zeros((len(num_snrs), Confmat_Set.shape[-1]))

    for i, snr in enumerate(num_snrs):
        Accuracy_Mods[i, :] = np.diagonal(Confmat_Set[i]) / Confmat_Set[i].sum(1)

    for j in range(0, Confmat_Set.shape[-1]):
        plt.plot(num_snrs, Accuracy_Mods[:, j])

    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Overall Accuracy")
    plt.title(f"Overall Accuracy on {data_name} dataset")
    plt.grid()
    plt.legend(class_names)
    plt.savefig(plot_dir + '/' + 'acc_mods.png', dpi=300)
    plt.close()

    return Accuracy_Mods


if __name__ == "__main__":
    test_log_path = os.path.join("yield_test", "logger_newline_test.log")
    logger = set_logger(test_log_path, log_name="logger-newline-test", rewrite=True)

    logger.info("Start logger newline test")

    # Simulate progress updates that commonly cause same-line rendering issues.
    for step in tqdm(range(5), desc="progress"):
        if step == 2:
            logger.info("Info during tqdm progress")
        if step == 3:
            raise RuntimeError("Intentional test exception during tqdm progress")

    logger.error("This ERROR should appear on a fresh new line in terminal")

    try:
        raise RuntimeError("Intentional test exception for logger formatting")
    except RuntimeError:
        logger.exception("This EXCEPTION traceback should also start on a new line")

    logger.info("Logger newline test finished")