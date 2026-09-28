import os
import sys
import importlib


import torch
import numpy as np
from collections import Counter
from tqdm.auto import tqdm, trange
from taskRecog.util import Opt, set_dataloader
from taskRecog.Wrapper import Task
from data.Loader import TaskDataset
from taskRecog.Parser import get_parser

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, cohen_kappa_score, mean_squared_error
from taskRecog.util import os_makedirs, os_rmdirs, set_logger, fix_seed
from taskRecog.util import save_training_process, save_confmat, save_snr_acc
from copy import deepcopy

# pip install skfeature-chappers
from skfeature.function.sparse_learning_based import RFS
from skfeature.function.similarity_based.fisher_score import fisher_score

from skfeature.utility import construct_W
from skfeature.function.similarity_based.lap_score import lap_score

# need replicate wrapperholistic

class sampler(Opt):
    def __init__(self, init=None):
        super().__init__(init)
        self.num_steps = 64
        self.local_dir = ''
        self.name = ''

def ranker_snr_dict(step_acc_list, num_steps):
    _step_acc_list = deepcopy(step_acc_list)
    _step_acc_list.sort(key=lambda x: x[1])
    _step_acc_list = _step_acc_list[:num_steps]
    ranker_step_dict = {k: v for k, v in _step_acc_list}
    return ranker_step_dict

def ranker_check_load(key, hyper, num_steps, logger):
    snrs_tier = []

    if key not in hyper.dict:
        raise ValueError(f'Non setting of {key}, please config it in model_opts.hyper.')
    else:
        ranker_save_path = hyper.dict[key]
        try:
            ranker_step_acc = torch.load(ranker_save_path)

            for step_acc_list in ranker_step_acc:
                ranker_step_dict = ranker_snr_dict(step_acc_list, num_steps)
                snrs_tier.append(ranker_step_dict)

            logger.info('-'*80)
            logger.info(f'Read the saved ranker_step_acc from {ranker_save_path} successfully!')

        except:
            logger.exception(f'Error when loading {ranker_save_path}')
            raise SystemExit()

    return snrs_tier

def SN_ranker(model, Samples, Label):
    step_acc_list,pred_label = [], []
    sig_len = Samples[0].size(2)
    for idx in trange(sig_len, colour='green', leave=False):
        pred_i = []
        label_i = []
        for (sample, label) in zip(Samples, Label):
            sample = sample.to(model.hyper.device)
            snr_sample_copy = sample.detach().clone()
            # snr_sample_copy[:,:, idx] = torch.mean(snr_sample_copy[:,:, idx]) #  = torch.mean() is the settings as https://codeocean.com/capsule/8397297/tree/v1
            snr_sample_copy[:,:, idx] = 0 #  = 0 is the settings as https://github.com/dl4amc/dds
            pre_lab = model.predict(snr_sample_copy, return_cpu = False)
            pred_i.append(pre_lab)
            label_i.append(label)


        pred_i = torch.cat(pred_i)
        label_i = torch.cat(label_i)
        pred_label.append((pred_i, label_i))

    for idx, (pred_i, label_i) in enumerate(pred_label):
        pred_i = pred_i.cpu().numpy()
        label_i = label_i.cpu().numpy()
        acc_idx = accuracy_score(label_i, pred_i)
        step_acc_list.append((idx, acc_idx))

    return step_acc_list

def HN_tier(cldnn_dict, cnn_dict, meta_dict):
    # tier 2
    cldnn_cnn_list = [(k, cldnn_dict[k] + cnn_dict[k]) for k in set(cldnn_dict).intersection(cnn_dict)]
    cldnn_cnn_dict = {k: v for k, v in cldnn_cnn_list}
    cnn_meta_list = [(k, cnn_dict[k] + meta_dict[k]) for k in set(cnn_dict).intersection(meta_dict)]
    cnn_meta_dict = {k: v for k, v in cnn_meta_list}
    cldnn_meta_list = [(k, cldnn_dict[k] + meta_dict[k]) for k in set(cldnn_dict).intersection(meta_dict)]
    cldnn_meta_dict = {k: v for k, v in cldnn_meta_list}

    # tier 1
    cldnn_cnn_meta_list = [(k, cldnn_cnn_dict[k] + meta_dict[k]) for k in set(cldnn_cnn_dict).intersection(meta_dict)]
    cldnn_cnn_meta_list.sort(key=lambda x: x[1])

    tier_1_samples = cldnn_cnn_meta_list
    tier_1 = [ele[0] for ele in tier_1_samples]

    # remove tier 1 entries from tier 2
    for ele in tier_1:
        del cldnn_cnn_dict[ele]
        del cnn_meta_dict[ele]
        del cldnn_meta_dict[ele]

    cldnn_cnn_list = [(k, v) for k, v in cldnn_cnn_dict.items()]
    cnn_meta_list = [(k, v) for k, v in cnn_meta_dict.items()]
    cldnn_meta_list = [(k, v) for k, v in cldnn_meta_dict.items()]

    tier_2_samples = cldnn_cnn_list + cnn_meta_list + cldnn_meta_list
    tier_2_samples.sort(key=lambda x: x[1])
    tier_2 = [ele[0] for ele in tier_2_samples]

    # remove tier 1 & tier 2 entries from tier 3
    for ele in tier_1:
        del cldnn_dict[ele]
        del cnn_dict[ele]
        del meta_dict[ele]

    for ele in tier_2:
        try: del cldnn_dict[ele]
        except: pass
        try: del cnn_dict[ele]
        except: pass
        try: del meta_dict[ele]
        except: pass

    cldnn_list = [(k, v) for k, v in cldnn_dict.items()]
    cnn_list = [(k, v) for k, v in cnn_dict.items()]
    meta_list = [(k, v) for k, v in meta_dict.items()]

    tier_3_samples = cldnn_list + cnn_list + meta_list
    tier_3_samples.sort(key=lambda x: x[1])

    holistic_list = []

    holistic_list = holistic_list + tier_1_samples
    holistic_list = holistic_list + tier_2_samples
    holistic_list = holistic_list + tier_3_samples

    return holistic_list

class samplingTask(Task):
    """Refer to Ensemble Wrapper Subsampling for Deep Modulation Classification, TCCN 2021.\n
    https://codeocean.com/capsule/8397297/tree/v1\n
    https://github.com/dl4amc/dds
    """
    def __init__(self, args):
        super().__init__(args)

    def tuning(self):
        raise ValueError('Non support to tuning!')


    def exp_config(self, args):
        super().exp_config(args)

        self.sampling_tag = False
        self.sampler = sampler()
        if 'num_steps' in vars(args):
            self.sampler.num_steps = args.num_steps
        self.sampler.name = args.sampler_name if 'sampler_name' in vars(args) else 'subsamplerNet'

        self.model_fit_dir = os.path.join(self.fit_dir, self.model_name,  self.sampler.name, f'step{args.num_steps}')
        self.model_pred_dir = os.path.join(self.model_fit_dir, 'pred_results')
        self.model_result_file = os.path.join(self.model_pred_dir, 'results.npz')

        if args.test and args.clean:
            os_rmdirs(self.model_fit_dir)



    def load_preTrain_model(self, logger):
        logger.info('*'*80)
        logger.info('Loading the pre-training model for subsampling!')
        hyper = self.load_hyper(logger)
        # model = self.model_import()
        model = importlib.import_module(self.model_opts.import_path)
        model = getattr(model, self.model_opts.class_name)
        model = model(hyper, logger)
        model.load_pretraing_file(file_path = hyper.pretraining_allDatafile)
        model.eval()

        return model

    def subsamplerNet(self, logger):
        self.sampler.local_dir = os.path.join(self.fit_dir, self.model_name,  self.sampler.name,  'sampler')
        os_makedirs(self.sampler.local_dir)
        self.sampler.save_path = os.path.join(self.sampler.local_dir, f'snrs_step_acc.pt')
        try:
            snrs_step_acc = torch.load(self.sampler.save_path)
            logger.info('-'*80)
            logger.info(f'Read the saved snrs_step_acc from {self.sampler.save_path} successfully!')
        except:
            logger.info('*'*80)
            logger.info('Process the sampling with subsamplerNet for every step on every snr.')
            model = self.load_preTrain_model(logger)
            with tqdm(total = len(self.data_opts.snr_envs), desc='SNRs', mininterval=0.3, colour='red') as pbar:
                snrs_step_acc = []
                for step, snr in enumerate(self.data_opts.snr_envs):
                    snr_samples, snr_lables, _ = self.data_opts.snr_slice('train', snr)

                    num_chunk = int(snr_samples.shape[0] / model.hyper.batch_size)
                    Samples = torch.chunk(snr_samples, num_chunk, dim=0)
                    Label = torch.chunk(snr_lables, num_chunk, dim=0)

                    step_acc_list = SN_ranker(model,Samples,Label)
                    snrs_step_acc.append(step_acc_list)
                    pbar.update(1)

            torch.save(snrs_step_acc, self.sampler.save_path)
            logger.info(f'Save the snrs_step_acc to {self.sampler.save_path} !')

        snrs_sampling_idxs = []
        for _step_acc_list in snrs_step_acc:
            step_acc_list = deepcopy(_step_acc_list)
            step_acc_list.sort(key=lambda x: x[1])
            step_acc_list = step_acc_list[:self.sampler.num_steps]
            step_acc_list.sort(key=lambda x: x[0])
            step_idxs = [ele[0] for ele in step_acc_list]
            # logger.info(step_idxs)
            snrs_sampling_idxs.append(step_idxs)

        return snrs_sampling_idxs


    def holistic(self, logger):
        '''Process as https://codeocean.com/capsule/8397297/tree/v1 '''
        cldnn_tiers3 = ranker_check_load(key='cldnn_save_path', hyper=self.model_opts.hyper, num_steps=self.sampler.num_steps, logger=logger)
        vtcnn_tiers3 = ranker_check_load(key='vtcnn_save_path', hyper=self.model_opts.hyper, num_steps=self.sampler.num_steps,logger=logger)

        self.model_opts.hyper.meta_save_path= 'yield_results/2024-MaskAug/Mask.Baseline/RML2016.10a/subsamplerNet/fit/{}/subsamplerNet/sampler/snrs_step_acc.pt'.format(self.model_name)
        assert self.model_name != 'cldnn' or self.model_name != 'vtcnn'
        meta_tiers3 = ranker_check_load(key='meta_save_path', hyper=self.model_opts.hyper, num_steps=self.sampler.num_steps,logger=logger)

        logger.info('*'*80)
        logger.info('Process the sampling with holisticNet for every step on every snr.')

        snrs_sampling_idxs = []
        with tqdm(total = len(self.data_opts.snr_envs), desc='SNRs', mininterval=0.3, colour='red') as pbar:
            for step, (cldnn_dict, cnn_dict, meta_dict) in enumerate(zip(cldnn_tiers3, vtcnn_tiers3, meta_tiers3)):

                holistic_list = HN_tier(cldnn_dict, cnn_dict, meta_dict)
                holistic_list = holistic_list[:self.sampler.num_steps]
                holistic_list.sort(key=lambda x: x[1])
                holistic_list.sort(key=lambda x: x[0])

                step_idxs = [ele[0] for ele in holistic_list]
                snrs_sampling_idxs.append(step_idxs)

                pbar.update(1)

        assert len(snrs_sampling_idxs) == len(self.data_opts.snr_envs)
        return snrs_sampling_idxs

    def ensembleWrapper(self, logger):
        self.sampler.local_dir = os.path.join(self.model_fit_dir, 'sampler')
        os_makedirs(self.sampler.local_dir)
        self.sampler.save_path = os.path.join(self.sampler.local_dir, f'snrs_sampling_idxs.pt')
        try:
            snrs_sampling_idxs = torch.load(self.sampler.save_path)
            logger.info('-'*80)
            logger.info(f'Read the saved snrs_sampling_idxs from {self.sampler.save_path} successfully!')
        except:
            logger.info('*'*80)
            logger.info('Process the sampling with ensembleWrapper for every step on every snr.')

            def load_model(key):
                model_hyper = Opt(self.model_opts.hyper.dict[key])
                model_hyper.num_classes = self.data_opts.num_classes
                # model_hyper.model_fit_dir = None
                model_hyper.sig_len = self.data_opts.info.sig_len
                model_hyper.device = self.model_opts.hyper.device
                model_hyper.model_name = model_hyper.class_name
                model_hyper.data_name = self.data_name
                model = importlib.import_module(model_hyper.import_path)
                model = getattr(model, model_hyper.class_name)
                model = model(model_hyper, logger)
                model.load_pretraing_file(file_path = model_hyper.pretraining_allDatafile)
                model.eval()
                return model

            cnn = load_model('vtcnn')
            cldnn = load_model('cldnn')
            meta = self.load_preTrain_model(logger)

            def e_greedy_dds(k, train_set, epsilon=0.1, prev_snr_acc=0):
                '''This is the version directly replicated from the provided paper code link https://github.com/dl4amc/dds.
                However, we counter many errors and strong performance issue about this code.
                '''
                holistic_list =  []

                train_set_copy = deepcopy(train_set)
                Samples_copy, Label_copy = train_set_copy
                cnn_list = SN_ranker(model=cnn, Samples = Samples_copy, Label =Label_copy)
                cldnn_list = SN_ranker(model=cldnn, Samples = Samples_copy, Label =Label_copy)
                meta_list = SN_ranker(model=meta, Samples = Samples_copy, Label =Label_copy)

                cnn_dict = ranker_snr_dict(cnn_list, k)
                cldnn_dict = ranker_snr_dict(cldnn_list, k)
                meta_dict = ranker_snr_dict(meta_list, k)

                holistic_list = HN_tier(cldnn_dict, cnn_dict, meta_dict)
                holistic_list.sort(key=lambda x: x[1])
                holistic_list = holistic_list[:k]
                holistic_list.sort(key=lambda x: x[0])

                snr_idxs = [ele[0] for ele in holistic_list]

                if k == 0:
                    # In this place, ResNet is supposed to be re-trained to generate the curr_snr_acc.
                    # However, in the source code of this method, curr_snr_acc should be always greater than prev_snr_acc which is 0 and never updated. Thus, we jump to the final return.
                    return snr_idxs
                else:
                    train_set_copy = deepcopy(train_set)
                    for i in range(min(k, int(epsilon * self.data_opts.info.sig_len))):
                        # print(f'K: {k}\t i: {i}')
                        train_set = deepcopy(train_set_copy)
                        _Sample, Label = train_set
                        Sample = []
                        for sample in _Sample:
                            sample[:,:,snr_idxs[i]]=0
                            Sample.append(sample)

                        train_set = (Sample, Label)
                        final_idxs = e_greedy_dds(k-1, train_set)
                        final_idxs.append(snr_idxs[i])

                        if final_idxs is not None:
                            return final_idxs
                    return None

            def e_greedy(k, train_set, epsilon=0.1, host_dir='None'):
                '''Thus, we employ the modified version of ensembleWrapper, as described in the originl paper ramjee2021TCCN, as noted in the end of Sec.D 'having an unaltered Holistic Subsampler that greedily chooses the best sample at each iteration, which leads to the leftmost leaf of the tree in Figure 5.'
                '''
                train_set_copy = deepcopy(train_set)
                Samples, Label = train_set_copy
                final_idxs, tabu_list = [], []
                with tqdm(total = k, desc='Greedy', mininterval=0.3, colour='blue') as pbar:
                    for step, i in enumerate(list(range(k))):
                        cnn_list = SN_ranker(model=cnn, Samples = Samples, Label =Label)
                        cldnn_list = SN_ranker(model=cldnn, Samples = Samples, Label =Label)
                        meta_list = SN_ranker(model=meta, Samples = Samples, Label =Label)

                        cnn_dict = ranker_snr_dict(cnn_list, self.data_opts.info.sig_len)
                        cldnn_dict = ranker_snr_dict(cldnn_list, self.data_opts.info.sig_len)
                        meta_dict = ranker_snr_dict(meta_list, self.data_opts.info.sig_len)

                        holistic_list = HN_tier(cldnn_dict, cnn_dict, meta_dict)
                        holistic_list.sort(key=lambda x: x[1])
                        snr_idxs = [ele[0] for ele in holistic_list]
                        selected_idx = snr_idxs[0]

                        for selected_idx in snr_idxs:
                            if selected_idx not in tabu_list:
                                tabu_list.append(selected_idx)
                                _Samples = []
                                for sample in Samples:
                                    sample[:,:,selected_idx]=0
                                    _Samples.append(sample)
                                Samples = deepcopy(_Samples)
                                final_idxs.append(selected_idx)
                                break
                        pbar.update(1)
                final_idxs.sort()
                return final_idxs

            snrs_sampling_idxs = []
            with tqdm(total = len(self.data_opts.snr_envs), desc='SNRs', mininterval=0.3, colour='red') as pbar:
                for step, snr in enumerate(self.data_opts.snr_envs):
                    logger.info(f'Process the sampling on SNR: {snr}')

                    snr_save_dir = os.path.join(self.sampler.local_dir, f'SNR{snr}')
                    os_makedirs(snr_save_dir)
                    snr_save_path = os.path.join(snr_save_dir, f'snr{snr}_selected_idxs.pt')
                    try:
                        final_idxs = torch.load(snr_save_path)
                        logger.info('-'*80)
                        logger.info(f'Read the saved snr{snr}_selected_idxs from {snr_save_path} successfully!')
                    except:
                        snr_samples, snr_lables, _ = self.data_opts.snr_slice('train', snr)

                        num_chunk = int(snr_samples.shape[0] / meta.hyper.batch_size)
                        Samples = torch.chunk(snr_samples, num_chunk, dim=0)
                        Label = torch.chunk(snr_lables, num_chunk, dim=0)

                        train_set = (Samples, Label)

                        final_idxs = e_greedy(k=self.sampler.num_steps, train_set=train_set, epsilon=0.1,host_dir = snr_save_dir)
                        snrs_sampling_idxs.append(final_idxs)
                        torch.save(final_idxs, snr_save_path)
                        logger.info(f'Save the snr{snr}_selected_idxs to {snr_save_path} !')

                    pbar.update(1)

            torch.save(snrs_sampling_idxs, self.sampler.save_path)
            logger.info(f'Save the snrs_sampling_idxs to {self.sampler.save_path} !')

        # assert len(snrs_sampling_idxs) == len(self.data_opts.snr_envs)
        return snrs_sampling_idxs



    def fqi(self,logger):
        self.sampler.local_dir = os.path.join(self.fit_dir, self.model_name,  self.sampler.name,  'sampler')
        os_makedirs(self.sampler.local_dir)
        self.sampler.save_path = os.path.join(self.sampler.local_dir, f'snrs_step_acc.pt')
        try:
            snrs_step_acc = torch.load(self.sampler.save_path)
            logger.info('-'*80)
            logger.info(f'Read the saved snrs_step_acc from {self.sampler.save_path} successfully!')
        except:
            logger.info('*'*80)
            logger.info('Process the sampling with FQI for every step on every snr.')
            model = self.load_preTrain_model(logger)
            with tqdm(total = len(self.data_opts.snr_envs), desc='SNRs', mininterval=0.3, colour='red') as pbar:
                snrs_step_acc = []
                for step, snr in enumerate(self.data_opts.snr_envs):
                    snr_samples, snr_lables, _ = self.data_opts.snr_slice('train', snr)

                    num_chunk = int(snr_samples.shape[0] / model.hyper.batch_size)
                    Samples = torch.chunk(snr_samples, num_chunk, dim=0)
                    Label = torch.chunk(snr_lables, num_chunk, dim=0)

                    step_acc_list = []

                    pred_label = []
                    for idx in trange(self.data_opts.info.sig_len, colour='green', leave=False):
                        pred_i = []
                        label_i = []
                        for (sample, label) in zip(Samples, Label):
                            sample = sample.to(model.hyper.device)
                            snr_sample_copy = sample.detach().clone()
                            snr_sample_copy[:,:, idx] = torch.mean(snr_sample_copy[:,:, idx]) #  = torch.mean() is the settings as https://codeocean.com/capsule/8397297/tree/v1 ,  = 0 is the settings as https://github.com/dl4amc/dds
                            pre_lab = model.predict(snr_sample_copy, return_cpu = False)
                            pred_i.append(pre_lab)
                            label_i.append(label)

                        pred_i = torch.cat(pred_i)
                        label_i = torch.cat(label_i)
                        pred_label.append((pred_i, label_i))

                    for idx, (pred_i, label_i) in enumerate(pred_label):
                        pred_i = pred_i.cpu().numpy()
                        label_i = label_i.cpu().numpy()
                        acc_idx = mean_squared_error(label_i, pred_i)
                        step_acc_list.append((idx, acc_idx))

                    snrs_step_acc.append(step_acc_list)
                    pbar.update(1)

            torch.save(snrs_step_acc, self.sampler.save_path)
            logger.info(f'Save the snrs_step_acc to {self.sampler.save_path} !')

        snrs_sampling_idxs = []
        for _step_acc_list in snrs_step_acc:
            step_acc_list = deepcopy(_step_acc_list)
            step_acc_list.sort(key=lambda x: x[1], reverse=True)
            step_acc_list = step_acc_list[:self.sampler.num_steps]
            step_acc_list.sort(key=lambda x: x[0])
            step_idxs = [ele[0] for ele in step_acc_list]
            snrs_sampling_idxs.append(step_idxs)

        return snrs_sampling_idxs

    def rfs(self, logger):
        self.sampler.local_dir = os.path.join(self.fit_dir, f'{self.sampler.name}.sampler')
        os_makedirs(self.sampler.local_dir)

        self.sampler.save_path = os.path.join(self.sampler.local_dir, f'snrs_step_order.pt')
        try:
            snrs_step_order = torch.load(self.sampler.save_path)
            logger.info('-'*80)
            logger.info(f'Read the saved snrs_step_order from {self.sampler.save_path} successfully!')
        except:
            logger.info('*'*80)
            logger.info('Process the sampling with RFS for every step on every snr.')

            ## Depend on snr information. However this implementation is extremely time-consuming, Thus, we choose to implement rfs on validation set.
            with tqdm(total = len(self.data_opts.snr_envs), desc='SNRs', mininterval=0.3, colour='red') as pbar:
                snrs_step_order = []
                for step, snr in enumerate(self.data_opts.snr_envs):
                    snr_step_save_path = os.path.join(self.sampler.local_dir, f'snrID{step}_step_order.pt')
                    if os.path.exists(snr_step_save_path):
                        score_des = torch.load(snr_step_save_path)
                    else:
                        snr_samples, snr_lables, _ = self.data_opts.snr_slice('val', snr)
                        snr_samples = snr_samples.cpu().numpy()
                        snr_lables = snr_lables.cpu().numpy()
                        x_train  = np.append(snr_samples[:,0,:], snr_samples[:,1,:], axis = 0)
                        label = np.append(snr_lables, snr_lables, axis = 0)
                        score_des = RFS.rfs(x_train, label, mode = 'index', verbose = True)

                        torch.save(score_des, snr_step_save_path)
                    snrs_step_order.append(score_des)
                    pbar.update(1)


            torch.save(snrs_step_order, self.sampler.save_path)
            logger.info(f'Save the snrs_step_order to {self.sampler.save_path} !')

        snrs_sampling_idxs = []
        for _step_des_list in snrs_step_order:
            step_idxs = _step_des_list[:self.sampler.num_steps]
            step_idxs.sort()
            snrs_sampling_idxs.append(step_idxs)

        return snrs_sampling_idxs


    def fisher(self,logger):
        self.sampler.local_dir = os.path.join(self.fit_dir, f'{self.sampler.name}.sampler')
        os_makedirs(self.sampler.local_dir)

        self.sampler.save_path = os.path.join(self.sampler.local_dir, f'snrs_step_order.pt')
        try:
            snrs_step_order = torch.load(self.sampler.save_path)
            logger.info('-'*80)
            logger.info(f'Read the saved snrs_step_order from {self.sampler.save_path} successfully!')
        except:
            logger.info('*'*80)
            logger.info('Process the sampling with Fisher Score for every step on every snr.')

            ## Depend on snr information. However this implementation is extremely time-consuming, Thus, we choose to implement rfs on validation set.
            with tqdm(total = len(self.data_opts.snr_envs), desc='SNRs', mininterval=0.3, colour='red') as pbar:
                snrs_step_order = []
                for step, snr in enumerate(self.data_opts.snr_envs):
                    snr_samples, snr_lables, _ = self.data_opts.snr_slice('val', snr)
                    snr_samples = snr_samples.cpu().numpy()
                    snr_lables = snr_lables.cpu().numpy()
                    x_train  = np.append(snr_samples[:,0,:], snr_samples[:,1,:], axis = 0)
                    label = np.append(snr_lables, snr_lables, axis = 0)
                    score_des = fisher_score(abs(x_train), label, mode = 'index')
                    snrs_step_order.append(score_des)
                    pbar.update(1)


            torch.save(snrs_step_order, self.sampler.save_path)
            logger.info(f'Save the snrs_step_order to {self.sampler.save_path} !')

        snrs_sampling_idxs = []
        for _step_des_list in snrs_step_order:
            step_idxs = _step_des_list[:self.sampler.num_steps]
            step_idxs.sort()
            snrs_sampling_idxs.append(step_idxs)

        return snrs_sampling_idxs

    def laplacian(self,logger):
        self.sampler.local_dir = os.path.join(self.fit_dir, f'{self.sampler.name}.sampler')
        os_makedirs(self.sampler.local_dir)

        self.sampler.save_path = os.path.join(self.sampler.local_dir, f'snrs_step_order.pt')
        try:
            snrs_step_order = torch.load(self.sampler.save_path)
            logger.info('-'*80)
            logger.info(f'Read the saved snrs_step_order from {self.sampler.save_path} successfully!')
        except:
            logger.info('*'*80)
            logger.info('Process the sampling with Laplacian Score for every step on every snr.')

            ## Depend on snr information. However this implementation is extremely time-consuming, Thus, we choose to implement rfs on validation set.
            with tqdm(total = len(self.data_opts.snr_envs), desc='SNRs', mininterval=0.3, colour='red') as pbar:
                snrs_step_order = []
                for step, snr in enumerate(self.data_opts.snr_envs):
                    snr_samples, _, _ = self.data_opts.snr_slice('val', snr)
                    snr_samples = snr_samples.cpu().numpy()
                    # snr_lables = snr_lables.cpu().numpy()
                    x_train  = np.append(snr_samples[:,0,:], snr_samples[:,1,:], axis = 0)
                    # label = np.append(snr_lables, snr_lables, axis = 0)

                    kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
                    W = construct_W.construct_W(x_train, **kwargs_W)
                    score_des = lap_score(x_train, W=W, mode = 'index')
                    snrs_step_order.append(score_des)
                    pbar.update(1)


            torch.save(snrs_step_order, self.sampler.save_path)
            logger.info(f'Save the snrs_step_order to {self.sampler.save_path} !')

        snrs_sampling_idxs = []
        for _step_des_list in snrs_step_order:
            step_idxs = _step_des_list[:self.sampler.num_steps]
            step_idxs.sort()
            snrs_sampling_idxs.append(step_idxs)

        return snrs_sampling_idxs

    def sampling(self, logger):

        if self.sampler.name == 'subsamplerNet':
            snrs_sampling_idxs = self.subsamplerNet(logger)
        elif self.sampler.name == 'holistic':
            snrs_sampling_idxs = self.holistic(logger)
        elif self.sampler.name == 'ensembleWrapper':
            snrs_sampling_idxs = self.ensembleWrapper(logger)
        elif self.sampler.name == 'fqi':
            snrs_sampling_idxs = self.fqi(logger)
        elif self.sampler.name == 'rfs':
            snrs_sampling_idxs = self.rfs(logger)
        elif self.sampler.name == 'fisher':
            snrs_sampling_idxs = self.fisher(logger)
        elif self.sampler.name == 'lap':
            snrs_sampling_idxs = self.laplacian(logger)

        self.sampling_tag = True
        return deepcopy(snrs_sampling_idxs)


    def sampling_dataset(self, data_set, data_idx):
        Signals, Labels = data_set

        data_sampling_set = torch.zeros((Signals.shape[0], Signals.shape[1], self.sampler.num_steps)).float()

        from taskRecog.util import snr_slice
        for snr, step_idxs in zip(self.data_opts.snr_envs, self.snrs_sampling_idxs):
            sig_i, _, idx_i = snr_slice(data_set, data_idx, self.data_opts.SNR_values, snr)


            # step_idxs = step_idxs.tolist()
            # sig_data = []
            # for step in step_idxs:
            #     data_i = sig_i[:,:, step]
            #     sig_data.append(data_i)
            # sig_data = torch.stack(sig_data, dim=2)
            sig_data = sig_i[:,:, step_idxs]

            data_sampling_set[idx_i] = sig_data


        return data_sampling_set, Labels


    def load_fitset(self, cid_hyper=None):
        batch_size = 64 if cid_hyper is None else cid_hyper.batch_size

        if 'snrs_sampling_idxs' not in self.dict:
             raise ValueError('Non sampling stage! Please run sampling function first.')

        train_set = self.sampling_dataset(self.data_opts.train_set, self.data_opts.train_idx)
        val_set = self.sampling_dataset(self.data_opts.val_set, self.data_opts.val_idx)

        train_loader = set_dataloader(batch_size= batch_size, data_set=train_set)
        val_loader = set_dataloader(batch_size= batch_size, data_set=val_set)

        return train_loader, val_loader

    def load_testset(self, cid_hyper=None):
        batch_size = 64 if cid_hyper is None else cid_hyper.batch_size
        Signals, Labels = self.sampling_dataset(self.data_opts.test_set, self.data_opts.test_idx)
        Sample_list = []
        Label_list = []

        for snr in self.data_opts.snr_envs:
            _, _, idx_i = self.data_opts.snr_slice('test', snr)
            sig_i = Signals[idx_i]
            lab_i = Labels[idx_i]

            if batch_size > sig_i.shape[0]:
                batch_size = sig_i.shape[0]
            num_chunk = int(sig_i.shape[0] / batch_size)

            Sample = torch.chunk(sig_i, num_chunk, dim=0)
            Label = torch.chunk(lab_i, num_chunk, dim=0)

            Sample_list.append(Sample)
            Label_list.append(Label)

        return Sample_list, Label_list

    def load_hyper(self, clogger):
        clogger.critical('*'*80)
        clogger.critical('Dataset: {}\t Model:{} \t Class: {}'.format(
            self.data_name, self.model_name, self.data_opts.num_classes))

        cid_hyper = self.hyper_config(clogger)
        cid_hyper.num_classes = self.data_opts.num_classes
        cid_hyper.model_fit_dir = self.model_fit_dir
        cid_hyper.model_name = self.model_name
        cid_hyper.data_name = self.data_name

        if self.sampling_tag:
            cid_hyper.sig_len = self.sampler.num_steps

        return cid_hyper

    def model_config(self, clogger):
        '''
        get the corresponding model's hyperparameters(default no tuning)
        use them to create a model instance and return
        '''
        model_hyper = self.hyper_config(clogger)
        model_hyper.sig_len = self.sampler.num_steps
        model = importlib.import_module(self.model_opts.import_path)
        model = getattr(model, self.model_opts.class_name)
        model = model(model_hyper, clogger)
        return model

    def conduct(self, force_update=None, ):
        if force_update is not None:
            if force_update in [True, False]:
                self.force_update = force_update
            else:
                raise ValueError(
                    'force_update parameter is incorrect, please set with True or False.')

        if not os.path.exists(self.model_result_file) or self.force_update:
            if os.path.exists(self.model_result_file):
                os_rmdirs(self.model_pred_dir)
            os_makedirs(self.model_pred_dir)

            task_logger = self.logger_config(
                self.model_fit_dir, 'train')
            self.load_data(logger=task_logger)

            self.hyper_config(task_logger)
            cuda_exist = torch.cuda.is_available()
            if self.args.ocm > 0.25 and cuda_exist:
                torch.cuda.empty_cache()
                from taskRecog.util import check_mem
                total, used = check_mem(self.args.gid)
                # total = int(total)
                # used = int(used)
                # max_mem = int(total * self.args.ocm)
                block_mem = int((total - used) * self.args.ocm)
                with torch.no_grad():
                    x = torch.ones((256,1024,block_mem)).cuda(device=self.model_opts.hyper.device)
                    del x
                    task_logger.info('Total GPU Memory: {} MB; Used {} MB; OCM: {}; Reserved: {} MB'.format(total, used, self.args.ocm, block_mem))
            self.snrs_sampling_idxs = self.sampling(logger=task_logger)

            self.conduct_fit(clogger=task_logger,
                             result_file=self.model_result_file)

            self.fit_statue = True

            self.evaluate(elogger=task_logger) # toDo: add eval dir with sampler.num_steps tag
        else:
            self.evaluate()


    def evaluate(self, elogger = None, force_update=False, ave_confMax = False):
        eLogger = set_logger(os.path.join(self.eval_dir, '_log_', '{}.{}.sub{}.eval.log'.format(self.data_name, self.model_name, self.sampler.num_steps)), '{}.{}'.format(
                self.data_name, self.model_name.upper()), self.logger_level) if elogger is None else elogger

        if self.data_statue is False:
            self.load_data(logger=eLogger)

        self.model_eval_dir = os.path.join(self.eval_dir, self.model_name, f'sampling{self.sampler.num_steps}')
        self.eval_acc_dir = os.path.join(self.model_eval_dir, 'accuracy')
        self.eval_plot_dir = os.path.join(self.model_eval_dir, 'figures')

        os_makedirs(self.eval_acc_dir)
        os_makedirs(self.eval_plot_dir)
        os_makedirs(self.model_pred_dir)

        # for i in self.cid_list: # multiple cross validation in the future version

        if os.path.exists(self.model_result_file) and force_update is False:
            with np.load(self.model_result_file) as data:
                pre_lab_all, label_all = data['pred'], data['label']
        else:
            pre_lab_all, label_all,_ = self.conduct_fit()


        Confmat_Set = np.zeros((len(self.data_opts.num_snrs), self.data_opts.num_classes, self.data_opts.num_classes), dtype=int)
        Accuracy_list = np.zeros(len(self.data_opts.num_snrs), dtype=float)

        for snr_i, (pred_i, label_i) in enumerate(zip(pre_lab_all, label_all)):
            cm_i =  confusion_matrix(label_i, pred_i)
            Confmat_Set[snr_i, :, :] = cm_i
            Accuracy_list[snr_i] = accuracy_score(label_i, pred_i)

        pre_lab_all = np.concatenate(pre_lab_all)
        label_all = np.concatenate(label_all)

        F1_score = f1_score(label_all, pre_lab_all, average='macro')
        kappa = cohen_kappa_score(label_all, pre_lab_all)
        acc = np.mean(Accuracy_list)

        eLogger.info('Overall Accuracy is: {:.2f}%'.format(acc * 100))
        eLogger.info(f'Macro F1-score is: {F1_score:.4f}')
        eLogger.info(f'Kappa Coefficient is: {kappa:.4f}')

        if ave_confMax:
            save_confmat(Confmat_Set, self.data_opts.num_snrs, self.data_opts.classes, self.eval_plot_dir)


        Accuracy_Mods = save_snr_acc(Accuracy_list, Confmat_Set, self.data_opts.num_snrs, self.data_name, self.data_opts.classes.keys(), self.eval_plot_dir)

        tgt_acc_file = os.path.join(self.eval_acc_dir, 'acc.npz')
        np.savez(tgt_acc_file, acc_overall = Accuracy_list, acc_mods= Accuracy_Mods)
        eLogger.info('Save accuracy file to the location: {}'.format(tgt_acc_file))

        return F1_score, kappa, acc