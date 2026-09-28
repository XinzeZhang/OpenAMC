import os
import sys

import torch
import numpy as np
from collections import Counter
from tqdm.auto import tqdm, trange
from taskRecog.util import Opt, set_dataloader
from taskRecog.Wrapper import Task
from data.Loader import TaskDataset
from taskRecog.Parser import get_parser

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, cohen_kappa_score
from taskRecog.util import os_makedirs, os_rmdirs, set_logger, fix_seed
from taskRecog.util import save_training_process, save_confmat, save_snr_acc
from copy import deepcopy

from taskRecog.featureselection.snr_aware.softmaskWrapper import SoftMaskTask
from taskRecog.util import chunk_list_nsub_dict


class SnrPredMaskTask(SoftMaskTask):
    def __init__(self, args, snrmodel):
        super().__init__(args)
        self.masktask_config(args, snrmodel)

    def masktask_config(self, args, snrmodel):
        self.snr_pred_target = args.snr_pred
        self.snr_num_classes = args.num_env
        self.snrmodel = snrmodel

        self.ori_result_path = os.path.join(self.model_pred_dir, 'results.ori.npz')
        self.predmask_result_path = os.path.join(self.model_pred_dir, 'results.predmask.npz')
        self.refmask_result_path = os.path.join(self.model_pred_dir, 'results.refmask.npz')

    def predSNR_masking_data(self, data_set, pre_lab_all):
        Signals, Labels = data_set
        data_sampling_set = Signals.detach().clone()
        assert len(pre_lab_all) == data_sampling_set.shape[0]
        self.logger.info(f'Masking the dataset {Signals.size()} with snrs_masks')

        for i in trange(len(pre_lab_all)):
            pred_snr_id = pre_lab_all[i]
            step_idxs = self.snrs_mask_idxs[pred_snr_id]
            data_sampling_set[i,:,:] = Signals[i,:, :].mul(torch.tensor(step_idxs))

        return data_sampling_set, Labels

    def snrpred(self, dataset, batch_size = 1024):
        pre_batches = []
        Signals, _ = dataset
        num_chunk = int(Signals.shape[0] / batch_size)

        Sample = torch.chunk(Signals, num_chunk, dim=0)
        for step, sig_batch in tqdm(enumerate(Sample), total=len(Sample)):
            pre_lab = self.snrmodel.predict(sig_batch)
            pre_batches.append(pre_lab)

        pre_lab_all = np.concatenate(pre_batches).tolist()
        return pre_lab_all

    def sampling_dataset(self, data_set, data_idx, data_tag = 'pred'):

        assert data_tag in ['pred', 'ref', 'non']

        if data_tag == 'non':
            data_sampling_set, Labels = data_set
        else:
            if data_tag == 'pred':
                pre_lab_all = self.snrpred(data_set)
            elif data_tag == 'ref':
                Cor_SNRs = map(lambda x: self.data_opts.SNR_values[x], data_idx)
                Cor_SNRs = list(Cor_SNRs)
                pre_lab_all = [self.sampler.snr_dict[snr] for snr in Cor_SNRs]

            data_sampling_set, Labels = self.predSNR_masking_data(data_set,pre_lab_all)

        return data_sampling_set, Labels

    def load_fitset(self, cid_hyper=None):
        batch_size = 64 if cid_hyper is None else cid_hyper.batch_size

        if 'snrs_mask_idxs' not in self.dict:
             raise ValueError('Non sampling stage! Please run sampling function first.')

        tag = 'pred' if self.snr_pred_target == 'all' else 'ref'
        train_set = self.sampling_dataset(self.data_opts.train_set, self.data_opts.train_idx, tag)
        val_set = self.sampling_dataset(self.data_opts.val_set, self.data_opts.val_idx, tag)

        train_loader = set_dataloader(batch_size= batch_size, data_set=train_set)
        val_loader = set_dataloader(batch_size= batch_size, data_set=val_set)

        return train_loader, val_loader

    def conduct(self, force_update=None, finetune = False):
        if force_update is not None:
            if force_update in [True, False]:
                self.force_update = force_update
            else:
                raise ValueError(
                    'force_update parameter is incorrect, please set with True or False.')
        result_statue = os.path.exists(self.ori_result_path) and os.path.exists(self.predmask_result_path) and os.path.exists(self.refmask_result_path)

        self.logger = self.logger_config(self.model_fit_dir, 'train')
        self.load_data(logger=self.logger)

        self.snrs_mask_idxs = self.masking(logger=self.logger)

        if not result_statue or self.force_update:
            if result_statue:
                os_rmdirs(self.model_pred_dir)
            os_makedirs(self.model_pred_dir)

            self.conduct_fit(clogger=self.logger, finetune=finetune)

            self.fit_statue = True


        ori_acc, pred_mask_acc, ref_mask_acc = self.evaluate(elogger=self.logger)

        return ori_acc, pred_mask_acc, ref_mask_acc

    def conduct_fit(self, clogger = None, finetune = False, xfit_stats = True):
        try:
            if clogger is None:
                clogger = self.logger_config(
                    self.model_fit_dir, 'train')

            cid_hyper = self.load_hyper(clogger)
            train_loader, val_loader = None, None
            if xfit_stats:
                train_loader, val_loader = self.load_fitset(cid_hyper)
                clogger.critical('Loading training set and validation set.')
                clogger.info(f'Fit batch size: {train_loader.batch_size}')
                clogger.info(f"Train_loader batch: {len(train_loader)}")
                clogger.info(f"Val_loader batch: {len(val_loader)}")
                clogger.critical('>'*40)

            # if finetune:
            #     cid_hyper.pretraining_file = cid_hyper.pretraining_allDatafile
            # else:
            #     cid_hyper.pretraining_file = ''

            model = self.model_import()
            model = model(cid_hyper, clogger) #todo: check model_fit_dir in model and model trainer
            clogger.critical('Loading Model.')
            clogger.critical(f'Model: \n{str(model)}')

            # if self.model_opts.arch == 'torch_nn':
            clogger.info(">>> Total params: {:.2f}M".format(
                    sum(p.numel() for p in list(model.parameters())) / 1000000.0))

            clogger.critical('Start fit.')
            epochs_stats =  model.xfit(train_loader, val_loader,finetune = finetune,xfit_stats=False)
            if epochs_stats is not None and set(['val_loss','val_acc', 'train_loss', 'train_acc', 'lr_list']).issubset(epochs_stats.columns):
                loss_dir = os.path.join(self.model_fit_dir, 'loss_curve')
                lossfig_dir = os.path.join(loss_dir, 'figure')
                save_training_process(epochs_stats, plot_dir=lossfig_dir)

            clogger.critical('>'*40+'\nEnd fit.')
            results = self.eval_testset(model,clogger)

            return results
        except:
            clogger.exception(
                '{}\nGot an error on conduction.\n{}'.format('!'*50, '!'*50))
            raise SystemExit()

    def eval_testset(self, model, logger):

        def eval_samples(model, sample_list, label_list):
            pre_lab_all = []
            label_all = []
            # loop of SNRs in test_sample_list
            for (Sample, Label) in tqdm(zip(test_sample_list, test_lable_list), total=len(test_sample_list)):
                pred_i = []
                label_i = []
                for (sample, label) in zip(Sample, Label):
                    pre_lab = model.predict(sample)
                    pred_i.append(pre_lab)
                    label_i.append(label)
                pred_i = np.concatenate(pred_i)
                label_i = np.concatenate(label_i)

                pre_lab_all.append(pred_i)
                label_all.append(label_i)

            return pre_lab_all, label_all

        results = Opt()

        model.eval()
        logger.critical('>'*40)
        logger.critical('Evaluation on the original testing set.')
        test_set = self.sampling_dataset(self.data_opts.test_set, self.data_opts.test_idx, 'non')
        test_sample_list, test_lable_list = self.snr_testset(model.hyper.batch_size, test_set)
        pre_lab_all, label_all = eval_samples(model, test_sample_list, test_lable_list)
        np.savez(self.ori_result_path, pred = pre_lab_all, label= label_all)
        logger.critical('Save result file to the location: {}'.format(self.ori_result_path))
        results.ori = (deepcopy(pre_lab_all), deepcopy(label_all))

        logger.critical('>'*40)
        logger.critical('Evaluation on the pred_masking testing set.')
        test_set = self.sampling_dataset(self.data_opts.test_set, self.data_opts.test_idx, 'pred')
        test_sample_list, test_lable_list = self.snr_testset(model.hyper.batch_size, test_set)
        pre_lab_all, label_all = eval_samples(model, test_sample_list, test_lable_list)
        np.savez(self.predmask_result_path, pred = pre_lab_all, label= label_all)
        logger.critical('Save result file to the location: {}'.format(self.predmask_result_path))
        results.predmask = (deepcopy(pre_lab_all), deepcopy(label_all))

        logger.critical('>'*40)
        logger.critical('Evaluation on the ref_masking testing set.')
        test_set = self.sampling_dataset(self.data_opts.test_set, self.data_opts.test_idx, 'ref')
        test_sample_list, test_lable_list = self.snr_testset(model.hyper.batch_size, test_set)
        pre_lab_all, label_all = eval_samples(model, test_sample_list, test_lable_list)
        np.savez(self.refmask_result_path, pred = pre_lab_all, label= label_all)
        logger.critical('Save result file to the location: {}'.format(self.refmask_result_path))
        results.refmask = (deepcopy(pre_lab_all), deepcopy(label_all))

        logger.critical('-'*80)

        return results


    def evaluate(self, elogger = None, xfit_stats=False, ave_confMax = False):

        if elogger is None:
            self.logger = set_logger(os.path.join(self.eval_dir, '_log_', '{}.{}.eval.log'.format(self.data_name, self.model_name)), '{}.{}'.format(
                self.data_name, self.model_name.upper()), self.logger_level)

        if self.data_statue is False:
            self.load_data(logger=self.logger)
            self.snrs_mask_idxs = self.masking(logger=self.logger)

        self.model_eval_dir = os.path.join(self.eval_dir, self.model_name)
        self.eval_acc_dir = os.path.join(self.model_eval_dir, 'accuracy')
        self.eval_plot_dir = os.path.join(self.model_eval_dir, 'figures')

        os_makedirs(self.eval_acc_dir)
        os_makedirs(self.eval_plot_dir)
        os_makedirs(self.model_pred_dir)

        # for i in self.cid_list: # multiple cross validation in the future version

        if self.args.force_update is False:
            force_update = False
        else:
            force_update = True

        results = Opt()
        if force_update is False and os.path.exists(self.ori_result_path) and os.path.exists(self.predmask_result_path) and os.path.exists(self.refmask_result_path):
            with np.load(self.ori_result_path) as data:
                pre_lab_all, label_all = data['pred'], data['label']
                results.ori = (deepcopy(pre_lab_all), deepcopy(label_all))
            with np.load(self.predmask_result_path) as data:
                pre_lab_all, label_all = data['pred'], data['label']
                results.predmask = (deepcopy(pre_lab_all), deepcopy(label_all))
            with np.load(self.refmask_result_path) as data:
                pre_lab_all, label_all = data['pred'], data['label']
                results.refmask = (deepcopy(pre_lab_all), deepcopy(label_all))
        else:
            results = self.conduct_fit(finetune = False, xfit_stats = xfit_stats)

        pre_lab_all, label_all = results.ori
        ori_acc = self.record_result(pre_lab_all, label_all, 'ori',ave_confMax)
        pre_lab_all, label_all = results.predmask
        predmask_acc = self.record_result(pre_lab_all, label_all, 'predmask',ave_confMax)
        pre_lab_all, label_all = results.refmask
        refmask_acc = self.record_result(pre_lab_all, label_all, 'refmask',ave_confMax)

        return ori_acc, predmask_acc, refmask_acc