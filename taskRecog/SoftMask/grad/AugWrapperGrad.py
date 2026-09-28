import os
import torch
import numpy as np
from tqdm.auto import tqdm,trange
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, cohen_kappa_score

from taskRecog.util import Opt, os_makedirs, chunk_list_nsub,set_logger, set_dataloader,os_rmdirs
from taskRecog.util import save_training_process, save_confmat, save_snr_acc
from taskRecog.Wrapper import Task
import importlib
from numpy.random import uniform
from copy import deepcopy

from taskRecog.dataAug.SoftMask.grad.util import Sampler, loss_calweight, AddMask, eval_samples
from taskRecog.dataAug.SoftMask.grad.util import MaskTrainer
# CLPSO
# Liang, J.J., Qin, A.K., Suganthan, P.N. and Baskar, S., 2006. Comprehensive learning particle swarm optimizer for global optimization of multimodal functions. IEEE transactions on evolutionary computation, 10(3), pp.281-295.
# ARO
# Wang, L., Cao, Q., Zhang, Z., Mirjalili, S., & Zhao, W. (2022). Artificial rabbits optimization: A new bio-inspired meta-heuristic algorithm for solving engineering optimization problems. Engineering Applications of Artificial Intelligence, 114, 105082.

# def hard_mask(x, code):
#     b = torch.rand_like(x).to(code.device)
#     maskcode = torch.zeros_like(b).to(code.device)
#     activate_a = torch.empty_like(b).to(code.device)
#     activate_a[:,] = torch.sigmoid(code)
#     maskcode[activate_a >= b] = 1

#     return maskcode.float()



class SoftAugTask(Task):
    def __init__(self, args):
        super().__init__(args)

        self.model_fit_dir = os.path.join(self.fit_dir)
        self.model_pred_dir = os.path.join(self.model_fit_dir, 'pred_results')

        self.model_hyper = self.hyper_config()

        self.masking_tag = False
        # from taskRecog.dataAug.SoftMask.grad.util import Sampler
        self.sampler = Sampler(args)
        self.sampler.local_dir = os.path.join(self.fit_dir, 'sampler')
        # self.sampler.local_dir = os.path.join(self.fit_dir, self.model_name, 'sampler')
        os_makedirs(self.sampler.local_dir)

        if 'sampler_file' in self.model_opts.hyper.dict and os.path.exists(self.model_opts.hyper.sampler_file):
            self.sampler.save_path = self.model_opts.hyper.sampler_file
        else:
            self.sampler.save_path = os.path.join(self.sampler.local_dir, f'snrs_step_mask.pt')

        self.ori_result_path = os.path.join(
            self.model_pred_dir, 'results.ori.npz')
        self.val_result_path = os.path.join(
            self.model_pred_dir, 'results.val.npz')
        self.valmask_result_path = os.path.join(
            self.model_pred_dir, 'results.valmask.npz')
        self.refmask_result_path = os.path.join(
            self.model_pred_dir, 'results.refmask.npz')
        # self.snr_num_classes = args.num_env
        assert 'validaug' in vars(args)
        assert args.validaug in ['ref', 'non', 'aug']
        self.validaug = args.validaug

        assert 'lossweight' in vars(args)
        assert args.lossweight in ['one', 'minus', 'plus', 'mratio', 'inverse']
        self.lossweight = args.lossweight
        self.IQ = args.IQ
        self.mask_type = args.mask
        self.args.num_sampling = args.num_sampling if 'num_sampling' in vars(args) else 0

    def logger_config(self, dir, stage, name = None):
        log_path = os.path.join(dir, '{}.{}.log'.format(stage,self.data_name))
        log_name = '{}.{}.{}'.format(
            self.exp_name,self.data_name, self.model_name) if name is None else name
        logger = set_logger(log_path, log_name, self.logger_level)
        return logger



    def sampling_dataset(self, data_set, data_idx, data_tag='ref'):
        assert data_tag in ['ref', 'non']
        if data_tag == 'non':
            data_sampling_set, Labels = data_set
        else:
            Cor_SNRs = list(map(lambda x: self.data_opts.SNR_values[x], data_idx))
            snr_lab_all = [self.sampler.snr_dict[snr] for snr in Cor_SNRs]
            data_sampling_set, Labels = self.SNR_masking_data(
                data_set, snr_lab_all)
        return data_sampling_set, Labels

    def SNR_masking_data(self, data_set, snr_lab_all):
        Signals, Labels = data_set
        data_sampling_set = Signals.detach().clone()
        assert len(snr_lab_all) == data_sampling_set.shape[0]
        self.logger.info(
            f'Masking the dataset {Signals.size()} with snrs_masks')
        for i in trange(len(snr_lab_all)):
            snr_ith = snr_lab_all[i]
            step_idxs = self.snrs_mask_idxs[snr_ith]
            data_sampling_set[i, :, :] = Signals[i, :, :].mul(
                step_idxs.to(Signals.device))

        return data_sampling_set, Labels

    def aug_dataset(self, data_set, data_idx, tag='one', set_name = 'trainset'):
        assert tag in ['one', 'minus', 'plus', 'mratio', 'inverse']
        # snr_ithLabels = self.get_snrs_labels(data_idx)
        Cor_SNRs = map(lambda x: self.data_opts.SNR_values[x], data_idx)
        Cor_SNRs = list(Cor_SNRs)
        snr_ithLabels = [self.sampler.snr_dict[snr] for snr in Cor_SNRs]
        snr_ithLabels = torch.tensor(snr_ithLabels)
        Signals, Labels = data_set
        # assert len(snr_ithLabels) == Signals.shape[0]

        mask_ratio_list = []
        snr_augweight = np.empty(len(self.snrs_mask_idxs))
        # if self.args.num_sampling > 0:
        #     snr_samplemask = torch.empty(size=(len(self.snrs_mask_idxs), self.args.num_sampling, mask_dim))
        # maskstatus = np.empty(len(self.snrs_mask_idxs))
        for ith_snr, step_idxs in enumerate(self.snrs_mask_idxs):
            mask_ratio, r = loss_calweight(tag, step_idxs)
            mask_ratio_list.append(mask_ratio)
            snr_augweight[ith_snr] = r


        augdata, auglabel, augweight,record_idx = [], [], [],[]
        for i in trange(snr_ithLabels.shape[0]):
            snr_ith = snr_ithLabels[i].item()
            # if maskstatus[snr_ith] > 0:
            step_idxs = self.snrs_mask_idxs[snr_ith]
            # if self.mask_type == 'hard': # soft-> 0,1 with prob.
            #     step_idxs = hard_mask(x=Signals[i, :, :], code=step_idxs)

            new_signal = Signals[i, :, :].mul(step_idxs.to(Signals.device))
            augdata.append(new_signal)
            auglabel.append(Labels[i])
            augweight.append(snr_augweight[snr_ith])
            record_idx.append(i)
            if self.args.num_sampling > 0 and tag != 'inverse':
                for j in range(self.args.num_sampling):
                    unit = torch.ones_like(step_idxs)
                    diff = step_idxs - unit
                    new_step_idxs = diff.mul(torch.rand(unit.size()).to(diff.device)) + unit

                    new_signal = Signals[i, :, :].mul(new_step_idxs.to(Signals.device))
                    augdata.append(new_signal)
                    auglabel.append(Labels[i])
                    augweight.append(snr_augweight[snr_ith])
                    record_idx.append(i)

        self.logger.info('Generating the aug. {} {} with snrs_masks, about {:.2f}% of ori. {} with '.format(set_name, len(record_idx), len(record_idx) / Signals.shape[0] * 100, set_name))
        self.logger.info(f'{set_name}.mask_ratio: {["{:.3f}".format(number) for number in mask_ratio_list]}')
        self.logger.info(f'{set_name}.snr_weight: {["{:.3f}".format(number) for number in snr_augweight.tolist()]}')
        self.logger.info(
            f'Sample weight Max: {max(augweight)}\t Min{min(augweight)}')

        data_sampling_set = torch.cat((Signals, torch.stack(augdata, dim=0)), dim=0)
        aug_Labels = torch.cat((Labels, torch.stack(auglabel, dim=0)), dim=0)
        if tag == 'inverse':
            aug_weight = torch.cat((torch.tensor(augweight),torch.ones(Signals.shape[0])), dim=0)
        else:
            aug_weight = torch.cat((torch.ones(Signals.shape[0]), torch.tensor(augweight)), dim=0)

        return data_sampling_set, aug_Labels, aug_weight

    def load_valset(self,valid_aug = 'non', remove_weight = False, ):
        if valid_aug == 'non': # which means preserve the original valset
            Signals, Labels = self.data_opts.val_set
            ori_weight = torch.ones(Labels.shape[0])
            val_set = Signals, Labels, ori_weight
        elif valid_aug == 'aug': # which means aug the original valset
            val_set = self.aug_dataset(
                self.data_opts.val_set, self.data_opts.val_idx, 'one', set_name='ValSet')
        else:  # selg.validaug == 'ref' # which means replace the original valset with the masking, note that, this method may only contribute in the SNR-Aware conditions.
            Cor_SNRs = list(
                map(lambda x: self.data_opts.SNR_values[x], self.data_opts.val_idx))
            snr_lab_all = [self.sampler.snr_dict[snr] for snr in Cor_SNRs]
            data_sampling_set, Labels = self.SNR_masking_data(
                self.data_opts.val_set, snr_lab_all)
            ori_weight = torch.ones(Labels.shape[0])
            val_set = data_sampling_set, Labels, ori_weight

        if remove_weight:
            Signals, Labels, Weights = val_set
            val_set = Signals, Labels
        return val_set

    def load_fitset(self, cid_hyper=None):
        batch_size = 64 if cid_hyper is None else cid_hyper.batch_size
        if 'snrs_mask_idxs' not in self.dict:
            raise ValueError(
                'Non sampling stage! Please run sampling function first.')

        train_set = self.aug_dataset(data_set=self.data_opts.train_set, data_idx=self.data_opts.train_idx, tag=self.lossweight,set_name='TrainSet')
        val_set = self.load_valset(valid_aug=self.validaug) # in the current exp., this settings is non.

        train_loader = set_dataloader(
            batch_size=batch_size, data_set=train_set)
        val_loader = set_dataloader(batch_size=batch_size, data_set=val_set)

        return train_loader, val_loader

    def snr_testset(self, batch_size, test_set):
        Sample_list = []
        Label_list = []
        Signals, Labels = test_set
        for snr in self.data_opts.snr_envs:
            _, _, idx_i = self.data_opts.snr_slice('test', snr)
            sig_i = Signals[idx_i]
            lab_i = Labels[idx_i]

            num_chunk = int(sig_i.shape[0] / batch_size)

            Sample = torch.chunk(sig_i, num_chunk, dim=0)
            Label = torch.chunk(lab_i, num_chunk, dim=0)

            Sample_list.append(Sample)
            Label_list.append(Label)

        return Sample_list, Label_list

    def result_statue_check(self,):
        tag = False
        if os.path.exists(self.ori_result_path) \
            and os.path.exists(self.refmask_result_path)\
            and os.path.exists(self.val_result_path) \
            and os.path.exists(self.valmask_result_path):
            tag = True
        return tag

    def conduct(self,):
        # self.force_update = force_update
        result_statue = self.result_statue_check()
        if not result_statue:
            from taskRecog.util import lock_mem
            lock_mem(gid=self.args.gid, ocm=self.args.ocm)
            self.logger = self.logger_config(self.model_fit_dir, 'train')
            self.load_data(logger=self.logger)
            self.snrs_mask_idxs = self.masking(logger=self.logger)
            os_makedirs(self.model_pred_dir)
            self.conduct_fit(self.model_hyper, clogger=self.logger)
            result_statue = self.result_statue_check()

        return result_statue

    def conduct_fit(self, model_hyper, clogger=None, finetune=False, xfit_stats=True):
        try:
            if clogger is None:
                clogger = self.logger_config(
                    self.model_fit_dir, 'train')

            model = importlib.import_module(self.model_opts.import_path)
            model = getattr(model, self.model_opts.class_name)
            if self.args.test:
                model_hyper.epochs = 2
            model = model(model_hyper, clogger)
            # cid_hyper = self.load_hyper(clogger)

            # check pretrain file:
            if 'pretraining_file' in model_hyper.dict and model_hyper.pretraining_file is not None and os.path.exists(model_hyper.pretraining_file):
                xfit_stats = False
            else:
                xfit_stats = True

            train_loader, val_loader = None, None
            if xfit_stats:
                train_loader, val_loader = self.load_fitset(model.hyper)
                clogger.critical('Loading training set and validation set.')
                clogger.info(f'Fit batch size: {train_loader.batch_size}')
                clogger.info(f"Train_loader batch: {len(train_loader)}")
                clogger.info(f"Val_loader batch: {len(val_loader)}")
                clogger.critical('>'*40)

            epochs_stats = model.xfit(
                train_loader, val_loader, finetune=finetune, xfit_stats=xfit_stats)
            if epochs_stats is not None and set(['val_loss', 'val_acc', 'train_loss', 'train_acc', 'lr_list']).issubset(epochs_stats.columns):
                loss_dir = os.path.join(self.model_fit_dir, 'loss_curve')
                lossfig_dir = os.path.join(loss_dir, 'figure')
                save_training_process(epochs_stats, plot_dir=lossfig_dir)

            clogger.critical('>'*40+'\nEnd fit.')
            results = self.eval_testset(model, clogger)

            return results
        except:
            clogger.exception(
                '{}\nGot an error on conduction.\n{}'.format('!'*50, '!'*50))
            raise SystemExit()


    def eval_testset(self, model, logger):
        results = Opt()

        model.eval()
        logger.critical('>'*40)
        logger.critical('Evaluation on the original validation set.')
        val_set = self.load_valset(valid_aug='non',remove_weight=True)
        val_loader = set_dataloader(batch_size=model.hyper.batch_size, data_set=val_set)
        _,pre_lab_all, label_all = model.loader_predict(val_loader)
        np.savez(self.val_result_path, pred=pre_lab_all, label=label_all)
        logger.critical(
            'Save result file to the location: {}'.format(self.val_result_path))
        results.val = (deepcopy(pre_lab_all), deepcopy(label_all))

        logger.critical('>'*40)
        logger.critical('Evaluation on the snr-mask validation set.')
        val_set = self.load_valset(valid_aug='ref',remove_weight=True)
        val_loader = set_dataloader(batch_size=model.hyper.batch_size, data_set=val_set)
        _,pre_lab_all, label_all = model.loader_predict(val_loader)
        np.savez(self.valmask_result_path, pred=pre_lab_all, label=label_all)
        results.valmask = (deepcopy(pre_lab_all), deepcopy(label_all))
        logger.critical(
            'Save result file to the location: {}'.format(self.valmask_result_path))


        logger.critical('>'*40)
        logger.critical('Evaluation on the original testing set.')
        test_set = self.sampling_dataset(
            self.data_opts.test_set, self.data_opts.test_idx, 'non')
        test_sample_list, test_lable_list = self.snr_testset(
            model.hyper.batch_size, test_set)
        pre_lab_all, label_all = eval_samples(
            model, test_sample_list, test_lable_list)
        np.savez(self.ori_result_path, pred=pre_lab_all, label=label_all)
        logger.critical(
            'Save result file to the location: {}'.format(self.ori_result_path))
        results.ori = (deepcopy(pre_lab_all), deepcopy(label_all))

        logger.critical('>'*40)
        logger.critical('Evaluation on the snr-masking testing set.')
        test_set = self.sampling_dataset(
            self.data_opts.test_set, self.data_opts.test_idx, 'ref')
        test_sample_list, test_lable_list = self.snr_testset(
            model.hyper.batch_size, test_set)
        pre_lab_all, label_all = eval_samples(
            model, test_sample_list, test_lable_list)
        np.savez(self.refmask_result_path, pred=pre_lab_all, label=label_all)
        logger.critical('Save result file to the location: {}'.format(
            self.refmask_result_path))
        results.refmask = (deepcopy(pre_lab_all), deepcopy(label_all))
        logger.critical('-'*80)
        return results


    def record_result(self,pre_lab_all, label_all, tag, ave_confMax):
        if tag == 'val' or tag == 'valmask':
            acc = accuracy_score(pre_lab_all, label_all)
            return acc
        else:
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

            self.logger.info(f'On the {tag} testing datset:')
            self.logger.info('Overall Accuracy is: {:.2f}%'.format(acc * 100))
            self.logger.info(f'Macro F1-score is: {F1_score:.4f}')
            self.logger.info(f'Kappa Coefficient is: {kappa:.4f}')

            save_dir = os.path.join(self.eval_plot_dir, tag)
            if ave_confMax:
                save_confmat(Confmat_Set, self.data_opts.num_snrs, self.data_opts.classes, save_dir)

            Accuracy_Mods = save_snr_acc(Accuracy_list, Confmat_Set, self.data_opts.num_snrs, self.data_name, self.data_opts.classes.keys(), save_dir)

            tgt_acc_file = os.path.join(self.eval_acc_dir, f'acc.{tag}.npz')
            np.savez(tgt_acc_file, acc_overall = Accuracy_list, acc_mods= Accuracy_Mods)
            self.logger.info('Save accuracy file to the location: {}'.format(tgt_acc_file))

            return acc

    def evaluate(self, elogger=None, xfit_stats=False, ave_confMax=False):
        if elogger is None:
            self.logger = set_logger(os.path.join(self.eval_dir, '_log_', '{}.{}.eval.log'.format(self.data_name, self.model_name)), '{}.{}'.format(
                self.data_name, self.model_name.upper()), self.logger_level)

        if self.data_statue is False:
            self.load_data(logger=self.logger)
            self.snrs_mask_idxs = self.masking(logger=self.logger)

        self.model_eval_dir = os_makedirs(
            os.path.join(self.eval_dir, self.model_name))
        self.eval_acc_dir = os_makedirs(
            os.path.join(self.model_eval_dir, 'accuracy'))
        self.eval_plot_dir = os_makedirs(
            os.path.join(self.model_eval_dir, 'figures'))

        force_update = False if self.args.force_update is False else True

        results = Opt()
        result_statue = self.result_statue_check()
        if force_update is False and result_statue:
            with np.load(self.val_result_path) as data:
                pre_lab_all, label_all = data['pred'], data['label']
                results.val = (deepcopy(pre_lab_all), deepcopy(label_all))
            with np.load(self.ori_result_path) as data:
                pre_lab_all, label_all = data['pred'], data['label']
                results.ori = (deepcopy(pre_lab_all), deepcopy(label_all))
            with np.load(self.valmask_result_path) as data:
                pre_lab_all, label_all = data['pred'], data['label']
                results.valmask = (deepcopy(pre_lab_all), deepcopy(label_all))
            with np.load(self.refmask_result_path) as data:
                pre_lab_all, label_all = data['pred'], data['label']
                results.refmask = (deepcopy(pre_lab_all), deepcopy(label_all))
        else:
            results = self.conduct_fit(self.model_hyper, finetune=False, xfit_stats=xfit_stats)

        pre_lab_all, label_all = results.val
        val_acc = self.record_result(
            pre_lab_all, label_all, 'val', ave_confMax)

        pre_lab_all, label_all = results.valmask
        valmask_acc = self.record_result(
            pre_lab_all, label_all, 'valmask', ave_confMax)

        pre_lab_all, label_all = results.ori
        ori_acc = self.record_result(
            pre_lab_all, label_all, 'ori', ave_confMax)
        pre_lab_all, label_all = results.refmask
        refmask_acc = self.record_result(
            pre_lab_all, label_all, 'refmask', ave_confMax)

        score = Opt()
        score.val_acc= val_acc
        score.valmask_acc =valmask_acc
        score.ori_acc = ori_acc
        score.refmask_acc = refmask_acc
        return score


    def masking(self, logger):
        self.model_opts.tuner.algo = 'Adam'
        self.sampler.env_config(self.data_opts.snr_envs, logger)
        if os.path.exists(self.sampler.save_path):
            snrs_step_mask = torch.load(self.sampler.save_path, map_location = f'cuda:{self.args.gid}')
            assert len(snrs_step_mask) == self.sampler.num_env
            logger.info('-'*80)
            logger.info(f'Read the saved snrs_step_mask from {self.sampler.save_path} successfully!')
        else:
            logger.info('*'*80)
            logger.info(f'Process the sampling with {self.model_opts.tuner.algo} for every step on every env.')

            with tqdm(total = len(self.sampler.env_list), desc='SNR_Env.', mininterval=0.3, colour='red') as pbar:
                snrs_step_mask = []
                for step, snr_list in enumerate(self.sampler.env_list):
                    snr_zone_str = f'{step}thEnv.snr.{snr_list[0]}-{snr_list[-1]}' if len(
                    snr_list) > 1 else f'snr.{snr_list[0]}'
                    tuner_dir = os.path.join(self.model_fit_dir, 'SNR_tuner', snr_zone_str)

                    os_makedirs(tuner_dir)
                    self.model_opts.tuner.dir = tuner_dir
                    tLogger = self.logger_config(
                    dir=tuner_dir, stage='tuning', name=f'{self.model_opts.class_name}.{self.model_opts.tuner.algo}.{snr_zone_str}')

                    step_idxs = self.snr_evoMask(tLogger,snr_list)
                    snrs_step_mask.append(step_idxs)
                    pbar.update(1)

            os_makedirs(self.sampler.local_dir)
            torch.save(snrs_step_mask, self.sampler.save_path)
            logger.info(f'Save the snrs_step_mask to {self.sampler.save_path} !')

        self.sampling_tag = True
        # snrs_step_mask
        for env,  step_idxs in enumerate(snrs_step_mask):
            logger.info(f'Env: {self.sampler.env_list[env]}\t Mask step idxs: {step_idxs}')
        return snrs_step_mask


    def pack_tuningdata(self, snr_list):
        val_sig_list, val_lab_list = [], []
        tra_sig_list, tra_lab_list = [], []
        for snr in snr_list:
            val_sig, val_lab, _ = self.data_opts.snr_slice('val', snr)
            val_sig_list.append(val_sig)
            val_lab_list.append(val_lab)

            tra_sig, tra_lab, _ = self.data_opts.snr_slice('train', snr)
            tra_sig_list.append(tra_sig)
            tra_lab_list.append(tra_lab)

        val_sig_list = torch.cat(val_sig_list, dim=0)
        val_lab_list = torch.cat(val_lab_list, dim=0)

        tra_sig_list = torch.cat(tra_sig_list, dim=0)
        tra_lab_list = torch.cat(tra_lab_list, dim=0)

        func_data = Opt()
        # func_data.logger = logger
        # func_data.merge(self.model_opts,['hyper', 'import_path', 'class_name', 'trainer_module'])
        # func_data.batch_size = 4400 if 'batch_size' not in vars(self.model_opts.tuner) else self.model_opts.tuner.batch_size
        func_data.val_set = (val_sig_list, val_lab_list)
        func_data.train_set = (val_sig_list, val_lab_list)
        return func_data


    def snr_evoMask(self, logger= None, snr_list=None):

        logger.info('*'*50)

        results_pt = os.path.join(self.model_opts.tuner.dir, 'hyper.tuning.pt' )
        if os.path.exists(results_pt):
            logger.info(f'Loading pre-tuning result from {results_pt}')
            maskcode = torch.load(results_pt)
        else:
            func_data = self.pack_tuningdata(snr_list)

            mask_size = (2,128) if self.IQ else (128,)
            mask_model = AddMask(mask_size=mask_size, mask_type=self.mask_type)

            _model = importlib.import_module(self.model_opts.import_path)
            _model = getattr(_model, self.model_opts.class_name)
            model = _model(self.model_hyper, logger)

            model.load_pretraing_file(file_path=self.model_hyper.pretraining_allDatafile)
            logger.info(f'Using pretraining file in the location {self.model_hyper.pretraining_allDatafile}')

            maskHyper = Opt(init=self.model_hyper)
            maskHyper.update(self.model_opts.maskhyper)
            maskHyper.model_fit_dir = self.model_opts.tuner.dir # todo!!!
            # maskHyper.patience = 20 if self.args.test is False else 2
            # maskHyper.epochs = 400 if self.args.test is False else 2
            # maskHyper.lr = 0.015
            # maskHyper.gamma = 0.9
            logger.info(f'Start evolve the input masks by algo: Adam with epochs {maskHyper.epochs}')

            maskTrainer = MaskTrainer(hyper=maskHyper, logger=logger, model=model, mask=mask_model)

            train_loader = set_dataloader(batch_size= 2200, data_set=func_data.train_set)
            val_loader = set_dataloader(batch_size= 2200, data_set=func_data.val_set)

            maskTrainer.loop(train_loader=train_loader, val_loader=val_loader)

            loss_dir = os.path.join(self.model_opts.tuner.dir, 'loss_curve')
            lossfig_dir = os.path.join(loss_dir, 'figure')
            save_training_process(maskTrainer.epochs_stats, plot_dir=lossfig_dir)
            mask_model.load_state_dict(torch.load(maskTrainer.best_file))
            mask_model.eval()
            maskcode = mask_model.get_mask()

            # for i in range(maxCycle):
            results_pt = os.path.join(self.model_opts.tuner.dir, 'hyper.tuning.pt')
            torch.save(maskcode, results_pt)
            logger.info(f'Saving tuning result to {results_pt}')

        logger.info(f'Masks value: {maskcode.tolist()}')

        return maskcode

# class Praticle2fitness:
#     def setup(self, data = None, maxCycle = 100) -> None:
#         self.init_data(data)
#         self.epochs = maxCycle
#         self.acc_list = []
#         self.loss_list = []

#     def fit(self,):
#         self.trainer.before_val_step(logging =False)
#         from models.nn._baseTrainer import AverageMeter
#         import time
#         self.addmask.train()
#         for epoch in range(self.epochs):
#             self.t_s = time.time()

#             lossmeter = AverageMeter()
#             accmeter = AverageMeter()
#             if self.using_lossweight:
#                 for step, (sig_batch, lab_batch, lab_weight) in enumerate(self.trainer.val_loader):

#                     sig_batch = sig_batch.to(self.trainer.hyper.device)
#                     sig_batch = self.addmask(sig_batch)
#                     lab_batch = lab_batch.to(self.trainer.hyper.device)
#                     lab_weight = lab_weight.to(self.trainer.hyper.device)
#                     loss, acc = self.trainer.cal_loss_acc(
#                         sig_batch, lab_batch, lab_weight)
#                     self.optim.zero_grad()
#                     loss.backward()
#                     self.optim.step()
#                     lossmeter.update(loss.item())
#                     accmeter.update(acc)
#             else:
#                 for step, (sig_batch, lab_batch) in enumerate(self.trainer.val_loader):
#                     sig_batch = sig_batch.to(self.trainer.hyper.device)
#                     sig_batch = self.addmask(sig_batch)
#                     lab_batch = lab_batch.to(self.trainer.hyper.device)
#                     loss, acc = self.trainer.cal_loss_acc(sig_batch, lab_batch)
#                     self.optim.zero_grad()
#                     loss.backward()
#                     self.optim.step()
#                     lossmeter.update(loss.item())
#                     accmeter.update(acc)

#             self.logger.info('====> Epoch: {} Time: {:.2f}\tTrain Loss: {:.6E}\tTrain Acc: {:.3f}% '.format(
#             epoch, time.time() - self.t_s, lossmeter.avg, accmeter.avg * 100))

#             self.acc_list.append(accmeter.avg), self.loss_list.append(lossmeter.avg)
#         self.addmask.eval()
#         return self.addmask.string().detach().clone()


#     def init_data(self, data):
#         self.valid_data = data.val_set

#         if len(self.valid_data) == 3:
#             self.using_lossweight = True
#         else:
#             self.using_lossweight = False

#         self.sample_hyper = data.hyper
#         self.import_path = data.import_path
#         self.class_name = data.class_name
#         self.logger = data.logger
#         self.batch_size = data.batch_size

#         self.problem_dim = self.sample_hyper.sig_len



#         _model = importlib.import_module(self.import_path)
#         _model = getattr(_model, self.class_name)
#         sample_model = _model(self.sample_hyper, self.logger)

#         self.logger.critical('Loading Model.')
#         self.logger.critical(f'Model: \n{str(sample_model)}')
#         self.logger.info(">>> Total params: {:.2f}M".format(
#                     sum(p.numel() for p in list(sample_model.parameters())) / 1000000.0))
#         self.logger.critical('Start fit.')

#         trainer = importlib.import_module(data.trainer_module[0])
#         trainer = getattr(trainer, data.trainer_module[1])

#         val_loader = set_dataloader(batch_size= self.batch_size, data_set=self.valid_data)

#         self.logger.critical('Loading training set and validation set.')
#         # self.logger.info(f"Train_loader batch: {len(train_loader)}")
#         self.logger.info(f"Val_loader batch: {len(val_loader)}")
#         self.logger.critical('>'*40)
#         self.logger.info(f'Finding pretraining file in the location {self.sample_hyper.pretraining_allDatafile}')
#         sample_model.load_pretraing_file(file_path=self.sample_hyper.pretraining_allDatafile)

#         self.trainer = trainer(hyper = self.sample_hyper, logger = self.logger)
#         # self.trainer.hyper.patience = self.trainer.hyper.epochs # Actually, in TuningCell, patience does not work as the same as in trainer.
#         self.trainer.model = sample_model.eval()
#         self.trainer.train_loader = None
#         self.trainer.val_loader = val_loader
#         self.trainer.before_train()

#         self.addmask = AddMask(mask_size=(self.problem_dim,))
#         self.addmask.to(self.trainer.hyper.device)
#         self.optim = torch.optim.Adam(self.addmask.parameters(), lr=0.015)
