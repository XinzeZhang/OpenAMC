import os
import torch
from taskRecog.util import Opt, set_dataloader, os_makedirs, os_rmdirs, set_logger
from taskRecog.featureselection.snr_aware.softmaskTuner import maskTuner, Praticle2fitness, PO
from taskRecog.featureselection.snr_aware.softmaskWrapper import SoftMaskTask
from taskRecog.util import save_training_process
import numpy as np
from tqdm.auto import tqdm, trange
from copy import deepcopy
import random
from taskRecog.dataAug.SoftMask.HardAugWrapper import eval_samples

class WeightAugMaskTask(SoftMaskTask):
    def __init__(self, args):
        super().__init__(args)
        self.masktask_config(args)

    def masktask_config(self, args):  # masktask_config(self, args, snrmodel):
        self.ori_result_path = os.path.join(
            self.model_pred_dir, 'results.ori.npz')
        self.val_result_path = os.path.join(
            self.model_pred_dir, 'results.val.npz')
        self.refmask_result_path = os.path.join(
            self.model_pred_dir, 'results.refmask.npz')

        # self.snr_num_classes = args.num_env

        assert 'validaug' in vars(args)
        # assert args.validaug in ['pred', 'ref', 'non', 'aug']
        assert args.validaug in ['ref', 'non', 'aug']
        self.validaug = args.validaug

        assert 'lossweight' in vars(args)
        assert args.lossweight in ['one', 'minus', 'plus', 'mratio']
        self.lossweight = args.lossweight

        assert 'perexample' in vars(args) # in True or False
        self.perexample = args.perexample

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

    def step_masking(self, prob_idxs):
        step_idxs = [0 if random.random() < prob else 1 for prob in prob_idxs]
        return step_idxs

    def SNR_masking_data(self, data_set, snr_lab_all):
        Signals, Labels = data_set
        data_sampling_set = Signals.detach().clone()
        assert len(snr_lab_all) == data_sampling_set.shape[0]
        self.logger.info(
            f'Masking the dataset {Signals.size()} with snrs_masks')
        for i in trange(len(snr_lab_all)):
            pred_snr_id = snr_lab_all[i]
            prob_idxs = self.snrs_mask_idxs[pred_snr_id]
            step_idxs = self.step_masking(prob_idxs)
            data_sampling_set[i, :, :] = Signals[i, :, :].mul(
                torch.tensor(step_idxs).float())

        return data_sampling_set, Labels

    def load_valset(self, remove_weight = False):
        if self.validaug == 'non':
            Signals, Labels = self.data_opts.val_set
            ori_weight = torch.ones(Labels.shape[0])
            val_set = Signals, Labels, ori_weight
        elif self.validaug == 'aug':
            val_set = self.aug_trainset(
                self.data_opts.val_set, self.data_opts.val_idx, 'one')
        else:  # selg.validaug == 'ref'
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

        train_set = self.aug_trainset(
            self.data_opts.train_set, self.data_opts.train_idx, self.lossweight)
        val_set = self.load_valset()

        train_loader = set_dataloader(
            batch_size=batch_size, data_set=train_set)
        val_loader = set_dataloader(batch_size=batch_size, data_set=val_set)

        return train_loader, val_loader

    def loss_calweight(self, mask_ratio):
        if self.lossweight == 'mratio':
            r = mask_ratio
        elif self.lossweight == 'plus':
            r = 1 + mask_ratio
        elif self.lossweight == 'minus':
            r = 1 - mask_ratio
        else:
            r = 1

        return r


    def aug_trainset(self, data_set, data_idx, tag='one'):
        '''Aug. trainset with weight
        '''
        assert tag in ['one', 'minus', 'plus', 'mratio']

        maskstatus = []
        for prob_idxs in self.snrs_mask_idxs:
            mask_ratio = sum(prob_idxs) / self.data_opts.sig_len
            r = self.loss_calweight(mask_ratio)

            if mask_ratio == 0:
                maskstatus.append(0)
            else:
                maskstatus.append(r)

        Cor_SNRs = map(lambda x: self.data_opts.SNR_values[x], data_idx)
        Cor_SNRs = list(Cor_SNRs)
        ref_lab_all = [self.sampler.snr_dict[snr] for snr in Cor_SNRs]

        ref_lab_all = torch.tensor(ref_lab_all)

        Signals, Labels = data_set

        assert len(ref_lab_all) == Signals.shape[0]

        augdata, auglabel, augweight = [], [], []

        record_idx = []
        for i in trange(ref_lab_all.shape[0]):

            ref_snr_id = ref_lab_all[i].item()
            if maskstatus[ref_snr_id] > 0:
                prob_idxs = self.snrs_mask_idxs[ref_snr_id]
                step_idxs = self.step_masking(prob_idxs)
                new_signal = Signals[i, :, :].mul(
                    torch.tensor(step_idxs).float())
                augdata.append(new_signal)
                auglabel.append(Labels[i])
                if self.perexample:
                    i_maskratio = 1 -  sum(step_idxs) / self.data_opts.sig_len
                    i_r = self.loss_calweight(i_maskratio)
                    augweight.append(i_r)
                else:
                    augweight.append(maskstatus[ref_snr_id])
                record_idx.append(i)

        self.logger.info('Generating the aug. dataset {} with snrs_masks, about {:.2f}% of ori. trainset'.format(
            len(record_idx), len(record_idx) / Signals.shape[0] * 100))
        self.logger.info(
            f'Sample weight Max: {max(augweight)}\t Min{min(augweight)}')

        augdata = torch.stack(augdata, dim=0)
        auglabel = torch.stack(auglabel, dim=0)
        augweight = torch.tensor(augweight)

        ori_weight = torch.ones(ref_lab_all.shape[0])

        aug_weight = torch.cat([ori_weight, augweight], dim=0)

        data_sampling_set = torch.cat([Signals, augdata], dim=0)
        aug_Labels = torch.cat([Labels, auglabel], dim=0)

        return data_sampling_set, aug_Labels, aug_weight

    def snr_evoMask(self, snr_list=[0]):
        snr_zone_str = f'snr.{snr_list[0]}-{snr_list[-1]}' if len(
            snr_list) > 1 else f'snr.{snr_list[0]}'
        tuner_dir = os.path.join(self.model_fit_dir, 'SNR_tuner', snr_zone_str)

        os_makedirs(tuner_dir)
        self.model_opts.tuner.dir = tuner_dir
        tLogger = self.logger_config(
            dir=tuner_dir, stage='tuning', name=f'{self.model_opts.class_name}.{self.model_opts.tuner.algo_name}.{snr_zone_str}')

        snr_Tuner = augmaskTuner(
            self.model_opts, logger=tLogger, data_opts=self.data_opts, snr_list=snr_list)
        prob_idxs = snr_Tuner.meal_conduct()

        return prob_idxs

    def result_statue_check(self,):
        tag = False
        if os.path.exists(self.ori_result_path) \
            and os.path.exists(self.refmask_result_path)\
            and os.path.exists(self.val_result_path):
            tag = True
        return tag

    def conduct(self, force_update=None, finetune=False):
        if force_update is not None:
            if force_update in [True, False]:
                self.force_update = force_update
            else:
                raise ValueError(
                    'force_update parameter is incorrect, please set with True or False.')
        result_statue = self.result_statue_check()

        self.logger = self.logger_config(self.model_fit_dir, 'train')
        self.load_data(logger=self.logger)

        self.snrs_mask_idxs = self.masking(logger=self.logger)

        if not result_statue or self.force_update:
            if result_statue:
                os_rmdirs(self.model_pred_dir)
            os_makedirs(self.model_pred_dir)
            self.conduct_fit(clogger=self.logger, finetune=finetune)
            self.fit_statue = True

        score = self.evaluate(elogger=self.logger)
        return score

    def conduct_fit(self, clogger=None, finetune=False, xfit_stats=True):
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

            model = self.model_import()
            model = model(cid_hyper, clogger)
            clogger.critical('Loading Model.')
            clogger.critical(f'Model: \n{str(model)}')

            # if self.model_opts.arch != 'torch_nn':
            clogger.info(">>> Total params: {:.2f}M".format(
                    sum(p.numel() for p in list(model.parameters())) / 1000000.0))

            clogger.critical('Start fit.')
            epochs_stats = model.xfit(
                train_loader, val_loader, finetune=finetune, xfit_stats=False)
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
        logger.critical('Evaluation on the validation set.')
        val_set = self.load_valset(remove_weight=True)
        test_sample_list, test_lable_list = self.snr_testset(
            model.hyper.batch_size, val_set)
        pre_lab_all, label_all = eval_samples(
            model, test_sample_list, test_lable_list)
        np.savez(self.val_result_path, pred=pre_lab_all, label=label_all)
        logger.critical(
            'Save result file to the location: {}'.format(self.val_result_path))
        results.val = (deepcopy(pre_lab_all), deepcopy(label_all))

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
        logger.critical('Evaluation on the ref_masking testing set.')
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
            with np.load(self.refmask_result_path) as data:
                pre_lab_all, label_all = data['pred'], data['label']
                results.refmask = (deepcopy(pre_lab_all), deepcopy(label_all))
        else:
            results = self.conduct_fit(finetune=False, xfit_stats=xfit_stats)

        pre_lab_all, label_all = results.val
        val_acc = self.record_result(
            pre_lab_all, label_all, 'val', ave_confMax)
        pre_lab_all, label_all = results.ori
        ori_acc = self.record_result(
            pre_lab_all, label_all, 'ori', ave_confMax)
        pre_lab_all, label_all = results.refmask
        refmask_acc = self.record_result(
            pre_lab_all, label_all, 'refmask', ave_confMax)

        score = Opt()
        score.val_acc= val_acc
        score.ori_acc = ori_acc
        score.refmask_acc = refmask_acc
        return score


class augmaskTuner(maskTuner):
    def meal_conduct(self,):
        # tuner_algo_dir = os.path.join(self.tuner.dir, self.algo_name)
        func_data = self.pack_data()

        pop_size = self.tuner.pop_size if 'pop_size' in vars(
            self.tuner) else 20
        maxCycle = self.tuner.maxCycle if 'maxCycle' in vars(
            self.tuner) else 10

        self.logger.info('*'*50)
        self.logger.info(
            f'Start evolve the input masks by algo: {self.tuner.algo_name} with pop_size {pop_size} and maxCycle {maxCycle}')
        swarm = AugPO(pop_size=pop_size, maxCycle=maxCycle,
                      func_data=func_data, save_dir=self.tuner.dir)

        config, Maskcode = swarm.evolve()
        self.logger.info(f'Masks prob.: {Maskcode}')
        return Maskcode

    def pack_data(self):
        val_sig_list, val_lab_list = [], []
        for snr in self.snr_list:
            val_sig, val_lab, _ = self.data_opts.snr_slice('val', snr)
            val_sig_list.append(val_sig)
            val_lab_list.append(val_lab)

        val_sig_list = torch.cat(val_sig_list, dim=0)
        val_lab_list = torch.cat(val_lab_list, dim=0)
        func_data = Opt()
        func_data.logger = self.logger
        func_data.merge(self, ['hyper', 'import_path',
                        'class_name', 'trainer_module', 'batch_size'])
        # func_data.train_set = (None, None)
        func_data.val_set = (val_sig_list, val_lab_list,
                             torch.ones(val_lab_list.shape[0]))
        return func_data


class AugPO():
    def __init__(self, pop_size=20, maxCycle=20, func_data=None, save_dir=None, metric='val_acc', patience=10, algo='clpso'):
        self.pop_size = pop_size
        self.maxCycle = maxCycle

        self.problem_dims = func_data.hyper.sig_len
        self.func_data = func_data
        self.save_dir = save_dir  # to use
        self.logger = func_data.logger

        self.peval = AugP2f()
        self.peval.setup(data=self.func_data)

        self.fitness_metric = metric
        self.patience = patience
        self.algo = algo
        self.starting_perturb = [0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1, 0]

    def create_starting_solutions(self,):
        '''return starting_solutions, shape: (pop_size, problem_dim)'''
        from random import sample
        from math import ceil
        import random

        def chunk_into_n(lst, n):
            size = ceil(len(lst) / n)
            return list(
                map(lambda x: lst[x * size:x * size + size],
                    list(range(n)))
            )

        num_div = len(self.starting_perturb)
        pop_list = list(range(self.pop_size))
        init_list = chunk_into_n(lst=pop_list, n=num_div)

        idx_list = [i for i in range(self.problem_dims)]

        all_pop = []
        for i_pop_list, p_ratio in zip(init_list, self.starting_perturb):
            for i in i_pop_list:
                num_p = int(self.problem_dims * p_ratio)
                selected_idx = sample(idx_list, num_p)
                i_solution = np.zeros(self.problem_dims)
                i_solution[selected_idx] = random.uniform(0, 1)
                all_pop.append(i_solution)

        assert len(all_pop) == self.pop_size
        starting_solutions = np.stack(all_pop, axis=0)
        return starting_solutions

    def evolve(self,):
        def fitness_function(solution):
            config = {}
            for i in range(self.problem_dims):
                prob = solution[i]
                mask_i = 0 if random.random() < prob else 1
                config['inputMask_{}'.format(i)] = mask_i

            self.peval.reset_config(config)
            results = self.peval.step()
            fitness = results[self.fitness_metric]

            return fitness

        results_pt = os.path.join(self.save_dir, 'hyper.tuning.pt')
        if os.path.exists(results_pt):
            self.logger.info(f'Loading pre-tuning result from {results_pt}')
            config = torch.load(results_pt)
        else:
            problem = {
                "fit_func": fitness_function,
                "lb": [0 for i in range(self.problem_dims)],
                "ub": [1 for i in range(self.problem_dims)],
                "minmax": "max",
                'verbose': False,
            }

            term_dict = {
                # after 30 epochs, if the global best doesn't improve then we stop the program
                "max_early_stop": self.patience
            }

            if self.algo == 'clpso':
                from mealpy.swarm_based import PSO
                algo_func = PSO.CL_PSO(
                    epoch=self.maxCycle, pop_size=self.pop_size)

            elif self.algo == 'aro':
                from mealpy.swarm_based import ARO
                algo_func = ARO.OriginalARO(
                    epoch=self.maxCycle, pop_size=self.pop_size)

            s_solutions = self.create_starting_solutions()
            algo_func.solve(problem, termination=term_dict,
                            starting_positions=s_solutions)
            best_solution = algo_func.solution[0]
            config = Opt()

            for i in range(self.problem_dims):
                prob_i = best_solution[i]
                config.dict[f'prob_{i}'] = prob_i

            results_pt = os.path.join(self.save_dir, 'hyper.tuning.pt')
            torch.save(config, results_pt)
            self.logger.info(f'Saving tuning result to {results_pt}')

        Maskcode = [config.dict[f'prob_{i}']
                    for i in range(self.problem_dims)]

        return config, Maskcode

class AugP2f(Praticle2fitness):
    def step(self,):
        self.trainer.before_val_step(logging=False)
        for step, (sig_batch, lab_batch, lab_weight) in enumerate(self.trainer.val_loader):
            with torch.no_grad():
                sig_batch = sig_batch.to(self.trainer.hyper.device)
                lab_batch = lab_batch.to(self.trainer.hyper.device)
                lab_weight = lab_weight.to(self.trainer.hyper.device)
                loss, acc = self.trainer.cal_loss_acc(
                    sig_batch, lab_batch, lab_weight)
                self.trainer.val_acc.update(acc)

        v_acc = self.trainer.val_acc.avg
        return {'val_acc': v_acc, 'best_val_acc': v_acc * 100}

    def reset_config(self, new_config):
        _config = Opt()
        _config.merge(new_config, ['inputMask_{}'.format(i)
                      for i in range(self.sample_hyper.sig_len)])
        inputMask = [_config.dict['inputMask_{}'.format(
            i)] for i in range(self.sample_hyper.sig_len)]
        val_set = mask_float(self.valid_data, inputMask)
        val_loader = set_dataloader(
            batch_size=self.batch_size, data_set=val_set)
        self.trainer.val_loader = val_loader
        return True


def mask_float(data_set, mask_op):
    '''
    data_set:(data, label)\n
    data:  torch.tensor (sample, dims, steps)\n
    im_op: list lens== steps
    '''
    data, label, lab_weight = data_set
    _data = data.detach().clone()

    for id, v in enumerate(mask_op):
        _data[:, :, id] = _data[:, :, id].detach().clone() * v

    new_data_set = (_data, label, lab_weight)
    return new_data_set
