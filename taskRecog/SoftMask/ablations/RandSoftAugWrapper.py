import os
import torch
import numpy as np
from tqdm.auto import trange
from taskRecog.util import Opt, set_dataloader,os_makedirs
from taskRecog.featureselection.snr_aware.softmaskTuner import maskTuner
from taskRecog.dataAug.SoftMask.ablations.RandHardAugWrapper import WeightAugMaskTask as BaseWeightAugMaskTask
from taskRecog.dataAug.SoftMask.ablations.RandHardAugWrapper import AugP2f
import random

class WeightAugMaskTask(BaseWeightAugMaskTask):

    def SNR_masking_data(self, data_set, snr_lab_all):
        Signals, Labels = data_set
        data_sampling_set = Signals.detach().clone()
        assert len(snr_lab_all) == data_sampling_set.shape[0]
        self.logger.info(
            f'Masking the dataset {Signals.size()} with snrs_masks')
        for i in trange(len(snr_lab_all)):
            pred_snr_id = snr_lab_all[i]
            mask_ratio = self.snrs_mask_idxs[pred_snr_id]
            step_idxs = [random.uniform(0, 2) if random.random() < mask_ratio else 1 for _ in range(self.data_opts.sig_len) ]
            data_sampling_set[i, :, :] = Signals[i, :, :].mul(
                torch.tensor(step_idxs).float())

        return data_sampling_set, Labels

    def aug_trainset(self, data_set, data_idx, tag='one'):
        '''Aug. trainset with weight
        '''
        assert tag in ['one', 'minus', 'plus', 'mratio']

        maskstatus = []
        for mask_ratio in self.snrs_mask_idxs:
            # mask_ratio = 1 - sum(step_idxs) / self.data_opts.sig_len
            if tag == 'mratio':
                r = mask_ratio
            elif tag == 'plus':
                r = 1 + mask_ratio
            elif tag == 'minus':
                r = 1 - mask_ratio
            else:
                r = 1

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
                mask_ratio = self.snrs_mask_idxs[ref_snr_id]
                step_idxs = [random.uniform(0, 2) if random.random() < mask_ratio else 1 for _ in range(self.data_opts.sig_len) ]
                new_signal = Signals[i, :, :].mul(
                    torch.tensor(step_idxs).float())
                augdata.append(new_signal)
                auglabel.append(Labels[i])
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
        snr_Tuner = softmaskTuner(
            self.model_opts, logger=tLogger, data_opts=self.data_opts, snr_list=snr_list)
        mask_ratio = snr_Tuner.meal_conduct()
        return mask_ratio


class softmaskTuner(maskTuner):
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
        swarm = SoftPO(pop_size=pop_size, maxCycle=maxCycle,
                      func_data=func_data, save_dir=self.tuner.dir)

        _, p_ratio = swarm.evolve()
        self.logger.info(f'Masks Prob.: {p_ratio}')
        return p_ratio

    def pack_data(self):
        # train_sig, train_label, _ = self.data_opts.snr_slice('train', snr)
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


class SoftPO():
    def __init__(self, pop_size=20, maxCycle=20, func_data=None, save_dir=None, metric='val_acc', patience=20, algo='clpso'):
        self.pop_size = pop_size
        self.maxCycle = maxCycle
        self.sig_len = func_data.hyper.sig_len
        self.problem_dims = 1
        self.func_data = func_data
        self.save_dir = save_dir  # to use
        # self.tuning_dict = tuning_dict
        self.logger = func_data.logger
        self.peval = AugP2f()
        self.peval.setup(data=self.func_data)
        self.fitness_metric = metric
        self.patience = patience
        self.algo = algo

    def evolve(self,):
        def fitness_function(solution):
            config = {}

            p_ratio = solution[0]
            step_idxs = [random.uniform(0, 2) if random.random() < p_ratio else 1 for _ in range(self.sig_len) ]

            for i, tag in enumerate(step_idxs):
                config['inputMask_{}'.format(i)] = tag

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
                "lb": [0],
                "ub": [1],
                "minmax": "max",
                'verbose': False,
            }

            term_dict = {
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

            algo_func.solve(problem, termination=term_dict)
            best_solution = algo_func.solution[0]

            config = Opt()

        return config, best_solution[0]