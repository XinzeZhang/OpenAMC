import os
import torch
import numpy as np
from tqdm.auto import trange
from taskRecog.util import Opt, set_dataloader,os_makedirs
from taskRecog.featureselection.snr_aware.softmaskTuner import maskTuner
from taskRecog.dataAug.SoftMask.ablations.RandHardAugWrapper import WeightAugMaskTask as BaseWeightAugMaskTask
from taskRecog.dataAug.SoftMask.ablations.RandHardAugWrapper import AugP2f

class WeightAugMaskTask(BaseWeightAugMaskTask):
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
        step_idxs = snr_Tuner.meal_conduct()
        return step_idxs


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

        config, Maskcode = swarm.evolve()
        self.logger.info(f'Masks value: {Maskcode}')
        return Maskcode

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
        from random import sample
        import random
        def fitness_function(solution):
            config = {}
            idx_list = [i for i in range(self.sig_len)]
            p_ratio = solution[0]
            num_p = int(self.sig_len * p_ratio)
            selected_idx = sample(idx_list, num_p)

            for i in idx_list:
                if i in selected_idx:
                    config['inputMask_{}'.format(i)] = random.uniform(0, 2)
                else:
                    config['inputMask_{}'.format(i)] = 1

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

            idx_list = [i for i in range(self.sig_len)]
            num_p = int(self.sig_len * best_solution)
            if num_p == self.sig_len:
                num_p = self.sig_len - 1
            selected_idx = sample(idx_list, num_p)

            config = Opt()

            for i in idx_list:
                if i in selected_idx:
                    config.dict[f'inputMask_{i}'] = random.uniform(0, 2)
                else:
                    config.dict[f'inputMask_{i}'] = 1

            results_pt = os.path.join(self.save_dir, 'hyper.tuning.pt')
            torch.save(config, results_pt)
            self.logger.info(f'Saving tuning result to {results_pt}')

        Maskcode = [config.dict[f'inputMask_{i}']
                    for i in range(self.sig_len)]

        return config, Maskcode