import os
import torch
import numpy as np
from tqdm.auto import trange
from taskRecog.util import Opt, set_dataloader,os_makedirs
from taskRecog.featureselection.snr_aware.softmaskTuner import maskTuner
from taskRecog.dataAug.SoftMask.HardAugWrapper import WeightAugMaskTask as BaseWeightAugMaskTask
from taskRecog.dataAug.SoftMask.HardAugWrapper import AugP2f

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
        # for i, t in enumerate(Maskcode):
        #     self.logger.info(f'inputMasks_{i}: {t}')
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
        self.problem_dims = func_data.hyper.sig_len
        self.func_data = func_data
        self.save_dir = save_dir  # to use
        # self.tuning_dict = tuning_dict
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
                i_solution = np.ones(self.problem_dims)
                i_solution[selected_idx] = random.uniform(0, 2)
                all_pop.append(i_solution)

        assert len(all_pop) == self.pop_size
        starting_solutions = np.stack(all_pop, axis=0)
        return starting_solutions

    def evolve(self,):
        def fitness_function(solution):
            config = {}
            for i in range(self.problem_dims):
                state_i = solution[i]
                config['inputMask_{}'.format(i)] = state_i

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
                "ub": [2 for i in range(self.problem_dims)],
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
                state_i = best_solution[i]
                config.dict[f'inputMask_{i}'] = state_i

            results_pt = os.path.join(self.save_dir, 'hyper.tuning.pt')
            torch.save(config, results_pt)
            self.logger.info(f'Saving tuning result to {results_pt}')

        Maskcode = [config.dict[f'inputMask_{i}']
                    for i in range(self.problem_dims)]

        return config, Maskcode