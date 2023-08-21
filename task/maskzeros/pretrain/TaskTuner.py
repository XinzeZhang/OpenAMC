import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import torch
import torch.utils.data as Data

import ray
from task.base.TaskLoader import Opt
# from ray.tune.suggest.bohb import TuneBOHB
# from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.schedulers import ASHAScheduler
from ray.tune.schedulers import PopulationBasedTraining 
# https://arxiv.org/pdf/1711.09846.pdf.
from ray.tune.schedulers.pb2 import PB2 
# pip install GPy sklearn
# https://arxiv.org/abs/2002.02518 (NIPS 2020)
from ray import tune
from ray.air import session, FailureConfig
from ray.air.config import RunConfig
from ray.air.config import CheckpointConfig
# from ray.air.checkpoint import Checkpoint

from ray.tune.search.nevergrad import NevergradSearch
from ray.tune.search.ax import AxSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune import ExperimentAnalysis

import nevergrad as ng

import importlib

from task.util import os_rmdirs,set_logger,set_fitset
from task.base.TaskTuner import HyperTuner
from task.maskzeros.TaskLoader import mask_zeros
from random import sample

def trial_str_creator(trial):
    return "{}_{}".format(trial.trainable_name, trial.trial_id)

class maskTuner(HyperTuner):
    def __init__(self, opts = None, logger= None, subPack = None):
        super().__init__(opts, logger, subPack)

    def gen_p2e(self,):
        config = {}
        for key in self.tuning.dict.keys():
            config[key] = self.hyper.dict[key]
            
        self.points_to_evaluate = [config]
        
        idx = [i for i in range(self.hyper.sig_len)]
        tags = ['inputMask_{}'.format(i) for i in range(self.hyper.sig_len)]
        
        for mr in [0.05, 0.1, 0.15, 0.2]:
            num = int(self.hyper.sig_len * mr)
            for t in range(2):
                selected_idx = sample(idx, num)
                new_config = config.copy()
                for s_idx in selected_idx:
                    new_config['inputMask_{}'.format(s_idx)] = 0
                
                self.points_to_evaluate.append(new_config)

                

    def _conduct(self,):
        
        func_data = Opt()
        # func_data.logger = self.logger
        func_data.merge(self,['hyper', 'import_path', 'class_name', 'trainer_module'])
        func_data.merge(self.subPack, ['train_set', 'val_set'])
        
        # ray.init(num_cpus=self.tuner.num_cpus)
        os.environ['RAY_COLOR_PREFIX'] = '1'
        ray.init()
        sched = ASHAScheduler(time_attr='training_iteration', max_t=self.tuner.max_training_iteration, grace_period= self.tuner.min_training_iteration) if self.using_sched else None
        # self.tuner.num_samples = 80
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(maskTuningCell, data=func_data), 
                resources=self.resource),
            param_space=self.tuning.dict,
            tune_config=
            tune.TuneConfig(
            search_alg=self.algo_func,
            metric='best_val_acc',
            mode="max",
            num_samples=self.tuner.num_samples,
            scheduler=sched,
            trial_name_creator=trial_str_creator,
            trial_dirname_creator=trial_str_creator,
            # name=self.algo_name,
            # resources_per_trial=self.resource,
            # verbose=1,
            # raise_on_failed_trial = False
            ),
            run_config=RunConfig(
                name=self.algo_name,
                storage_path=self.tuner.dir,
                verbose=1,
                failure_config=FailureConfig(max_failures=self.tuner.num_samples // 2),
                stop={'training_iteration':self.tuner.max_training_iteration,
                    #   'stop_counter': self.hyper.patience
                      },
                checkpoint_config=CheckpointConfig(
                    num_to_keep=3,
                    checkpoint_score_attribute ='val_acc',
                    checkpoint_score_order='max',
                    checkpoint_frequency= 1
                )      
            )
        )
        
        results = tuner.fit() 

        df = results.get_dataframe()
        df.to_csv(os.path.join(self.tuner.dir, '{}.trial.csv'.format(self.algo_name)))
        ray.shutdown()
        
        best_result = results.get_best_result(self.metric, 'max', scope='all')
        self.best_config.merge(best_result.config)
        self.best_result = best_result.metrics
        self.best_checkpoint_path = os.path.join(best_result.checkpoint.path, 'model.pth')
      
    

class maskTuningCell(tune.Trainable):
    '''
    Trainable class of tuning stochastic neural network
    see https://docs.ray.io/en/latest/tune/examples/tune-pytorch-cifar.html#the-train-function
    '''
    def setup(self, config, data=None):
        self.train_data = data.train_set
        self.valid_data = data.val_set
        self.base_hyper = data.hyper
        self.import_path = data.import_path
        self.class_name = data.class_name
        # self.logger = data.logger
        
        logger_path = os.path.join(self.logdir, '{}.log'.format(self.class_name))
        self.logger = set_logger(logger_path, f'{self.class_name}.tuning', rewrite=False)
        
        
        _hyper = Opt()
        _hyper.merge(self.base_hyper)
        _hyper.update(config) # Using ignore_unk will be very risky
        
        
        self.sample_hyper = _hyper
        self.num_workers = 0 if 'num_workers' not in _hyper.dict else  _hyper.num_workers
        
        for i in range(_hyper.sig_len):
            if 'inputMask_{}'.format(i) not in _hyper.dict:
                raise ValueError('Missing hyper config: "inputMask_{}"!'.format(i))
            
        inputMask = [_hyper.dict['inputMask_{}'.format(i)] for i in range(_hyper.sig_len)]
        
        train_set = mask_zeros(data.train_set, inputMask)
        val_set = mask_zeros(data.val_set, inputMask)
        
        train_loader, val_loader = set_fitset(batch_size=_hyper.batch_size, train_set=train_set, val_set=val_set)
        
        self.logger.critical('Loading training set and validation set.')
        self.logger.info(f"Train_loader batch: {len(train_loader)}")
        self.logger.info(f"Val_loader batch: {len(val_loader)}")
        self.logger.critical('>'*40)
        
        _model = importlib.import_module(self.import_path)
        _model = getattr(_model, self.class_name)
        sample_model = _model(self.sample_hyper, self.logger)
        
        self.logger.critical('Loading Model.')
        self.logger.critical(f'Model: \n{str(sample_model)}')
        self.logger.info(">>> Total params: {:.2f}M".format(
                    sum(p.numel() for p in list(sample_model.parameters())) / 1000000.0))
        self.logger.critical('Start fit.')
            
        trainer = importlib.import_module(data.trainer_module[0])
        trainer = getattr(trainer, data.trainer_module[1])
        
        self.logger.info(f'Finding pretraining file in the location {self.sample_hyper.pretraining_file}')
        sample_model.load_pretraing_file(file_path=self.sample_hyper.pretraining_file)
        
        self.trainer = trainer(sample_model, train_loader, val_loader, self.base_hyper, self.logger)
        # self.trainer.cfg.patience = self.trainer.cfg.epochs # Actually, in TuningCell, patience does not work as the same as in trainer.
        self.trainer.before_train()
    
    def step(self,):
        self.trainer.before_val_step()
        for step, (sig_batch, lab_batch) in enumerate(self.trainer.val_loader):
            with torch.no_grad():
                sig_batch = sig_batch.to(self.trainer.cfg.device)
                lab_batch = lab_batch.to(self.trainer.cfg.device)

                loss, acc = self.trainer.cal_loss_acc(sig_batch, lab_batch)
                self.trainer.val_acc.update(acc)
        
        v_acc = self.trainer.val_acc.avg
        self.logger.info('Val Acc: {:.3f}%'.format(v_acc * 100 ))
        
        return {
            'val_acc': v_acc,
            'best_val_acc': v_acc * 100, 
        }
    
    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        torch.save(self.trainer.model.state_dict(), checkpoint_path)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        self.trainer.model.load_state_dict(torch.load(checkpoint_path))        
        

   