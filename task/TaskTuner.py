import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import torch

import ray
from task.TaskLoader import Opt
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

import nevergrad as ng

import importlib
from models._comTrainer import Trainer


class HyperTuner(Opt):
    def __init__(self, opts = None, logger= None, subPack = None):
        super().__init__()

        if opts is not None:
            self.merge(opts)

        self.points_to_evaluate = []
        if 'points_to_evaluate' in self.tuner.dict:
            self.points_to_evaluate = self.tuner.points_to_evaluate
            assert len(self.points_to_evaluate) > 0

        self.best_config = Opt()
        
        self.logger = logger
        
        train_loader, val_loader = subPack.load_fitset()
        self.train_data = train_loader
        self.valid_data = val_loader
        self.batch_size = subPack.batch_size
        
        self.metric = 'val_acc'
        if 'num_samples' not in self.tuner.dict:
            self.tuner.num_samples = 20
            if self.algo_name == 'grid':
                self.tuner.num_samples = 1
        
        if 'resource' not in self.tuner.dict:
            self.resource = {
            "cpu": 10, 
            "gpu": 1  # set this for GPUs
        }
        else:
            self.resource = self.tuner.resource
            
        self.loss_lower_bound = 0
        self.algo_init()
                  
    def algo_init(self,):
        if 'algo' not in self.tuner.dict:
            self.algo_name = 'rand'
        elif self.tuner.algo not in ['bayes','tpe','pso', 'rand', 'grid']:
            raise ValueError('Non supported tuning algo: {}'.format(self.tuner.algo))
        else:
            self.algo_name = self.tuner.algo
        
        if self.algo_name == 'bayes':
            self.tuner.name = 'Bayes_Search'
            self.algo_func = ConcurrencyLimiter(AxSearch(metric=self.metric, mode='min',verbose_logging = False), max_concurrent=6)
            
        elif self.algo_name == 'tpe':
            # Tree-structured Parzen Estimator https://docs.ray.io/en/master/tune/examples/optuna_example.html
            self.tuner.name = 'TPE_Search'
            self.algo_func =  ConcurrencyLimiter(
            OptunaSearch(
                metric=self.metric, mode='min',points_to_evaluate=self.points_to_evaluate
                ), 
            max_concurrent=6
            )
            
        elif self.algo_name == 'pso':
            # https://github.com/facebookresearch/nevergrad
            self.tuner.name = 'PSO_Search'
            _popsize= min((20, self.tuner.num_samples // 10))
            self.algo_func = NevergradSearch(
            optimizer=ng.optimizers.ConfiguredPSO(
                popsize= _popsize
                ),
            metric=self.metric,
            mode="min",
            points_to_evaluate=self.points_to_evaluate
            )
        elif self.algo_name == 'rand' or self.algo_name == 'grid':
            self.tuner.name = 'Rand_Search'   
            self.algo_func = BasicVariantGenerator(max_concurrent=4)

    def once_sample(self,):
        config = Opt(init=self.tuning)
        for key in self.tuning.dict:
            config.dict[key] = self.tuning.dict[key].sample()
        
        _hyper = Opt()
        _hyper.merge(self.hyper)
        _hyper.update(config) # Using ignore_unk will be very risky
        model = importlib.import_module(self.import_path)
        model = getattr(model, self.class_name)
        model = model(_hyper, self.logger)
        fit_info = model.xfit(self.train_data, self.valid_data,)
        t_acc, v_acc = fit_info.train_acc.max(), fit_info.val_acc.max()   
        
        metric_dict = {
            'tra_acc': t_acc,
            'val_acc': v_acc,
        }
        self.best_result = metric_dict[self.metric]
        self.best_config.merge(config)
        # self.logger.info("Best config is:", self.best_config.dict)
        return self.best_config

    def conduct(self,):
        if self.tuner.num_samples == 1 and self.algo_name == 'rand':
            self.best_config = self.once_sample()
        else:
            self.best_config = self._conduct()
        
        return self.best_config    

    def _conduct(self,):
        
        func_data = Opt()
        func_data.logger = self.logger
        func_data.merge(self,['hyper', 'import_path', 'class_name', 'trainer_path', 'trainer_name' ,'train_data', 'valid_data'])
        
        ray.init(num_cpus=30)
        sched = ASHAScheduler()
        # self.tuner.num_samples = 80
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(TuningCell, data=func_data), 
                resources=self.resource),
            param_space=self.tuning.dict,
            tune_config=
            tune.TuneConfig(
            # name=self.algo_name,
            search_alg=self.algo_func,
            # resources_per_trial=self.resource,
            metric=self.metric,
            mode="max",
            num_samples=self.tuner.num_samples,
            scheduler=sched,
            # verbose=1,
            # raise_on_failed_trial = False
            ),
            run_config=RunConfig(
                name=self.algo_name,
                storage_path=self.tuner.dir,
                verbose=1,
                failure_config=FailureConfig(max_failures=self.tuner.num_samples // 2),
                stop={'training_iteration':100},
                checkpoint_config=CheckpointConfig(
                    checkpoint_frequency=3,
                    checkpoint_at_end = True
                ),
                sync_config=tune.SyncConfig(
                    syncer=None
                )                
            )
        )
        
        results = tuner.fit() 
            
        df = results.get_dataframe()
        df.to_csv(os.path.join(self.tuner.dir, '{}.trial.csv'.format(self.algo_name)))
        ray.shutdown()
        
        best_result = results.get_best_result(self.metric, 'min')
        self.best_config.merge(best_result.config)
        self.best_result = best_result.metrics
        # self.logger.info("Best config is:", self.best_config.dict)
        
        return self.best_config    
    

class TuningCell(tune.Trainable):
    '''
    Trainable class of tuning stochastic neural network
    see https://docs.ray.io/en/latest/tune/examples/tune-pytorch-cifar.html#the-train-function
    '''
    def setup(self, config, data=None):
        self.train_data = data.train_data
        self.valid_data = data.valid_data
        self.base_hyper = data.hyper
        self.import_path = data.import_path
        self.class_name = data.class_name
        self.logger = data.logger
        
        _hyper = Opt()
        _hyper.merge(self.base_hyper)
        _hyper.update(config) # Using ignore_unk will be very risky
        
        self.sample_hyper = _hyper
        _model = importlib.import_module(self.import_path)
        _model = getattr(_model, self.class_name)
        sample_model = _model(self.sample_hyper, self.logger)
        
        trainer = importlib.import_module(data.trainer_path)
        trainer = getattr(trainer, data.trainer_name)
        
        self.trainer = trainer(sample_model, self.train_data, self.valid_data, self.base_hyper, self.logger)
        self.trainer.cfg.patience = self.trainer.cfg.epochs # Actually, in TuningCell, patience does not work as the same as in trainer.
        self.trainer.before_train()
    
    def step(self,):
        self.trainer.before_train_step()
        _, t_acc = self.trainer.run_train_step()
        self.trainer.after_train_step()
        self.trainer.before_val_step()
        _, v_acc = self.trainer.run_val_step()
        self.trainer.after_val_step()
        self.trainer.iter += 1      
        
        return {
            'tra_acc': t_acc,
            'val_acc': v_acc,
        }
    
    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        torch.save(self.sample_model.state_dict(), checkpoint_path)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        self.sample_model.load_state_dict(torch.load(checkpoint_path))        