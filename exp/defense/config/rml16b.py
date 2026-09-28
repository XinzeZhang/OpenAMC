from models.nn.AWN import AWN_config
from models.nn.AMC_Net import AMCNet_config
from models.nn.MCLDNN import MCLDNN_config
from models.nn.CLDNN import CLDNN_config
from models.nn.ResNet import ResNet_config
from models.nn.VT_CNN import VTCNN2_config
from models.nn.CTDNN import CTDNN_config
from models.nn.MCDformer import MCDformer_config


from taskDefense.advTraining.FTML.defense_net import AWN_config as AWN_v1_config
from models.nn._baseNet import BaseNetConfig

class awn(AWN_config): pass
class amcnet(AMCNet_config): pass
class mcl(MCLDNN_config): pass
class cldnn(CLDNN_config): pass
class res(ResNet_config): pass
class vtcnn(VTCNN2_config): pass
class ctdnn(CTDNN_config): pass
class mcd(MCDformer_config): pass

# from taskRecog.dataAug.advTraining.advTrainer import AWN_Trainer_transmi,common_Trainer_transmi
# class awn(AWN_config):
#     def task_modify(self):
#         pass
        # self.hyper.epochs = 5
        # self.hyper.patience = 2
        # self.trainer_module = ('taskRecog/dataAug/advTraining/advTrainer.py', 'AWN_Trainer_transmi')
        # self.hyper.pretraining_file = 'data/RML2016.10a/pre_AT_models/RML2016.10a_awn.abest.pt'
        # self.hyper.transform_method='None'
        # self.hyper.surrogate='None'



# class awn_v1(AWN_v1_config):
#     def task_modify(self):
#         pass
#         # self.hyper.epochs = 5 # only for unit test
#         # self.hyper.patience = 2 # only for unit test

class awn_v2(AWN_v1_config):
    def ablation_modify(self):
        self.class_name = 'AWN_V2'

# class amcnet(AMCNet_config):
#     def task_modify(self):
#         # self.trainer_module = ('taskRecog/dataAug/advTraining/advTrainer.py', 'common_Trainer_transmi')
#         self.hyper.pretraining_file = 'data/RML2016.10a/pre_AT_models/RML2016.10a_amcnet.abest.pt'
        # self.hyper.transform_method='None'
        # self.hyper.surrogate='None'
