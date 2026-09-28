"""Attack model configuration for HisarMod2019.1."""

from models.nn.AMC_Net import AMCNet_config
from models.nn.AWN import AWN_config
from models.nn.CTDNN import CTDNN_config
from models.nn.MCDformer import MCDformer_config
from models.nn.MCLDNN import MCLDNN_config
from models.nn.MsmcNet import MsmcNet_config
from models.nn.ResNet import ResNet_config


class amcnet(AMCNet_config):
    def task_modify(self):
        self.hyper.pretraining_file = "checkpoints/HisarMod2019.1/nature/HisarMod2019.1_amcnet.best.pt"


class awn(AWN_config):
    def task_modify(self):
        self.hyper.pretraining_file = "checkpoints/HisarMod2019.1/nature/HisarMod2019.1_awn.best.pt"


class ctdnn(CTDNN_config):
    def task_modify(self):
        self.hyper.pretraining_file = "checkpoints/HisarMod2019.1/nature/HisarMod2019.1_ctdnn.best.pt"


class mcd(MCDformer_config):
    def task_modify(self):
        self.hyper.pretraining_file = "checkpoints/HisarMod2019.1/nature/HisarMod2019.1_mcd.best.pt"


class mcl(MCLDNN_config):
    def task_modify(self):
        self.hyper.pretraining_file = "checkpoints/HisarMod2019.1/nature/HisarMod2019.1_mcl.best.pt"


class msmc(MsmcNet_config):
    def task_modify(self):
        self.hyper.pretraining_file = "checkpoints/HisarMod2019.1/nature/HisarMod2019.1_msmc.best.pt"


class res(ResNet_config):
    def task_modify(self):
        self.hyper.pretraining_file = "checkpoints/HisarMod2019.1/nature/HisarMod2019.1_res.best.pt"
