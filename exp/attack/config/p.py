"""Attack model configuration for Panoradio.HF."""

from models.nn.AMC_Net import AMCNet_config
from models.nn.AWN import AWN_config
from models.nn.CTDNN import CTDNN_config
from models.nn.MCDformer import MCDformer_config
from models.nn.MCLDNN import MCLDNN_config
from models.nn.MsmcNet import MsmcNet_config
from models.nn.ResNet import ResNet_config


class amcnet(AMCNet_config):
    def task_modify(self):
        self.hyper.pretraining_file = "checkpoints/Panoradio.HF/nature/Panoradio.HF_amcnet.best.pt"


class awn(AWN_config):
    def task_modify(self):
        self.hyper.pretraining_file = "checkpoints/Panoradio.HF/nature/Panoradio.HF_awn.best.pt"


class ctdnn(CTDNN_config):
    def task_modify(self):
        self.hyper.pretraining_file = "checkpoints/Panoradio.HF/nature/Panoradio.HF_ctdnn.best.pt"


class mcd(MCDformer_config):
    def task_modify(self):
        self.hyper.pretraining_file = "checkpoints/Panoradio.HF/nature/Panoradio.HF_mcd.best.pt"


class mcl(MCLDNN_config):
    def task_modify(self):
        self.hyper.pretraining_file = "checkpoints/Panoradio.HF/nature/Panoradio.HF_mcl.best.pt"


class msmc(MsmcNet_config):
    def task_modify(self):
        self.hyper.pretraining_file = "checkpoints/Panoradio.HF/nature/Panoradio.HF_msmc.best.pt"


class res(ResNet_config):
    def task_modify(self):
        self.hyper.pretraining_file = "checkpoints/Panoradio.HF/nature/Panoradio.HF_res.best.pt"
