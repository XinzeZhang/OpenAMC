from taskDefense.advTraining.AMD import Adversarial_Multi_Distillation
from taskDefense.advTraining.EH_AT import EH_AT
from taskDefense.advTraining.PGD_AT import PGD_AT
from taskDefense.advTraining.PIAT import PIAT
from taskDefense.advTraining.TRADES import TRADES
from models.nn._baseTrainer import AnnealingTrainer
from taskDefense.advTraining.FRE.trainer import ftmlTrainer


# defense_zoo = dict(
#     nature = ('models.nn._baseTrainer', 'AnnealingTrainer'),
#     pgdat = ('taskDefense.advTraining.PGD_AT', 'PGD_AT'),
#     trades = ('taskDefense.advTraining.TRADES', 'TRADES'),
#     amd = ('taskDefense.advTraining.AMD', 'Adversarial_Multi_Distillation'),
#     fre = ('taskDefense.advTraining.FRE.trainer', 'ftmlTrainer'),
#     piat = ('taskDefense.advTraining.PIAT', 'PIAT'),
#     ehat = ('taskDefense.advTraining.EH_AT', 'EH_AT'),
#     # FTML2 = ('taskDefense.advTraining.FTML.trainerv2', 'ftmlTrainer'),
#     # FTML0 = ('taskDefense.advTraining.FTML.trainerv0', 'ftmlTrainer'),
# )

defense_zoo = dict(
    nature = dict(trainer = AnnealingTrainer),
    pgdat = dict(trainer = PGD_AT),
    trades = dict(trainer = TRADES),
    amd = dict(trainer = Adversarial_Multi_Distillation),
    fre = dict(trainer = ftmlTrainer),
    piat = dict(trainer = PIAT),
    ehat = dict(trainer = EH_AT),
)


def load_defense_class(defense_name):
    if defense_name not in defense_zoo:
        raise Exception('Unspported defense algorithm {}'.format(defense_name))
    return defense_zoo[defense_name]['trainer']

__version__ = '1.0.0'