"""Example launcher for the MVRG attack."""

import os

from taskAttack.Parser import get_parser
from taskAttack.Wrapper import Attack


MVRG_DEFAULTS = {
    "algo": "mvrg",
    "data": "a",
    "snr": [10],
    "surrogate_model": "mcd",
    "target_model": "mcd",
    "bound": "psr",
    "psr": -10,
    "batch_size": 512,
    "attackset": "attack",
}

# Common correctly-classified evaluation set used by the MVRG experiments.
MVRG_ATTACK_SET_DEFAULTS = {
    "data": "b",
    "models": ("amcnet", "awn", "ctdnn", "mcd", "mcl", "msmc", "res"),
    "snr_range": (4, 25),
    "samples_per_class": 100,
    "data_tag": "test",
    "batch_size": 512,
    "seed": 2022,
}


def main():
    parser = get_parser(parsing=False)
    parser.set_defaults(**MVRG_DEFAULTS)
    args = parser.parse_args()
    args.exp_name = os.path.join("paper", "mvrg", args.data)
    Attack(args, parser).conduct()


if __name__ == "__main__":
    main()
