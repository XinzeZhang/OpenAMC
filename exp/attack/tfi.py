"""Paper-aligned launcher for the TFI attack."""

import os

from taskAttack.Parser import get_parser
from taskAttack.Wrapper import Attack


TFI_DEFAULTS = {
    "algo": "tfi",
    "data": "b",
    "snr": ["all"],
    "surrogate_model": "ctdnn",
    "target_model": "awn",
    "bound": "psr",
    "psr": -10,
    "batch_size": 512,
    "attackset": "attack",
}

# Common correctly-classified evaluation set used by the TFI experiments.
TFI_ATTACK_SET_DEFAULTS = {
    "data": "b",
    "models": ("amcnet", "awn", "ctdnn", "mcd", "mcl", "msmc", "res"),
    "snr_range": (4, 25),
    "samples_per_class": 100,
    "data_tag": "test",
    "batch_size": 512,
    "seed": 2022,
    "round_to": 5,
}

TFI_ALGO_CONFIGS = {
    "b": {"decay": 0.75, "shrinkage": 0.65, "scale_num": 6, "scale_interval": 3},
    "h": {"decay": 0.75, "shrinkage": 0.65, "scale_num": 6, "scale_interval": 3},
    "p": {"decay": 0.55, "shrinkage": 0.48, "scale_num": 3, "scale_interval": 2},
    "dr2": {"decay": 0.65, "shrinkage": 0.95, "scale_num": 3, "scale_interval": 1},
    "dr4": {"decay": 0.65, "shrinkage": 0.90, "scale_num": 3, "scale_interval": 1},
    "dr16": {"decay": 0.65, "shrinkage": 0.90, "scale_num": 3, "scale_interval": 1},
}

TFI_SNR_RANGES = {
    "b": (4, 25), "h": (4, 25), "p": (10, 25),
    "dr2": (4, 25), "dr4": (4, 25), "dr16": (4, 25),
}


def main():
    parser = get_parser(parsing=False)
    parser.set_defaults(**TFI_DEFAULTS)
    args = parser.parse_args()
    args.exp_name = os.path.join("paper", "tfi", args.data)
    if args.data not in TFI_ALGO_CONFIGS:
        raise ValueError(f"No paper-aligned TFI defaults for {args.data!r}.")
    Attack(args, parser).conduct(algo_configs=TFI_ALGO_CONFIGS[args.data])


if __name__ == "__main__":
    main()
