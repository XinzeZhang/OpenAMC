"""Example launcher for the ICASSP 2025 Fading-Invariant Method (FIM)."""

import os

from taskAttack.Parser import get_parser
from taskAttack.channelAttack.fim import FIMAttack


FIM_DEFAULTS = {
    "algo": "pgd",
    "data": "a",
    "snr": [10],
    "surrogate_model": "mcd",
    "target_model": "mcd",
    "bound": "psr",
    "psr": -10,
    "batch_size": 500,
    "nim_model": "nim",
    "nim_seed": 2024,
    "nim_num_channels": 200,
    "nim_epochs": 20,
    "nim_patience": 50,
    "nim_batch_size": 256,
    "channel_seed": 8000,
    "num_channel": 30,
    "pilot_seed": 3030,
    "pilot_n_channel": 200,
}


def main():
    parser = get_parser(parsing=False)
    parser.set_defaults(**FIM_DEFAULTS)
    args = parser.parse_args()
    args.exp_name = os.path.join("paper", "fim", args.data)
    FIMAttack(args, parser).conduct()


if __name__ == "__main__":
    main()
