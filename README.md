# OpenAMC

OpenAMC is a PyTorch framework for automatic modulation recognition (AMC),
adversarial attacks, and adversarial defenses. This release contains the code
for three research works:

- **FIM — Fading-Invariant Adversarial Attacks on Neural Modulation
  Recognition**, published at **ICASSP 2025**. FIM uses a Neural Inverse Model
  (NIM) and transmit-power estimation to improve adversarial attacks under
  random wireless channel effects.
- **MVRG — Transferable Adversarial Attacks on Automatic Modulation
  Classification via Multi-Variance-Reduced Gradients**, submitted to
  **ICASSP 2027**. Authors: Kun He, Gao Liu, and Xinze Zhang. MVRG combines
  Nesterov look-ahead gradients, neighborhood variance tuning, and temporal
  Gaussian smoothing to improve transferability across AMC architectures.
- **TFI — Time-Frequency Interactive Cross-Architecture Transfer Attack Method for Automatic Modulation Classification**, submitted to **TIFS**. TFI combines Multi-scale Gradient Generalization (MGG) and Shrinkage-Shift Regularization (SSR) to improve transferability across heterogeneous AMC models.

Throughout this repository, the ICASSP 2025 method is named **FIM** and its
learned inverse component is named **NIM**, matching the published paper.

### Papers

- Xinze Zhang, Dengao Zhu, Xiyao Dong, and Kun He, “Fading-Invariant
  Adversarial Attacks on Neural Modulation Recognition,” *Proceedings of the
  IEEE International Conference on Acoustics, Speech and Signal Processing
  (ICASSP)*, 2025. DOI: [10.1109/ICASSP49660.2025.10890846](https://doi.org/10.1109/ICASSP49660.2025.10890846).
- Kun He, Gao Liu, and Xinze Zhang, “MVRG: Transferable Adversarial Attacks on
  Automatic Modulation Classification via Multi-Variance-Reduced Gradients,”
  submitted to *ICASSP 2027*.
- Kun He, Gao Liu, Xinze Zhang, and Shuo Zhang, “TFI: Time-Frequency Interactive Cross-Architecture Transfer Attack Method for Automatic Modulation Classification,” submitted to *TIFS*.

## 1. Installation

Python 3.10--3.13 is supported. We recommend `uv`:

```bash
git clone https://github.com/XinzeZhang/OpenAMC.git
cd OpenAMC
pip install uv
uv sync
```

You can replace `uv run python` in the commands below with the Python executable
from another environment that contains the dependencies in `pyproject.toml`.

## 2. Download data and checkpoints

Large datasets and model checkpoints are not stored in Git.

> Temporary download links -- replace these before the final release:
>
> - Processed datasets: [OpenAMC postdata (temporary)](https://example.com/openamc/postdata.tar.gz)
> - Pretrained checkpoints: [OpenAMC checkpoints (temporary)](https://example.com/openamc/checkpoints.tar.gz)

After downloading and extracting them, the expected layout is:

```text
OpenAMC/
├── data/
│   └── postdata/
│       ├── RML2016.10a_dict.split.pt
│       └── RML2016.10b_dict.split.pt
└── checkpoints/
    ├── RML2016.10a/
    │   └── nature/
    │       ├── RML2016.10a_mcd.best.pt
    │       ├── RML2016.10a_awn.best.pt
    │       └── ...
    └── RML2016.10b/
        └── nature/
            └── ...
```

For development or testing, use symbolic links instead of copying large files:

```bash
ln -s /absolute/path/to/downloaded/postdata data/postdata
ln -s /absolute/path/to/downloaded/checkpoints checkpoints
```

Both paths are ignored by Git. Confirm the links before running an experiment:

```bash
test -f data/postdata/RML2016.10a_dict.split.pt
test -f checkpoints/RML2016.10a/nature/RML2016.10a_mcd.best.pt
```

## 3. Dataset and model names

Common dataset identifiers are:

| CLI name | Dataset |
| --- | --- |
| `a` | RML2016.10a |
| `b` | RML2016.10b |
| `c` | RML2016.04c |
| `h` | HisarMod2019.1 |
| `p` | Panoradio.HF |
| `dr2` | MIMO.Nt4Nr2 |
| `dr4` | MIMO.Nt16Nr4 |
| `dr16` | MIMO.Nt64Nr16 |
| `rml18a` | RML2018.01a |

Common model identifiers include `mcd`, `awn`, `amcnet`, `ctdnn`, `mcl`,
`res`, and `vtcnn`. The complete registries are in `data/__init__.py` and
`models/__init__.py`. Dataset-specific checkpoint paths are configured in
`exp/attack/config/`.

### Included attacks

The public attack registry is intentionally limited to methods evaluated in the released papers:

- FIM: `fgm` and `pgd`, plus the channel-independent `pca` and `vae` baselines. “Vanilla” in the FIM paper means transmitting the unmodified receiver-side perturbation and is not a separate attack class.
- MVRG: `bim`, `mi`, `ni`, `vtmi`, `vtni`, `sfaa`, `feig`, `fciaa`, `pngd`, `mdam`, and `mvrg`.
- TFI uses the same ten reported baselines and is registered as `tfi`.

The registry keys follow the paper names; in particular, `vtmi`, `vtni`, and `fciaa` correspond to VT-MI, VT-NI, and FCIAA. Implementations unrelated to these papers are not distributed.

## 4. Construct the common attack set

Transfer attacks should be compared on the same inputs. OpenAMC can construct
a balanced attack set containing only signals that every model in a supplied
model list classifies correctly. Selection is performed independently at each
SNR, then balanced to the same number of samples for every modulation class and
SNR condition.

The MVRG reproduction and attack-set defaults are defined in `exp/attack/mvrg.py`; the shared `taskAttack/Parser.py` remains method-agnostic. Running the generator without construction arguments uses these paper-aligned defaults: RML2016.10b, models `amcnet awn ctdnn mcd mcl msmc res`, SNR bounds 4--25 (the available RML2016.10b SNRs are 4--18), 100 samples per class, the test split, seed 2022, and batch size 512. CUDA remains opt-in. Thus, the shortest GPU command is:

```bash
uv run python -m taskAttack.make_attackset -cuda -gid 0
```

The same settings can be written explicitly as:

```bash
uv run python -m taskAttack.make_attackset \
  -data b \
  --models amcnet awn ctdnn mcd mcl msmc res \
  --snr-range 4 18 \
  --samples-per-class 100 \
  --data-tag test \
  -batch_size 512 \
  -cuda -gid 0
```

The default outputs are:

```text
data/postdata/attack.RML2016.10b_dict.pt
data/postdata/attack.RML2016.10b_dict.pt.json
```

The `.pt` file contains a dictionary keyed by SNR. Each value is a tuple of
`(signals, labels, original_indices)` and is directly consumable by OpenAMC.
The adjacent JSON file records the model list, SNRs, classes, sample count, and
random seed used to construct it.

Configured checkpoints under `checkpoints/<dataset>/nature/` are loaded by
default. A checkpoint can be overridden for any model, including models with
custom checkpoint locations:

```bash
uv run python -m taskAttack.make_attackset \
  -data b --models awn mcd --snr-range 4 18 \
  --checkpoint awn=/absolute/path/to/awn.pt \
  --checkpoint mcd=/absolute/path/to/mcd.pt \
  --samples-per-class 100 -cuda -gid 0
```

Use a constructed attack set by adding `-attackset attack` to an MVRG or other
standard attack command:

```bash
uv run python -m exp.attack.mvrg \
  -data b -snr 4 18 -attackset attack \
  -surrogate_model awn -target_model mcd \
  -bound psr -psr -10 -cuda -gid 0
```

For the HisarMod2019.1 setting in the MVRG manuscript, use `-data h`, SNR range
`4 18`, and `--samples-per-class 5`.

For a TFI attack set, select the TFI profile. It uses all seven paper models, a cap of 100 samples per SNR/class cell, and rounds the common cell size down to a multiple of five. Panoradio.HF starts at 10 dB; the other five paper datasets start at 4 dB.

```bash
uv run python -m taskAttack.make_attackset --profile tfi -data b -cuda -gid 0
```

Use `-data h`, `-data p`, `-data dr2`, `-data dr4`, or `-data dr16` to construct the corresponding paper attack set.

## 5. Run MVRG

MVRG is the method from the manuscript **“MVRG: Transferable Adversarial
Attacks on Automatic Modulation Classification via Multi-Variance-Reduced
Gradients,” submitted to ICASSP 2027**. It is implemented in
`taskAttack/attackmethods/gradient/mvrg.py` and registered under the attack
name `mvrg`. It uses the ordinary attack wrapper; no propagation channel is
modeled for MVRG perturbations.

### Standard run

```bash
uv run python -m exp.attack.mvrg \
  -data a \
  -snr 10 \
  -surrogate_model mcd \
  -target_model mcd \
  -bound psr \
  -psr -10 \
  -batch_size 512 \
  -cuda -gid 0
```

Useful options:

- `-snr 10`: attack one SNR condition.
- `-snr 0 18`: attack every available SNR in the inclusive range.
- `-snr all`: attack all SNR conditions in the processed dataset.
- `-bound psr -psr -10`: constrain perturbation-to-signal ratio to -10 dB.
- `-bound pnr -pnr 0`: constrain perturbation-to-noise ratio instead.
- `-surrogate_ckp PATH` and `-target_ckp PATH`: override configured checkpoints.

The manuscript settings are 30 update iterations, momentum decay 1.0, 20
neighboring samples, Gaussian scale 2.0, and randomly selected odd kernel
widths from `{3, 5, ..., 63}`. These values can be overridden programmatically
through `Attack.conduct(..., algo_configs={...})`.

### Fast MVRG smoke test

This verifies the full data/model/attack path with one update and one neighbor:

```python
import sys

sys.argv = [
    "mvrg-smoke", "-data", "a", "-snr", "10",
    "-surrogate_model", "mcd", "-target_model", "mcd",
    "-algo", "mvrg", "-bound", "psr", "-psr", "-10",
    "-batch_size", "12000", "-cuda", "-gid", "0", "-test",
]

from taskAttack.Parser import get_parser
from taskAttack.Wrapper import Attack

args, parser = get_parser()
Attack(args, parser).conduct(
    eval=False,
    algo_configs={
        "epoch": 1,
        "num_neighbor": 1,
        "window_sizes": [11],
        "alpha": 0.01,
    },
)
```

## 6. Run TFI

TFI is the method from **“TFI: Time-Frequency Interactive Cross-Architecture Transfer Attack Method for Automatic Modulation Classification.”** It is implemented in `taskAttack/attackmethods/gradient/tfi.py` and registered as `tfi`. The launcher keeps all paper-specific defaults in `exp/attack/tfi.py`, leaving the shared parser method-agnostic.

TFI uses 10 iterations with step size `epsilon / 10`. The dataset-specific momentum, SSR shrinkage, MGG scale count, and scale interval are selected automatically by the launcher.

```bash
uv run python -m exp.attack.tfi \
  -data b -snr all -attackset attack \
  -surrogate_model ctdnn -target_model awn \
  -bound psr -psr -10 -batch_size 512 -cuda -gid 0
```

A reduced smoke run can use the ordinary test split and one SNR; programmatic callers may override `epoch` through `Attack.conduct(..., algo_configs={"epoch": 1})`.

## 7. Run FIM

FIM is the method from **“Fading-Invariant Adversarial Attacks on Neural
Modulation Recognition,” published at ICASSP 2025**. It is implemented in
`taskAttack/channelAttack/fim.py`. It wraps any existing receiver-side attack,
such as FGM or PGD, then:

1. Generates receiver-side adversarial perturbations.
2. Samples fading channels and constructs `(delta, delta / h)` pairs.
3. Trains or loads the Neural Inverse Model (NIM).
4. Uses pilot-based transmit-power estimation when `-rebound` is enabled.
5. Evaluates clean, ideal received, channel-faded, and FIM perturbations.

The NIM implementation is in
`taskAttack/channelAttack/neural_inverse_model.py`. The ordinary attack and
defense wrappers are unchanged; channel effects are used only by the FIM/channel
wrapper.

### Standard FIM run with PGD

```bash
uv run python -m exp.attack.fim \
  -data a \
  -snr 10 \
  -algo pgd \
  -surrogate_model mcd \
  -target_model mcd \
  -bound psr \
  -psr -10 \
  -batch_size 500 \
  -nim_model nim \
  -nim_seed 2024 \
  -nim_num_channels 200 \
  -nim_epochs 20 \
  -nim_patience 50 \
  -nim_batch_size 256 \
  -channel_seed 8000 \
  -num_channel 30 \
  -pilot_seed 3030 \
  -pilot_n_channel 200 \
  -rebound \
  -cuda -gid 0
```

To reproduce the FGM-based setting, change `-algo pgd` to `-algo fgm`.

Important FIM/channel options:

| Option | Meaning | Default |
| --- | --- | --- |
| `-nim_num_channels` | Sampled channel states used to build NIM pairs | `200` |
| `-nim_epochs` | Maximum NIM training epochs | `20` |
| `-num_channel` | Random channel realizations used for evaluation | `30` |
| `-rebound` | Enable transmit-power estimation/amplification | off |
| `-use_cpls` | Combined path-loss and shadowing channel | on |
| `-use_rayleigh` | Rayleigh fading channel | on |
| `-channel_seed` | First evaluation-channel seed | `2024` |
| `-pilot_seed` | First power-estimation pilot seed | `3030` |

### Fast FIM smoke test

Use a small number of sampled/evaluation channels and one NIM epoch:

```bash
uv run python -m exp.attack.fim \
  -data a -snr 10 -algo fgm \
  -surrogate_model mcd -target_model mcd \
  -bound psr -psr -10 -batch_size 12000 \
  -nim_num_channels 1 -nim_epochs 1 -nim_patience 1 \
  -nim_batch_size 12000 -num_channel 1 -pilot_n_channel 1 \
  -rebound -cuda -gid 0 -test
```

FIM saves the NIM checkpoint under the experiment output tree. A repeated run
with the same settings loads that checkpoint automatically.

## 8. Outputs

Normal runs write under `yield_results/`; runs with `-test` write under
`yield_test/`. Both are ignored by Git. Important files include:

```text
yield_results/<experiment>/<dataset>/surrogate/<model>/<attack>_results/
├── result.snr<SNR>.pt
├── target.<model>/
│   ├── evaluation.log
│   ├── clean/
│   ├── adv.rece/
│   ├── adv.trans/
│   └── adv.fim/
└── NIM.<model>/snr<SNR>/
```

For ordinary TFI or MVRG evaluation, the target directory contains clean and
adversarial accuracy tensors and logs. For FIM, `adv.rece` is the ideal attack
without channel effects, `adv.trans` is the uncorrected over-the-air attack,
and `adv.fim` is the FIM result.

## 9. Troubleshooting

- **Checkpoint load fails:** verify the filename and dataset directory under
  `checkpoints/<dataset>/nature/`, or pass `-surrogate_ckp`/`-target_ckp`.
- **Dataset file is missing:** verify the `data/postdata` link and the processed
  filename expected by `data/RML.py`.
- **CUDA out of memory:** lower `-batch_size`; FIM also benefits from lowering
  `-nim_batch_size`.
- **A quick run takes too long:** use the smoke-test settings, especially one
  SNR, one iteration for TFI, one neighboring sample for MVRG, or one channel/NIM epoch for FIM.
- **Reproducing a result:** keep attack, NIM, pilot, and evaluation channel seeds
  fixed and record the exact checkpoint files used.
