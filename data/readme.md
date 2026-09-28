# Attack Set Summary

Generated at: 2026-04-08 18:16:55

Attack sets are constructed from samples that every model in a reader-supplied
model list classifies correctly. The corresponding checkpoints must be available
under `checkpoints/<dataset>/nature/`, or supplied explicitly with repeated
`--checkpoint MODEL=PATH` arguments.

The defaults are owned by `exp/attack/mvrg.py`, not the shared parser. They are also the CLI defaults: RML2016.10b, models `amcnet awn ctdnn mcd mcl msmc res`, SNR bounds 4--25 (selecting the available 4--18 SNRs), 100 samples per class, test split, seed 2022, and batch size 512. CUDA is opt-in, so the shortest GPU command is `uv run python -m taskAttack.make_attackset -cuda -gid 0`. The fully explicit command is:

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

This writes `data/postdata/attack.RML2016.10b_dict.pt` and adjacent JSON metadata.
See **Construct the common attack set** in the repository main `README.md`
for checkpoint overrides, output selection, and commands for other datasets.

## Attack. Dataset Index
| Dataset          | SNRs           | # SNRS | Labels | # per  SNR per Label | # Total |
| :-: | :-: | :--: | :-: | :-: | :-: |
| MIMO.Nt4Nr2   | 4,5, ..., 20   |     17 | 6      | 3                    | 306     |
| MIMO.Nt16Nr4  | 4,5, ..., 20   |     17 | 6      | 40                   | 4080    |
| MIMO.Nt64Nr16 | 4,5, ..., 20   |     17 | 6      | 60                   | 6120    |
| HisarMod2019.1   | 4,6, ..., 18   |      8 | 26     | 5                    | 1040    |
| Panoradio.HF     | 10, 15, 20, 25 |      4 | 18     | 5                    | 360     |
| RML2016.10a      | 4,6, ..., 18   |      8 | 11     | 3                    | 264     |
| RML2016.10b      | 4,6, ..., 18   |      8 | 10     | 100                  | 8000    |

## Ori. Dataset Index
|           Dataset            |       Original Scale       | Split ratio | AWN training  time (m) |
| :--------------------------: | :------------------------: | :---------: | :--------------------: |
| MIMO_Nt4Nr2MIMO_Nt4Nr2 |    93K 128, 93000x2x128    |    6:2:2    |          1.6           |
|       MIMO_Nt16Nr4        |    93K 128, 93000x2x128    |    6:2:2    |          1.4           |
|       MIMO_Nt64Nr16       |    93K 128, 93000x2x128    |    6:2:2    |          2.3           |
|         Panoradio_HF         |  173K 2048, 172800x2x2048  |    6:2:2    |          37.2          |
|        HisarMod2019_1        |  780K 1024, 780000x2x1024  |    6:2:2    |          78.4          |
|         RML2016.10a          |  220K, 128, 220000x2x128   |    6:2:2    |          3.2           |
|         RML2016.10b          | 1200K, 128 , 1200000x2x128 |    6:2:2    |          15.1          |
|         RML2018.01a          |  2556K 1024, 2555904x2x1024       |    6:2:2    |         154.3          |
|         RML2022.01a          |  462K 128, 462000x2x128       |    6:2:2    |               |
|        ACMR          |  300K 256, 300000x2x128       |    6:2:2    |            |
|        RML24          |  1323K 2048, 1323000x2x2048       |    6:2:2    |            |


Dataset

https://panoradio-sdr.de/overview-of-open-datasets-for-rf-signal-classification/



https://github.com/ThalesGroup/pythagore-mod-reco


RML24
https://github.com/yiwawa/RML24-Cognitive-Radio-for-Satellite



RML22.01A:
https://github.com/venkateshsathya/RML22


RML14
https://github.com/yiwawa/RML24-Cognitive-Radio-for-Satellite

CommRad RF: A dataset of communication radio signals for detection, identification and classification
https://zenodo.org/records/14192970

SP2025
https://github.com/coulsonlee/Robust-ViT-for-AMR-SP2025/tree/main



MIMO
https://github.com/Richardzhangxx/AMR-Dataset-for-MIMO-system-with-precoding/tree/main

@article{ZHANG2022103650, title={Deep Learning Based Automatic Modulation Recognition: Models, Datasets, and Challenges}, author={Fuxin Zhang and Chunbo Luo and Jialang Xu and Yang Luo and FuChun Zheng}, journal={Digital Signal Processing}, year={2022}, doi = {https://doi.org/10.1016/j.dsp.2022.103650} }



Tang, Z., Luo, C., Yin, Y., Luo, Y., 2026. ACMR: an automatic composite-modulation recognition dataset and baselines. IEEE Trans. Veh. Technol. 1–11. https://doi.org/10.1109/TVT.2026.3656598
