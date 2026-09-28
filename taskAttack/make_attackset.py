"""Construct a balanced attack set jointly classified correctly by models.

The saved ``.pt`` file is consumed by the attack wrapper with
``-attackset attack``. For every selected SNR and modulation class, it contains
the same number of signals that every model in the supplied list classifies
correctly.
"""

import json
import os
from dataclasses import dataclass

import torch

from exp.attack.mvrg import MVRG_ATTACK_SET_DEFAULTS
from taskAttack.Parser import get_parser
from taskAttack.Wrapper import Attack
from taskRecog.util import set_dataloader


@dataclass
class JointlyCorrectSamples:
    signals: torch.Tensor
    labels: torch.Tensor
    indices: tuple
    counts: dict


def parse_checkpoint_overrides(values):
    """Parse repeated ``MODEL=PATH`` checkpoint overrides."""
    overrides = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(
                f"Invalid checkpoint override {value!r}; expected MODEL=PATH."
            )
        model, path = value.split("=", 1)
        if not model or not path:
            raise ValueError(
                f"Invalid checkpoint override {value!r}; expected MODEL=PATH."
            )
        overrides[model] = path
    return overrides


def predict(model, signals, labels, batch_size):
    loader = set_dataloader(
        data_set=(signals, labels), batch_size=batch_size, shuffle=False
    )
    _, predictions, returned_labels = model.loader_predict(loader)
    if len(predictions) != len(labels):
        raise ValueError(
            f"Prediction count {len(predictions)} does not match "
            f"sample count {len(labels)}."
        )
    if not torch.equal(returned_labels.cpu(), labels.cpu()):
        raise ValueError("The prediction loader changed the sample order.")
    return predictions.cpu()


def jointly_correct_samples(models, signals, labels, indices, batch_size):
    """Return samples classified correctly by every model."""
    shared_mask = torch.ones(len(labels), dtype=torch.bool)
    cpu_labels = labels.cpu()
    for model_name, model in models.items():
        predictions = predict(model, signals, labels, batch_size)
        shared_mask &= predictions == cpu_labels
        print(
            f"{model_name}: {int((predictions == cpu_labels).sum())}/"
            f"{len(labels)} correct; {int(shared_mask.sum())} jointly correct"
        )

    selected = torch.where(shared_mask)[0]
    selected_labels = labels[selected]
    unique_labels, label_counts = torch.unique(
        selected_labels, return_counts=True
    )
    counts = {
        int(label): int(count)
        for label, count in zip(unique_labels.tolist(), label_counts.tolist())
    }
    return JointlyCorrectSamples(
        signals=signals[selected],
        labels=selected_labels,
        indices=(indices[0][selected.cpu().numpy()],),
        counts=counts,
    )


def build_balanced_attack_set(
    data_pack,
    models,
    snrs,
    data_tag="test",
    samples_per_class=100,
    batch_size=512,
    seed=2022,
):
    """Build an equal-size SNR/class attack set from joint correct samples."""
    if not models:
        raise ValueError("At least one model must be supplied.")
    if not snrs:
        raise ValueError("At least one SNR must be selected.")
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive.")

    per_snr = {}
    common_labels = None
    for snr in snrs:
        signals, labels, indices = data_pack.snr_slice(data_tag, snr)
        print(f"\nSNR {snr}: evaluating {len(labels)} {data_tag} samples")
        samples = jointly_correct_samples(
            models, signals, labels, indices, batch_size
        )
        per_snr[snr] = samples
        labels_at_snr = set(samples.counts)
        common_labels = (
            labels_at_snr
            if common_labels is None
            else common_labels.intersection(labels_at_snr)
        )

    common_labels = sorted(common_labels or [])
    if not common_labels:
        raise ValueError(
            "No modulation class has jointly correct samples at every SNR."
        )

    available = min(
        per_snr[snr].counts[label]
        for snr in snrs
        for label in common_labels
    )
    selected_count = min(samples_per_class, available)
    if selected_count <= 0:
        raise ValueError("No jointly correct samples are available to save.")

    generator = torch.Generator().manual_seed(seed)
    attack_set = {}
    for snr in snrs:
        samples = per_snr[snr]
        selected_indices = []
        for label in common_labels:
            label_indices = torch.where(samples.labels == label)[0]
            order = torch.randperm(len(label_indices), generator=generator)
            selected_indices.append(label_indices[order[:selected_count]])
        selected_indices = torch.cat(selected_indices)
        attack_set[snr] = (
            samples.signals[selected_indices],
            samples.labels[selected_indices],
            (samples.indices[0][selected_indices.cpu().numpy()],),
        )

    return attack_set, selected_count, common_labels


def select_snrs(data_pack, snr_range):
    lower, upper = snr_range
    snrs = [snr for snr in data_pack.snr_envs if lower <= snr <= upper]
    if not snrs:
        raise ValueError(
            f"No dataset SNR lies in [{lower}, {upper}]. "
            f"Available SNRs: {data_pack.snr_envs}"
        )
    return snrs


def main():
    parser = get_parser(parsing=False)
    group = parser.add_argument_group("Attack-set construction")
    group.add_argument(
        "--models",
        nargs="+",
        default=MVRG_ATTACK_SET_DEFAULTS["models"],
        help="model names whose jointly correct samples form the attack set "
        "(default: the seven paper models)",
    )
    group.add_argument(
        "--snr-range",
        nargs=2,
        type=int,
        metavar=("MIN", "MAX"),
        default=MVRG_ATTACK_SET_DEFAULTS["snr_range"],
        help="inclusive SNR range (default: 4 25, matching the MVRG experiments)",
    )
    group.add_argument(
        "--samples-per-class",
        type=int,
        default=MVRG_ATTACK_SET_DEFAULTS["samples_per_class"],
        help="maximum samples retained per SNR/class pair (default: 100)",
    )
    group.add_argument(
        "--data-tag",
        choices=["train", "val", "test"],
        default=MVRG_ATTACK_SET_DEFAULTS["data_tag"],
        help="dataset split from which samples are selected (default: test)",
    )
    group.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="MODEL=PATH",
        help="override a configured checkpoint; may be supplied repeatedly (default: none)",
    )
    group.add_argument(
        "--output",
        default=None,
        help="output .pt path; defaults to data/postdata/attack.<dataset>_dict.pt",
    )
    parser.set_defaults(
        data=MVRG_ATTACK_SET_DEFAULTS["data"],
        batch_size=MVRG_ATTACK_SET_DEFAULTS["batch_size"],
        seed=MVRG_ATTACK_SET_DEFAULTS["seed"],
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    task = Attack(args, parser)
    data_pack = task.data_opts
    data_pack.pack_dataset()
    snrs = select_snrs(data_pack, args.snr_range)
    checkpoint_overrides = parse_checkpoint_overrides(args.checkpoint)

    models = {}
    for model_name in args.models:
        checkpoint = checkpoint_overrides.get(model_name, "")
        models[model_name] = task.model_import(
            model_name=model_name, ckp=checkpoint
        )

    attack_set, sample_count, labels = build_balanced_attack_set(
        data_pack=data_pack,
        models=models,
        snrs=snrs,
        data_tag=args.data_tag,
        samples_per_class=args.samples_per_class,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    output = args.output or os.path.join(
        "data", "postdata", f"attack.{data_pack.data_name}_dict.pt"
    )
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    torch.save(attack_set, output)

    metadata = {
        "dataset": data_pack.data_name,
        "data_tag": args.data_tag,
        "models": args.models,
        "snrs": snrs,
        "labels": labels,
        "samples_per_class_per_snr": sample_count,
        "total_samples": sum(len(values[1]) for values in attack_set.values()),
        "seed": args.seed,
    }
    metadata_path = f"{output}.json"
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"\nSaved attack set to {output}")
    print(f"Saved metadata to {metadata_path}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
