from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch


CHECKPOINT_SUFFIX = ".pt"
BUFFER_SUFFIXES = (
    ".running_mean",
    ".running_var",
    ".num_batches_tracked",
)


@dataclass
class CheckpointSummary:
    dataset: str
    method: str
    model: str
    variant: str
    relative_path: str
    checkpoint_type: str
    # parameter_entries: int
    # buffer_entries: int
    parameter_numel_k: int
    file_size_mb: float
    modified_at: str
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize checkpoint metadata into checkpoints/readme.md."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Checkpoint root directory to scan.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "readme.md",
        help="Markdown output file.",
    )
    return parser.parse_args()


def iter_checkpoint_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob(f"*{CHECKPOINT_SUFFIX}")):
        if path.is_file():
            yield path


def extract_state_dict(payload: object) -> tuple[str, dict[str, torch.Tensor] | None]:
    if isinstance(payload, dict):
        if all(torch.is_tensor(value) for value in payload.values()):
            return "state_dict", payload

        for key in ("model_state_dict", "state_dict"):
            nested = payload.get(key)
            if isinstance(nested, dict) and all(torch.is_tensor(value) for value in nested.values()):
                return f"wrapped:{key}", nested

        return "dict", None

    return type(payload).__name__, None


def infer_model_and_variant(dataset: str, file_path: Path) -> tuple[str, str]:
    stem = file_path.stem
    prefix = f"{dataset}_"
    if stem.startswith(prefix):
        stem = stem[len(prefix):]

    parts = stem.split(".")
    if len(parts) == 1:
        return parts[0], "-"

    return parts[0], ".".join(parts[1:])


def is_buffer_key(key: str) -> bool:
    return key.endswith(BUFFER_SUFFIXES)


def summarize_checkpoint(root: Path, path: Path) -> CheckpointSummary:
    relative_path = path.relative_to(root)
    dataset = relative_path.parts[0] if len(relative_path.parts) >= 1 else "-"
    method = relative_path.parts[1] if len(relative_path.parts) >= 2 else "-"
    model, variant = infer_model_and_variant(dataset, path)
    stat = path.stat()
    modified_at = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint_type, state_dict = extract_state_dict(payload)
        if state_dict is None:
            return CheckpointSummary(
                dataset=dataset,
                method=method,
                model=model,
                variant=variant,
                relative_path=relative_path.as_posix(),
                checkpoint_type=checkpoint_type,
                # parameter_entries=0,
                # buffer_entries=0,
                parameter_numel_k=0,
                file_size_mb=stat.st_size / (1024 * 1024),
                modified_at=modified_at,
                error="No tensor state_dict found",
            )

        parameter_numel = 0

        for value in state_dict.values():
            parameter_numel += value.numel()

        return CheckpointSummary(
            dataset=dataset,
            method=method,
            model=model,
            variant=variant,
            relative_path=relative_path.as_posix(),
            checkpoint_type=checkpoint_type,
            # parameter_entries=parameter_entries,
            # buffer_entries=buffer_entries,
            parameter_numel_k=round(parameter_numel / 1000),
            file_size_mb=stat.st_size / (1024 * 1024),
            modified_at=modified_at,
        )
    except Exception as error:
        return CheckpointSummary(
            dataset=dataset,
            method=method,
            model=model,
            variant=variant,
            relative_path=relative_path.as_posix(),
            checkpoint_type="unreadable",
            # parameter_entries=0,
            # buffer_entries=0,
            parameter_numel_k=0,
            file_size_mb=stat.st_size / (1024 * 1024),
            modified_at=modified_at,
            error=str(error),
        )


def format_int(value: int) -> str:
    return f"{value:,}"


def render_markdown(
    root: Path,
    output_path: Path,
    summaries: list[CheckpointSummary],
    display_root: str,
    display_output: str,
) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    datasets = sorted({item.dataset for item in summaries})
    methods = sorted({item.method for item in summaries})
    failures = [item for item in summaries if item.error]

    lines: list[str] = []
    lines.append("# Checkpoints Summary")
    lines.append("")
    lines.append(f"Generated at: {generated_at}")
    lines.append("")
    lines.append(f"Scanned root: `{display_root}`")
    lines.append("")
    lines.append("Regenerate with: `python checkpoints/summarize.py`")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    lines.append(f"| Datasets | {len(datasets)} |")
    lines.append(f"| Methods | {len(methods)} |")
    lines.append(f"| Checkpoints | {len(summaries)} |")
    lines.append(f"| Read failures | {len(failures)} |")
    lines.append("")
    lines.append("## Dataset Index")
    lines.append("")
    lines.append("| Dataset | Methods | Checkpoints |")
    lines.append("| --- | --- | ---: |")

    grouped: dict[str, dict[str, list[CheckpointSummary]]] = defaultdict(lambda: defaultdict(list))
    for item in summaries:
        grouped[item.dataset][item.method].append(item)

    for dataset in sorted(grouped):
        method_names = sorted(grouped[dataset])
        checkpoint_count = sum(len(grouped[dataset][method]) for method in method_names)
        lines.append(f"| {dataset} | {', '.join(method_names)} | {checkpoint_count} |")

    for dataset in sorted(grouped):
        lines.append("")
        lines.append(f"## {dataset}")
        for method in sorted(grouped[dataset]):
            lines.append("")
            lines.append(f"### {method}")
            lines.append("")
            lines.append("| Model | Variant | Type | Parameter numel (K) | Size (MB) | Modified | File |")
            lines.append("| --- | --- | ---  | ---: | ---: | --- | --- |")
            for item in sorted(grouped[dataset][method], key=lambda current: (current.model.lower(), current.variant.lower(), current.relative_path)):
                type_text = item.checkpoint_type if item.error is None else f"{item.checkpoint_type} ({item.error})"
                lines.append(
                    "| {model} | {variant} | {kind} | {param_numel_k} | {size:.2f} | {modified} | {path} |".format(
                        model=item.model,
                        variant=item.variant,
                        kind=type_text,
                        param_numel_k=format_int(item.parameter_numel_k),
                        size=item.file_size_mb,
                        modified=item.modified_at,
                        path=item.relative_path,
                    )
                )

    if failures:
        lines.append("")
        lines.append("## Failures")
        lines.append("")
        for item in failures:
            lines.append(f"- {item.relative_path}: {item.error}")

    lines.append("")
    lines.append(f"Output file: `{display_output}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = Path('./checkpoints')
    output = args.output.resolve()
    display_root = args.root.as_posix()
    if output.parent.name == "checkpoints" and output.name.lower() == "readme.md":
        display_output = "checkpoints/readme.md"
    else:
        display_output = args.output.as_posix()

    summaries = [summarize_checkpoint(root, path) for path in iter_checkpoint_files(root)]
    markdown = render_markdown(root, output, summaries, display_root, display_output)
    output.write_text(markdown, encoding="utf-8")
    print(f"Wrote {len(summaries)} checkpoint summaries to {output}")


if __name__ == "__main__":
    main()