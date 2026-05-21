"""waypoint CLI dispatcher — lazy-imports each subcommand module."""

from __future__ import annotations

import argparse
import importlib

_SUBCOMMANDS = {
    "pretrain":         ("waypoint_bio.pretrain",         "Pretrain a microbiome GPT2 model"),
    "benchmark":        ("waypoint_bio.benchmark",        "Benchmark a model on the Compass tasks"),
    "embed":            ("waypoint_bio.embed",            "Generate per-sample embeddings"),
    "prepare-dataset":  ("waypoint_bio.prepare_dataset",  "Convert an abundance matrix into waypoint format"),
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="waypoint")
    sub = parser.add_subparsers(
        dest="cmd",
        required=True,
        metavar="{pretrain,benchmark,embed,prepare-dataset}",
    )
    for name, (_module_path, help_text) in _SUBCOMMANDS.items():
        # Eager add_arguments() would pull in torch via pretrain/benchmark/embed,
        # making `waypoint --help` slow. Defer until the subcommand is chosen.
        sub.add_parser(name, help=help_text, add_help=False)

    args, rest = parser.parse_known_args()
    module_path, _ = _SUBCOMMANDS[args.cmd]
    module = importlib.import_module(module_path)

    sub_parser = argparse.ArgumentParser(prog=f"waypoint {args.cmd}")
    module.add_arguments(sub_parser)
    module.run(sub_parser.parse_args(rest))


if __name__ == "__main__":
    main()
