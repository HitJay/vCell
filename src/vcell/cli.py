"""Command-line interface for vCell: ``vcell gen-data | train | eval``.

Heavy imports (torch, model, trainer) are deferred into each subcommand so that
``vcell gen-data`` and ``vcell --help`` stay fast.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vcell.utils.config import Config


def cmd_gen_data(args: argparse.Namespace) -> None:
    from vcell.data.synthetic import generate_and_save

    path = generate_and_save(
        args.out,
        n_genes=args.n_genes,
        num_perturbations=args.num_perturbations,
        n_cells_per_pert=args.n_cells_per_pert,
        effect_sparsity=args.effect_sparsity,
        noise_std=args.noise_std,
        seed=args.seed,
    )
    print(f"Wrote synthetic dataset -> {path}")


def cmd_train(args: argparse.Namespace) -> None:
    from vcell.train.trainer import run_training

    config = Config.from_yaml(args.config) if args.config else Config()
    config.apply_overrides(args.set)
    run_training(config)


def cmd_eval(args: argparse.Namespace) -> None:
    from vcell.data.dataset import build_datasets
    from vcell.train.trainer import Trainer, load_checkpoint

    ckpt_path = Path(args.run) / "best.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
    model, config, _ = load_checkpoint(ckpt_path)
    bundle = build_datasets(config.data)
    trainer = Trainer(model, config)
    summary, _ = trainer.evaluate(bundle)
    print(json.dumps({"summary": summary}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vcell", description="Virtual cell perturbation-response model."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("gen-data", help="Generate a synthetic perturbation dataset (.npz).")
    g.add_argument("--out", required=True, help="Output .npz path.")
    g.add_argument("--n-genes", type=int, default=200, dest="n_genes")
    g.add_argument("--num-perturbations", type=int, default=16, dest="num_perturbations")
    g.add_argument("--n-cells-per-pert", type=int, default=300, dest="n_cells_per_pert")
    g.add_argument("--effect-sparsity", type=float, default=0.1, dest="effect_sparsity")
    g.add_argument("--noise-std", type=float, default=0.3, dest="noise_std")
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=cmd_gen_data)

    t = sub.add_parser("train", help="Train a VirtualCell model.")
    t.add_argument("--config", default=None, help="Path to a YAML config.")
    t.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="key=value",
        help="Config overrides, e.g. train.epochs=100 model.latent_dim=64",
    )
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval", help="Evaluate a trained run directory.")
    e.add_argument("--run", required=True, help="Run directory containing best.ckpt.")
    e.set_defaults(func=cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
