import argparse
import subprocess
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cmd = [
        "python",
        "scripts/train_classical_baseline.py",
        "--dataset", str(cfg["dataset"]),
        "--split", str(cfg["split"]),
        "--model", str(cfg["model"]),
        "--seed", str(cfg["seed"]),
        "--n_bits", str(cfg.get("n_bits", 2048)),
        "--radius", str(cfg.get("radius", 2)),
    ]

    print("Running command:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
