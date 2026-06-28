import argparse
import subprocess


DEFAULT_SEEDS = [42, 123, 2024, 3407, 9999]


def run(cmd):
    print("\nRunning:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["BBBP", "ClinTox", "Tox21_NR-AR", "Tox21_SR-p53", "Solubility", "Lipophilicity"])
    parser.add_argument("--splits", nargs="+", default=["scaffold"])
    parser.add_argument("--models", nargs="+", default=["random_forest", "xgboost", "mlp", "gin"])
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args()

    for dataset in args.datasets:
        for split in args.splits:
            for seed in args.seeds:
                for model in args.models:
                    if model in ["random_forest", "xgboost"]:
                        run([
                            "python", "scripts/train_classical_baseline.py",
                            "--dataset", dataset,
                            "--split", split,
                            "--model", model,
                            "--seed", str(seed),
                        ])

                    elif model == "mlp":
                        run([
                            "python", "scripts/run_mlp_experiment.py",
                            "--dataset", dataset,
                            "--split", split,
                            "--epochs", "50",
                            "--batch_size", "64",
                            "--lr", "1e-3",
                            "--seed", str(seed),
                        ])

                    elif model in ["gcn", "gin"]:
                        run([
                            "python", "scripts/run_gcn_experiment.py",
                            "--model", model,
                            "--dataset", dataset,
                            "--split", split,
                            "--epochs", "80",
                            "--batch_size", "64",
                            "--lr", "1e-3",
                            "--hidden_dim", "128",
                            "--num_layers", "3",
                            "--dropout", "0.2",
                            "--seed", str(seed),
                        ])

                    else:
                        raise ValueError(f"Unsupported model: {model}")


if __name__ == "__main__":
    main()
