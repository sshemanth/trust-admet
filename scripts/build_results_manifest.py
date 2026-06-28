from pathlib import Path
import subprocess
import pandas as pd
from datetime import datetime

ROOT = Path("outputs")

rows = []

try:
    git_hash = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        text=True
    ).strip()
except Exception:
    git_hash = "unknown"

for path in ROOT.rglob("*"):
    if not path.is_file():
        continue

    parts = path.parts

    dataset = ""
    split = ""
    model = ""
    artifact_type = ""

    if "models" in parts:
        idx = parts.index("models")
        if len(parts) > idx + 4:
            dataset = parts[idx + 1]
            split = parts[idx + 2]
            model = parts[idx + 3]
            artifact_type = "model"

    elif "reports" in parts:
        idx = parts.index("reports")
        artifact_type = parts[idx + 1]

    rows.append({
        "path": str(path),
        "artifact_type": artifact_type,
        "dataset": dataset,
        "split": split,
        "model": model,
        "size_kb": round(path.stat().st_size / 1024, 2),
        "modified": datetime.fromtimestamp(path.stat().st_mtime),
        "git_commit": git_hash,
    })

manifest = pd.DataFrame(rows)
manifest = manifest.sort_values(["artifact_type", "dataset", "model"])

out = ROOT / "reports" / "manifest.csv"
out.parent.mkdir(parents=True, exist_ok=True)
manifest.to_csv(out, index=False)

print(manifest.head(30).to_string(index=False))
print(f"\nSaved manifest to: {out}")
