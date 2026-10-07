"""Record reproducibility metadata without credentials or environment secrets."""
import importlib.metadata
import json
import pathlib
import platform
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from labkit import device, report
from labkit.config import get_tier


def main():
    names = ["torch", "transformers", "trl", "peft", "accelerate", "datasets",
             "bitsandbytes", "tokenizers", "jinja2", "jupytext", "pytest"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    meta = {"python": platform.python_version(), "platform": platform.platform(),
            "device": device.describe(), "precision": device.precision(),
            "model": get_tier().model_id, "packages": versions}
    report.write_json(meta, "environment.json", results_dir=ROOT / "results")
    proc = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                          capture_output=True, text=True, check=True)
    (ROOT / "submission" / "requirements-lock.txt").write_text(proc.stdout, encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
