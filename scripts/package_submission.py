"""Package Option A only after the report and complete experiment pass verification."""
import pathlib
import subprocess
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_submission_report.py")],
                   cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "verify.py")], cwd=ROOT, check=True)
    correct = ROOT / "adapters" / "correct"
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        if not (correct / name).is_file():
            raise SystemExit(f"Required correct adapter missing: {name}")
    dest = ROOT.parent / "lab21_2A202602600.zip"
    prefix = "lab21_2A202602600/"
    files = []
    for folder in ("submission", "results", "notebooks", "src", "scripts", "data"):
        files.extend(p for p in (ROOT / folder).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and p.name != ".gitkeep")
    files.extend(p for p in correct.iterdir() if p.is_file())
    files.extend(ROOT / n for n in ("README.md", "rubric.md", "requirements.txt",
                                    "requirements-cpu.txt", "pyproject.toml", ".gitattributes"))
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path, prefix + path.relative_to(ROOT).as_posix())
        archive.writestr(prefix + ".env", "COMPUTE_TIER=LAPTOP\n"
                        "BASE_MODEL=Qwen/Qwen3.5-0.8B\nMASK_MODE=assistant-only\nEPOCHS=2\n")
    print(f"Submission: {dest} ({dest.stat().st_size / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()
