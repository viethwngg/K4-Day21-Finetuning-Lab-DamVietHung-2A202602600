"""Make a portable Colab runner for this working tree, including unpushed fixes."""
import hashlib
import json
import pathlib
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    cells = [
        ("markdown", "# Lab 21 — Chạy bản bài làm trên Colab\n\n"
         "Chọn Runtime → Change runtime type → T4 GPU. Ô đầu tải file "
         "`lab21_source_2A202602600.zip` tạo cùng notebook này; không clone bản upstream. "
         "Chạy các ô theo thứ tự để đóng băng baseline trước train. Giữ toàn bộ eval và hai epoch."),
        ("code", '''from google.colab import files
import os, pathlib, zipfile
uploaded = files.upload()
assert len(uploaded) == 1, "Chọn đúng một ZIP source của bài làm"
source_zip = next(iter(uploaded))
workspace = pathlib.Path("/content/lab21").resolve()
workspace.mkdir(exist_ok=True)
with zipfile.ZipFile(source_zip) as archive:
    for member in archive.infolist():
        target = (workspace / member.filename).resolve()
        assert target.is_relative_to(workspace), "ZIP path outside workspace"
    archive.extractall(workspace)
os.chdir(workspace)
assert pathlib.Path("scripts/colab_run.py").is_file()
'''),
        ("code", '''import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
import torch
assert torch.cuda.is_available(), "Chọn T4 GPU rồi chạy lại"
print(torch.cuda.get_device_name(0))
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ.pop("EVAL_LIMIT", None)
subprocess.run([sys.executable, "scripts/verify.py", "--smoke"], check=True)
subprocess.run([sys.executable, "scripts/record_environment.py"], check=True)
'''),
        ("code", '''subprocess.run([sys.executable, "-u", "scripts/colab_run.py",
                "nb1", "nb2", "nb3", "nb4", "nb5"], check=True)
'''),
        ("code", '''subprocess.run([sys.executable, "scripts/package_submission.py"], check=True)
from IPython.display import Markdown, display
display(Markdown(pathlib.Path("submission/REPORT.md").read_text(encoding="utf-8")))
files.download("/content/lab21_2A202602600.zip")
'''),
    ]
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"},
                             "kernelspec": {"name": "python3", "display_name": "Python 3"}},
                "cells": []}
    for i, (kind, source) in enumerate(cells):
        cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(True),
                "id": hashlib.sha1(f"{i}:{source}".encode()).hexdigest()[:8]}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        notebook["cells"].append(cell)
    nb = ROOT / "colab" / "Lab21_SUBMISSION_RUN.ipynb"
    nb.write_text(json.dumps(notebook, ensure_ascii=True, indent=1), encoding="utf-8")
    source_zip = ROOT.parent / "lab21_source_2A202602600.zip"
    with zipfile.ZipFile(source_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for directory in ("src", "scripts", "tests", "notebooks", "data", "submission", "colab", "docs"):
            for path in sorted((ROOT / directory).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.write(path, path.relative_to(ROOT).as_posix())
        root_files = sorted(set(ROOT.glob("*.md")) | set(ROOT.glob("*.txt")))
        root_files += [ROOT / name for name in ("pyproject.toml", "Makefile", "LICENSE",
                                               ".gitattributes", ".gitignore", ".env.example")]
        for path in root_files:
            archive.write(path, path.name)
        archive.writestr(".env", "COMPUTE_TIER=LAPTOP\nBASE_MODEL=Qwen/Qwen3.5-0.8B\n"
                        "MASK_MODE=assistant-only\nEPOCHS=2\n")
    print(nb)
    print(source_zip)


if __name__ == "__main__":
    main()
