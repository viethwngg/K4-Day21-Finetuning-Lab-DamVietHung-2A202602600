"""Exercise NB2 -> NB5 without weights; report evidence must survive intact."""
import json
import runpy
import sys
from types import SimpleNamespace

import pytest

from labkit import generate
from labkit.config import get_tier

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("EVAL_LIMIT", raising=False)
    monkeypatch.setenv("COMPUTE_TIER", "LAPTOP")
    d = tmp_path / "data"
    d.mkdir()
    label = {"intent": "doi_tra", "urgency": "thap", "product": "áo",
             "sentiment": "trung_tinh"}
    targets = [{"input": f"Ticket {i}", "label": label} for i in range(2)]
    (d / "eval_target.jsonl").write_text(
        "\n".join(json.dumps(r) for r in targets), encoding="utf-8")
    (d / "eval_regression.jsonl").write_text(
        json.dumps({"instruction": "capital?", "keywords": ["Hà Nội"]}),
        encoding="utf-8")
    good = json.dumps(label, ensure_ascii=False)
    model = SimpleNamespace(eval=lambda: None)
    monkeypatch.setattr(generate, "load_base", lambda *a, **kw: (model, None))
    monkeypatch.setattr(generate, "free_memory", lambda: None)

    def predict(model, tok, prompts, **kw):
        label = kw["label"]
        if label.endswith("regression"):
            return ["Hà Nội"], 10.0
        if label.startswith("(a)"):
            return ["prose", "prose"], 10.0
        if label.startswith("(b)"):
            return [good, "prose"], 10.0
        return ["prose", good], 10.0

    monkeypatch.setattr(generate, "generate_batch", predict)
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(
        PeftModel=SimpleNamespace(from_pretrained=lambda model, *a: model)))
    runpy.run_path(str(ROOT / "notebooks" / "02_baselines.py"))
    return tmp_path


def test_baselines_frozen_before_training_and_outputs_retained(experiment):
    results = experiment / "results"
    frozen = json.loads((results / "baselines_frozen.json").read_text())
    assert frozen["frozen_at_utc"].endswith("+00:00")
    assert set(frozen["eval_checksums"]) == {"eval_target.jsonl", "eval_regression.jsonl"}
    assert frozen["model"] == get_tier().model_id
    assert not (experiment / "adapters").exists()
    saved = json.loads((results / "baseline_predictions.json").read_text(encoding="utf-8"))
    assert len(saved["target"]) == 2
    assert saved["regression"][0]["baseline_b_pred"] == "Hà Nội"


def test_qualitative_records_actual_losses_and_wins(experiment):
    runpy.run_path(str(ROOT / "notebooks" / "05_evaluate_and_verdict.py"))
    rows = json.loads((experiment / "results" / "qualitative.json").read_text(encoding="utf-8"))
    assert {r["outcome"] for r in rows} == {"win", "loss"}
    lost = next(r for r in rows if r["outcome"] == "loss")
    assert lost["baseline_b_score"] == 1.0 and lost["ft_score"] == 0.0
    assert lost["delta"] == -1.0
    assert json.loads(lost["baseline_b_pred"]) == lost["label"]


@pytest.mark.parametrize("changed", ["model", "regression_count", "data", "prompt"])
def test_evaluation_rejects_drift_from_frozen_baseline(experiment, monkeypatch, changed):
    if changed == "model":
        monkeypatch.setenv("BASE_MODEL", "different/model")
    elif changed == "prompt":
        monkeypatch.setattr(generate, "OPTIMIZED_PROMPT", "changed after freeze")
    elif changed == "regression_count":
        p = experiment / "data" / "eval_regression.jsonl"
        p.write_text(p.read_text() + "\n" + p.read_text())
    else:
        p = experiment / "data" / "eval_target.jsonl"
        p.write_text(p.read_text().replace("Ticket 0", "Changed ticket"))
    with pytest.raises(SystemExit):
        runpy.run_path(str(ROOT / "notebooks" / "05_evaluate_and_verdict.py"))
