"""Tests for paper additions: datasets, usage/cost, methods, statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calibral.config import load_config
from calibral.data import DATASETS, create_splits_from_frames, dataset_spec
from calibral.llm_labeler import LLMLabelResult, build_prompt, summarize_llm_usage


def _frames(n_train=200, n_test=50):
    tr = pd.DataFrame(
        {"sample_id": [f"tr{i}" for i in range(n_train)], "text": ["x"] * n_train,
         "label": [i % 2 for i in range(n_train)]}
    )
    te = pd.DataFrame(
        {"sample_id": [f"te{i}" for i in range(n_test)], "text": ["y"] * n_test,
         "label": [int(i < 30) for i in range(n_test)]}  # unbalanced test
    )
    return tr, te


def test_dataset_registry_has_binary_sets_with_domains():
    for name in ("imdb", "sst2", "yelp_polarity", "amazon_polarity", "rotten_tomatoes", "snappfood"):
        assert dataset_spec(name).domain
    # SnappFood's numeric ids are reversed: labels must come from the strings.
    assert dict(dataset_spec("snappfood").label_map) == {"HAPPY": 1, "SAD": 0}
    assert dataset_spec("sst2").test_split == "validation"
    with pytest.raises(ValueError):
        dataset_spec("nope")
    assert set(DATASETS) >= {"imdb", "sst2"}


def test_test_size_all_keeps_full_unbalanced_test():
    tr, te = _frames()
    bundle = create_splits_from_frames(
        tr, te, initial_labeled_size=10, pool_size=100, validation_size=20,
        test_size="all", seed=1,
    )
    assert len(bundle.test) == 50
    assert int(bundle.test["label"].sum()) == 30


def test_prompt_v2_uses_dataset_domain():
    prompt = build_prompt("great food", prompt_version="v2", domain="business reviews (Yelp)")
    assert "business reviews (Yelp)" in prompt
    assert "{domain}" not in prompt


def test_summarize_llm_usage_counts_cached_cost_and_estimates():
    results = [
        LLMLabelResult("positive", 0.9, "api", source="api", prompt_tokens=100,
                       completion_tokens=10, cost_usd=0.001),
        LLMLabelResult("negative", 0.9, "api", cached=True, source="cache",
                       prompt_tokens=100, completion_tokens=10, cost_usd=0.001),
        LLMLabelResult("negative", 0.9, "api", source="api", prompt_tokens=1_000_000,
                       completion_tokens=0),
        LLMLabelResult("negative", 0.0, "api", source="error"),
    ]
    usage = summarize_llm_usage(results, price_per_million_input=0.5,
                                price_per_million_output=1.0)
    assert usage["llm_cache_hits"] == 1
    assert usage["llm_errors"] == 1
    assert usage["llm_cost_known_calls"] == 2
    assert usage["llm_cost_estimated_calls"] == 1
    assert usage["llm_cost_usd"] == pytest.approx(0.502)


def test_paired_bootstrap_detects_clear_improvement():
    from calibral.significance import paired_bootstrap

    rng = np.random.default_rng(0)
    true = rng.integers(0, 2, size=2000)
    good = true.copy()
    bad = true.copy()
    flip = rng.random(2000) < 0.2
    bad[flip] = 1 - bad[flip]
    out = paired_bootstrap(true, good, bad, n_resamples=300)
    assert out["delta_macro_f1"] > 0.15
    assert out["ci95_low"] > 0
    assert out["p_one_sided"] < 0.01


def test_methods_registry_and_paper_configs():
    from calibral.experiment import METHODS

    cfg = load_config("configs/paper_al.yaml")
    assert set(cfg["experiment"]["methods"]) <= set(METHODS)
    assert len(cfg["seeds"]) >= 5
    for name in ("sst2", "yelp_polarity", "amazon_polarity", "rotten_tomatoes", "snappfood"):
        c = load_config(f"configs/paper/{name}.yaml")
        assert c["data"]["dataset_name"] == name
        assert c["llm"]["model"] == "auto"


def test_resolve_llm_model_auto(tmp_path):
    import json

    from calibral.experiment import resolve_llm_model

    (tmp_path / "best_llm.json").write_text(json.dumps({"best_model": "x/y-0731"}))
    cfg = {"llm": {"model": "auto"}, "bakeoff": {"output_dir": str(tmp_path)}}
    assert resolve_llm_model(cfg)["llm"]["model"] == "x/y-0731"
    assert cfg["llm"]["model"] == "auto"  # input not mutated


def test_mcnemar_exact():
    from calibral.probe_report import mcnemar_exact

    same = [True] * 10
    assert mcnemar_exact(same, same) == 1.0
    a = [True] * 20
    b = [False] * 20
    assert mcnemar_exact(a, b) < 0.001


def test_probe_report_renders_winner_and_archive(tmp_path):
    import json

    from calibral.probe_report import build_probe_report

    probe = tmp_path / "llm_probe"
    (probe / "predictions").mkdir(parents=True)
    rows = []
    for name, acc, cost in (("x/best-0731", 0.95, 0.00002), ("y/other", 0.9, 0.0001)):
        recs = [
            {"sample_id": f"s{i}", "gold": i % 2, "pred": i % 2, "confidence": 0.9,
             "correct": i < int(acc * 20)}
            for i in range(20)
        ]
        (probe / "predictions" / f"{name.replace('/', '__')}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in recs)
        )
        rows.append({"model": name, "status": "completed", "accuracy": acc,
                     "accuracy_ci95": [acc - 0.1, min(1, acc + 0.04)], "ece": 0.05,
                     "brier": 0.04, "n_errors": 0, "runtime_seconds": 10,
                     "cost_per_1k_labels_usd": cost * 1000, "cost_usd": cost * 20})
    (probe / "llm_probe_results.json").write_text(json.dumps(
        {"ranking": rows, "n_probe_samples": 20, "finished_at": "2026-11-23T13:00:00+00:00",
         "prompt_version": "v2", "shot_mode": "zero", "base_config": "configs/paper_al.yaml"}
    ))
    out = build_probe_report(tmp_path)
    text = out.read_text(encoding="utf-8")
    assert "THE WINNER IS" in text and "best-0731" in text
    assert out.name.startswith("PROBE_REPORT_2026-11-23_")
    assert (tmp_path / "PROBE_REPORT.md").exists()
    assert (tmp_path / "llm_probe" / "figures" / "probe_accuracy_ci.png").exists()
    hist = list((tmp_path / "llm_probe" / "history").iterdir())
    assert len(hist) == 1 and (hist[0] / "PROBE_REPORT.md").exists()


def test_label_map_applied_when_loading(monkeypatch):
    import datasets

    import calibral.data as D

    fake = datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict({"comment": ["bad", "good"], "label": ["SAD", "HAPPY"]}),
            "test": datasets.Dataset.from_dict({"comment": ["meh"], "label": ["SAD"]}),
        }
    )
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: fake)
    tr, te = D.load_dataset_frames("snappfood", "/tmp/unused_cache")
    assert tr.set_index("text")["label"].to_dict() == {"bad": 0, "good": 1}
    assert te["label"].tolist() == [0]


def test_snappfood_encoder_ablation_matches_main_config():
    main = load_config("configs/paper/snappfood.yaml")
    abl = load_config("configs/paper/snappfood_xlmr.yaml")
    assert main["model"]["name"] != abl["model"]["name"]
    assert "xlm-roberta" in abl["model"]["name"]
    # Only the encoder (and output folder / methods) may differ.
    for key in ("data", "active_learning", "train", "llm", "seeds", "distill", "cost"):
        assert main[key] == abl[key], key
    assert abl["paths"]["output_dir"] != main["paths"]["output_dir"]
    assert "llm_zeroshot" not in abl["experiment"]["methods"]
