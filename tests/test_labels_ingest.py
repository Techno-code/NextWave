"""Offline tests for labels, ingest and corpus. They need no web framework: pytest -q tests/test_labels_ingest.py"""
import csv, zipfile
from pathlib import Path
import pytest
from app import corpus, ingest, labels, train

def test_reviews_reference_known_ids_and_families():
    base = {r["id"] for r in labels.read(labels.BASE)}
    for r in labels.read(labels.REVIEW):
        assert r["id"] in base and r["reviewed"] in train.GROUP and r["rule"]

def test_apply_keeps_original_label():
    rows = labels.apply_reviews(labels.read(labels.BASE), {r["id"]: r for r in labels.read(labels.REVIEW)})
    ch = [r for r in rows if r["formulation_family"] != r["formulation_family_original"]]
    assert ch and all(r["label_review_rule"] for r in ch)
    assert len(rows) == 117

def test_apply_is_idempotent():
    rv = {r["id"]: r for r in labels.read(labels.REVIEW)}
    once = labels.apply_reviews(labels.read(labels.BASE), rv); twice = labels.apply_reviews(once, rv); assert once == twice

def test_card_reports_both_label_sets_and_interval():
    import json; c = json.loads((train.MODELS / "card.json").read_text())
    assert "original_label_metrics" in c and c["metrics"]["group_confident"][0]["ci95"]
    assert c["metrics"]["cv_accuracy"] > c["metrics"]["majority_baseline"] + .15

def test_candidates_find_decision_sentences():
    t = "Our firm grew. Choose which projects to fund under a capital limit. The weather was fine. How should the retailer split its ad budget across channels?"
    got = ingest.candidates(t); assert len(got) == 2 and all("weather" not in s for s in got)

def test_license_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "CORPUS", tmp_path / "c"); monkeypatch.setattr(ingest, "STATUS", tmp_path / "c" / "status.csv"); monkeypatch.setattr(ingest, "QUEUE", tmp_path / "q.csv")
    f = tmp_path / "in"; f.mkdir()
    body = "Chapter one.\n\n" + "Managers often need to decide which projects to fund. " * 3 + "\n\nChoose which suppliers to keep under a cost cap.\n\n" + "More context about operations and planning for the year. " * 4
    (f / "open.txt").write_text(body); (f / "closed.txt").write_text(body); (f / "nc.txt").write_text(body)
    (f / "sources.csv").write_text("file,title,license,allow_training,source_url\nopen.txt,Open Book,CC BY 4.0,yes,https://example.org\nnc.txt,NC Book,CC BY-NC 4.0,yes,\n")
    status, queue = ingest.build(f); by = {s["file"]: s for s in status}
    assert by["open.txt"]["training_allowed"] == "yes" and by["closed.txt"]["training_allowed"] == "no" and by["nc.txt"]["training_allowed"] == "no"
    assert queue and {q["source"] for q in queue} == {"Open Book"} and all(q["reviewed_family"] == "" for q in queue)

def test_merge_only_reviewed_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(labels, "CUSTOM", tmp_path / "custom.csv")
    q = tmp_path / "q.csv"; q.write_text("id,domain,problem,suggested_family,suggested_p,reviewed_family,source,license\n,,Choose which vendors to keep within a cost cap,knapsack,0.5,knapsack,Open Book,CC BY 4.0\n,,Pick sites for new clinics near patients,facility_location,0.7,,Open Book,CC BY 4.0\n,,Bogus label row here,knapsack,0.5,not_a_family,Open Book,CC BY 4.0\n")
    ingest.merge(str(q)); rows = labels.read(tmp_path / "custom.csv"); assert len(rows) == 118 and rows[-1]["formulation_family"] == "knapsack" and "Open Book" in rows[-1]["label_review_rule"]

def test_corpus_search_cites_title_and_license(tmp_path, monkeypatch):
    d = tmp_path / "corpus"; d.mkdir(); monkeypatch.setattr(corpus, "DIR", d); corpus._cache["sig"] = None
    (d / "a.jsonl").write_text('{"doc":"Open Book","license":"CC BY 4.0","n":3,"text":"Safety stock protects against demand variability and supplier delays in inventory planning."}\n{"doc":"Open Book","license":"CC BY 4.0","n":4,"text":"Brand colours and logo design guidelines for packaging."}\n')
    r = corpus.search("how much safety stock for supplier delays"); assert r and r[0][1]["n"] == 3 and r[0][1]["license"] == "CC BY 4.0"
    assert corpus.search("zebra quantum entanglement") == []
