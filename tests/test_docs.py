"""The README shows the committed benchmark results, and the result files agree."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def test_readme_tables_are_the_committed_results():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    tables = (RESULTS / "benchmark.md").read_text(encoding="utf-8")

    after_begin = readme.split("<!-- benchmark:begin", 1)[1].split("-->", 1)[1]
    block = after_begin.split("<!-- benchmark:end -->", 1)[0]
    assert block.strip() == tables.strip()


def test_result_files_describe_the_same_runs():
    data = json.loads((RESULTS / "benchmark.json").read_text(encoding="utf-8"))
    with open(RESULTS / "benchmark.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == len(data["runs"])
    for row, run in zip(rows, data["runs"]):
        assert (row["track"], row["sensor"], row["variant"]) == (
            run["track"],
            run["sensor"],
            run["variant"],
        )
        assert row["status"] == run["status"]
        assert int(row["laps_valid"]) == run["laps_valid"]
        assert int(row["cones_hit_total"]) == run["cones_hit_total"]

    tables = (RESULTS / "benchmark.md").read_text(encoding="utf-8")
    for run in data["runs"]:
        if run["best_lap_s"] is not None:
            assert f"{run['best_lap_s']:.2f}" in tables


def test_images_referenced_by_the_readme_exist():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for name in ("docs/closed_loop.gif", "docs/laps.png"):
        assert name in readme
        assert (ROOT / name).is_file()
