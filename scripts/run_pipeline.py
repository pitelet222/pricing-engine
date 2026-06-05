#!/usr/bin/env python3
"""
run_pipeline.py — Execute notebooks 01-06 in sequence to generate all artifacts.

Runs from the project root:
    python scripts/run_pipeline.py

Each notebook is executed in-place via nbconvert (cell outputs are written back
to the .ipynb file). The script exits with a non-zero code on the first failure
and prints the relevant error output to help diagnose the problem.

Total runtime is roughly 15-45 minutes depending on hardware, since notebooks
03 and 04 train neural and LightGBM models respectively.

Generated artifacts
-------------------
    data/processed/avocado_features.csv      (notebook 02)
    data/outputs/forecast_future.csv         (notebook 03)
    data/outputs/demand_model.pkl            (notebook 04)
    data/outputs/pricing_recommendations.csv (notebook 04)
    data/outputs/conformal_pi_stats.csv      (notebook 05)
    data/outputs/uncertainty_pricing.csv     (notebook 05)
    data/outputs/shap_top_drivers.csv        (notebook 06)
    data/outputs/region_label_encoder.pkl    (notebook 04)
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.data.manifest import write_manifest  # noqa: E402 — must come after sys.path insert

_NOTEBOOKS = [
    "notebooks/01_eda.ipynb",
    "notebooks/02_feature_engineering.ipynb",
    "notebooks/03_forecasting.ipynb",
    "notebooks/04_pricing.ipynb",
    "notebooks/05_uncertainty.ipynb",
    "notebooks/06_explainability.ipynb",
]

_BAR = "=" * 60


def _run(notebook_rel: str) -> None:
    nb_path = _ROOT / notebook_rel
    print(f"  ▶  {notebook_rel}", flush=True)
    t0 = time.perf_counter()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=900",  # 15-min cap per notebook
            "--ExecutePreprocessor.kernel_name=python3",
            str(nb_path),
        ],
        capture_output=True,
        text=True,
        cwd=_ROOT,
    )

    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"\n  ✗  FAILED ({elapsed:.0f}s)\n")
        # Show the last 3 000 characters of stderr — enough to see the traceback.
        tail = (result.stderr or result.stdout or "")[-3000:]
        print(tail)
        sys.exit(result.returncode)

    print(f"     ✓  done ({elapsed:.0f}s)", flush=True)


def main() -> None:
    print(_BAR)
    print("  Avocado Pricing Engine — full pipeline")
    print(_BAR)

    t_start = time.perf_counter()
    for nb in _NOTEBOOKS:
        _run(nb)

    total = time.perf_counter() - t_start

    manifest_path = write_manifest(
        _ROOT / "data" / "outputs",
        _ROOT / "data" / "processed",
    )

    print(_BAR)
    print(f"  All notebooks completed in {total:.0f}s.")
    print("  Artifacts written to data/outputs/ and data/processed/")
    print(f"  Manifest written to {manifest_path.relative_to(_ROOT)}")
    print(_BAR)


if __name__ == "__main__":
    main()
