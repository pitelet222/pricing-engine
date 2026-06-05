"""
manifest.py — artifact checksum manifest for stale-artifact detection.

write_manifest() is called by scripts/run_pipeline.py after the full pipeline
completes.  check_manifest() is called by DataStore._validate() at API startup
and returns a (possibly empty) list of human-readable issue strings.

Both functions are pure I/O — no ML dependencies.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Artifact registry
# ---------------------------------------------------------------------------
# Format: (dir_key, filename) — "outputs" → outputs_dir, "processed" → processed_dir.
_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("outputs", "forecast_future.csv"),
    ("outputs", "conformal_pi_stats.csv"),
    ("outputs", "region_label_encoder.pkl"),
    ("outputs", "demand_model.pkl"),
    ("outputs", "pricing_recommendations.csv"),
    ("outputs", "uncertainty_pricing.csv"),
    ("outputs", "shap_top_drivers.csv"),
    ("processed", "avocado_features.csv"),
)

_MANIFEST_NAME = "manifest.json"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_manifest(outputs_dir: pathlib.Path, processed_dir: pathlib.Path) -> pathlib.Path:
    """Compute SHA-256 checksums for all artifacts and write manifest.json.

    Only files that actually exist are included — missing files are silently
    omitted so a partial pipeline run still produces a useful manifest.

    Returns the path of the written manifest file.
    """
    dirs = {"outputs": outputs_dir, "processed": processed_dir}
    entries: dict[str, dict] = {}
    for dir_key, filename in _ARTIFACTS:
        path = dirs[dir_key] / filename
        if path.exists():
            entries[f"{dir_key}/{filename}"] = {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }

    payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "files": entries,
    }
    manifest_path = outputs_dir / _MANIFEST_NAME
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def check_manifest(outputs_dir: pathlib.Path, processed_dir: pathlib.Path) -> list[str]:
    """Verify artifact checksums against manifest.json.

    Returns a list of issue strings (one per problem found).  An empty list
    means everything is OK.  If no manifest exists, returns an empty list —
    backward-compatible with environments that pre-date this feature.
    """
    manifest_path = outputs_dir / _MANIFEST_NAME
    if not manifest_path.exists():
        return []

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return [f"Could not read manifest.json: {exc}"]

    dirs = {"outputs": outputs_dir, "processed": processed_dir}
    issues: list[str] = []

    for key, meta in payload.get("files", {}).items():
        # Each key has the form "dir_key/filename", e.g. "outputs/demand_model.pkl".
        parts = key.split("/", 1)
        if len(parts) != 2 or parts[0] not in dirs:
            issues.append(f"Malformed manifest entry: {key!r}")
            continue

        path = dirs[parts[0]] / parts[1]
        if not path.exists():
            issues.append(f"Artifact missing: {key}")
            continue

        if _sha256(path) != meta.get("sha256"):
            issues.append(f"Checksum mismatch for {key} — artifact may be stale")

    return issues
