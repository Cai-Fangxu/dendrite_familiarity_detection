"""Persistent Softmax calibration history shared by preparation and grid planning."""

import json
import math
from pathlib import Path
import tempfile


REGISTRY_NAME = "softmax_calibrations.json"


def read_calibrations(path):
    """Read calibration records from either a batch report or a registry."""
    path = Path(path)
    data = json.loads(path.read_text())
    records = data.get("calibrations") if isinstance(data, dict) else None
    if not isinstance(records, list):
        raise ValueError(f"{path}: expected a report or registry with calibrations")
    return records


def merge_calibrations(*groups):
    """Preserve distinct calibrations and reject conflicting target-n mappings."""
    targets, records = {}, []
    seen = set()
    for group in groups:
        for result in group:
            try:
                nd = int(result["nd"])
                beta_select = float(result["beta_select"])
                target = float(result["target_n"])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError("Invalid calibration mapping") from exc
            if (nd != result["nd"] or nd < 1 or not math.isfinite(beta_select)
                    or beta_select < 0 or not math.isfinite(target) or not 1 <= target <= nd):
                raise ValueError("Invalid nd, beta_select, or target_n in calibration mapping")
            pair = (nd, beta_select)
            if pair in targets and targets[pair] != target:
                raise ValueError(f"conflicting target_n for nd={nd}, beta_select={beta_select}")
            targets[pair] = target
            # Keep different seeds/tolerances as provenance, even if beta is identical.
            identity = json.dumps(result, sort_keys=True, allow_nan=False)
            if identity not in seen:
                records.append(result)
                seen.add(identity)
    return records


def update_registry(path, calibrations, previous_report=None):
    """Merge history and atomically replace the registry before overwriting a batch.

    Existing batch reports are imported during the first update, preserving their
    mappings on upgrade. Run one calibration writer per registry at a time.
    """
    path = Path(path)
    groups = [read_calibrations(path)] if path.exists() else []
    if previous_report is not None and Path(previous_report).exists():
        groups.append(read_calibrations(previous_report))
    records = merge_calibrations(*groups, calibrations)
    content = json.dumps(dict(schema_version=1, calibrations=records),
                         indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return records


def discover_sources(data_dir, report_paths=(), registry_path=None):
    """Combine explicit reports with their registries and the nearest data ancestor.

    Linked/sibling registries let old --calibration-report commands keep working.
    An explicit registry replaces ancestor discovery, while reports remain checked
    for conflicting mappings.
    """
    sources = []

    def add(path):
        path = Path(path).resolve()
        if path not in sources:
            sources.append(path)

    if registry_path is not None:
        add(registry_path)
    else:
        data_dir = Path(data_dir).resolve()
        for directory in (data_dir, *data_dir.parents):
            candidate = directory / REGISTRY_NAME
            if candidate.is_file():
                add(candidate)
                break
    for path in report_paths:
        path = Path(path).resolve()
        add(path)
        data = json.loads(path.read_text())
        linked = data.get("calibration_registry") if isinstance(data, dict) else None
        if linked:
            linked = Path(linked)
            linked = linked if linked.is_absolute() else path.parent / linked
            if linked.is_file():
                add(linked)
        sibling = path.parent / REGISTRY_NAME
        if sibling.is_file():
            add(sibling)
    return sources
