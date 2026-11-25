"""Fill enabled Fig. 1 rules from capacity summaries, using only the stdlib.

Edit SOURCE_DIRS below, then run from any working directory:
    python theory_check/fill_fig1_config.py --config theory_check/fig1_config.example.json

The default output is <config-stem>.filled.json; use --in-place to replace the
input. Each mapped folder is searched recursively for max_points_df.csv, so map
only folders belonging to the intended experimental condition. Exact capacity
ties keep the first row in sorted file order. Disabled rules are left untouched.
"""

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import tempfile


ROOT_DIR = Path(__file__).resolve().parents[1]
CALIBRATION_FILE = ROOT_DIR / "tmp_data" / "softmax_calibrations.json"

# Replace None with a folder or a max_points_df.csv path. Relative paths are
# relative to the repository root, not the config file or working directory.
# Paths may use {nd}, {n}, {ns}, e.g. "data/YOUR_SEARCH/nd{nd}/n{n}".
# Keep separate entries for normalized and decay rules, even if they share data.
SOURCE_DIRS = {
    "independent": {
        "threshold": "data/gridSearch_NonInt/60000/coeff_infty",
        "nwta": "data/gridSearch_standard/60000/coeff_infty",
        "variable_target": "data/gridSearch_int_variable_n_min1/60000/coeff_infty",
        "softmax_int": None,
        "softmax_nonint": None,
        "threshold_decay": "data/gridSearch_NonInt/60000/coeff_infty",
        "nwta_decay": "data/gridSearch_standard/60000/coeff_infty",
        "variable_target_decay": "data/gridSearch_int_variable_n_min1/60000/coeff_infty",
    },
    "shared": {
        "threshold": "data/gridSearch_NonInt/60000/coeff_1",
        "nwta": "data/gridSearch_standard/60000/coeff_1",
        "variable_target": "data/gridSearch_int_variable_n_min1/60000/coeff_1",
        "softmax_int": None,
        "softmax_nonint": None,
        "threshold_decay": "data/gridSearch_NonInt/60000/coeff_1",
        "nwta_decay": "data/gridSearch_standard/60000/coeff_1",
        "variable_target_decay": "data/gridSearch_int_variable_n_min1/60000/coeff_1",
    },
}

RULE_PARAMETERS = {
    "threshold": ("b", "kappa"),
    "nwta": ("b", "kappa"),
    "variable_target": ("b", "kappa"),
    "softmax_int": ("eta", "beta_select"),
    "softmax_nonint": ("eta", "beta_select"),
    "threshold_decay": ("b", "kappa", "beta_decay"),
    "nwta_decay": ("b", "kappa", "beta_decay"),
    "variable_target_decay": ("b", "kappa", "beta_decay"),
}


def number(value):
    """Read finite numeric CSV values; missing or invalid cells become None."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def same_number(value, expected):
    """Allow only CSV round-trip differences, not nearby parameter settings."""
    value = number(value)
    return value is not None and math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12)


def load_calibrations(path):
    """Load softmax target-n metadata; an absent registry supplies no matches."""
    path = Path(path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    records = data.get("calibrations") if isinstance(data, dict) else None
    if not isinstance(records, list):
        raise ValueError(f"{path}: expected a 'calibrations' list")
    mapping = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"{path}: invalid calibration record")
        nd, beta, target = (number(record.get(key)) for key in ("nd", "beta_select", "target_n"))
        if (None in (nd, beta, target) or nd < 1 or nd != int(nd)
                or beta < 0 or not 1 <= target <= nd):
            raise ValueError(f"{path}: invalid calibration record")
        key = (nd, beta)
        if key in mapping and not same_number(mapping[key], target):
            raise ValueError(f"{path}: conflicting target_n for {key}")
        mapping[key] = target
    return mapping


def matches(row, nd, n, ns, softmax, calibrations):
    """Match dimensions, using calibrated n when a softmax CSV omits it."""
    if not same_number(row.get("nd"), nd):
        return False
    # Older 60000-synapse summaries omit ns; the folder mapping must enforce
    # that budget. New summaries carry ns and are checked explicitly.
    if "ns" in row and not same_number(row["ns"], ns):
        return False
    n_columns = [key for key in ("n", "ndR", "target_n") if key in row]
    if n_columns:
        return all(same_number(row[key], n) for key in n_columns)
    if not softmax:
        return False
    targets = {target for (cal_nd, beta), target in calibrations.items()
               if cal_nd == nd and same_number(row.get("beta_select"), beta)}
    if len(targets) > 1:
        raise ValueError(f"Ambiguous softmax calibration for nd={nd}, beta_select={row.get('beta_select')}")
    return len(targets) == 1 and same_number(targets.pop(), n)


def best_row(source, nd, n, ns, softmax, calibrations):
    """Return the highest finite capacity and its provenance for this setting."""
    if source is None:
        return None
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("Each source must be a nonempty path or None")
    path = Path(str(source).format(nd=nd, n=f"{n:g}", ns=ns)).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    files = [path] if path.is_file() else sorted(path.rglob("max_points_df.csv"))
    best = None
    for csv_path in files:
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not {"nd", "capacity"}.issubset(reader.fieldnames or []):
                raise ValueError(f"{csv_path}: missing nd or capacity column")
            for line, row in enumerate(reader, start=2):
                capacity = number(row.get("capacity"))
                if capacity is None or not matches(row, nd, n, ns, softmax, calibrations):
                    continue
                if best is None or capacity > best[0]:
                    best = (capacity, row, csv_path, line)
    return best


def parameters(row, required):
    """Translate legacy CSV names; never copy decay beta into spherical rules."""
    values = {key: number(row.get(key)) for key in required}
    if "kappa" in values and "kappa" not in row:
        b, bk = number(row.get("b")), number(row.get("bk"))
        values["kappa"] = None if b is None or bk is None else number(bk - b)
    if "beta_decay" in values and "beta_decay" not in row:
        values["beta_decay"] = number(row.get("beta"))
    for key in ("kappa", "beta_select", "eta"):
        value = values.get(key)
        if value is not None and (value < 0 or (key == "eta" and value == 0)):
            values[key] = None
    return values


def fill_config(config, sources=None, calibration_file=CALIBRATION_FILE):
    """Return a filled copy and selection messages without mutating the input.

    nd/n are selectors, not fitted parameters. A missing match clears all fitted
    parameters and disables the rule. An incomplete winning row fills available
    values, clears its missing parameters, and disables the rule; it does not
    silently substitute a lower-capacity row.
    """
    if not isinstance(config, dict) or config.get("condition") not in ("independent", "shared"):
        raise ValueError('Specify one condition: "independent" or "shared"')
    if not isinstance(config.get("rules"), list):
        raise ValueError("rules must be a list")
    sources = SOURCE_DIRS if sources is None else sources
    result, messages, calibrations = copy.deepcopy(config), [], None
    for rule in result["rules"]:
        if not isinstance(rule, dict) or not isinstance(rule.get("enabled", True), bool):
            raise ValueError("Each rule must be an object with optional boolean enabled")
        name = rule.get("name")
        if not rule.get("enabled", True):
            messages.append(f"{name}: already disabled; unchanged")
            continue
        if not isinstance(name, str) or name not in RULE_PARAMETERS:
            raise ValueError(f"Unknown enabled rule: {name}")
        nd, n = rule.get("nd", config.get("nd", 100)), rule.get("n", config.get("n", 10))
        if isinstance(nd, bool) or not isinstance(nd, int) or nd < 1 or 60000 % nd:
            raise ValueError(f"{name}: nd must be a positive integer dividing 60000")
        if (isinstance(n, bool) or not isinstance(n, (int, float))
                or number(n) is None or not 0 < n <= nd):
            raise ValueError(f"{name}: n must be in (0, nd]")
        if name in ("nwta", "nwta_decay") and n != int(n):
            raise ValueError(f"{name}: n must be an integer")
        source = sources.get(config["condition"], {}).get(name)
        softmax = name.startswith("softmax")
        if softmax and source is not None and calibrations is None:
            calibrations = load_calibrations(calibration_file)
        best = best_row(source, nd, n, 60000 // nd, softmax, calibrations or {})
        values = parameters({} if best is None else best[1], RULE_PARAMETERS[name])
        rule.update(values)
        rule["enabled"] = all(value is not None for value in values.values())
        label = f"{name} (nd={nd}, n={n:g})"
        if best is None:
            reason = "no folder mapping" if source is None else "no matching finite-capacity row"
            if softmax and source is not None:
                reason += "; rows without n need a matching --calibration-file"
            messages.append(f"{label}: disabled, parameters=null ({reason})")
        else:
            capacity, _, path, line = best
            messages.append(f"{label}: capacity={capacity:g}, {path}:{line}; {values}")
            if not rule["enabled"]:
                missing = [key for key, value in values.items() if value is None]
                messages.append(f"  disabled: winning row has missing/invalid {', '.join(missing)}")
    return result, messages


def write_config(path, config):
    """Atomically replace the output so an interrupted write cannot truncate it."""
    content = json.dumps(config, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    """Fill a JSON config using the editable SOURCE_DIRS template above."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="Input experiment JSON")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output", type=Path, help="Output JSON (default: <stem>.filled.json)")
    output.add_argument("--in-place", action="store_true", help="Replace the input config")
    parser.add_argument("--calibration-file", type=Path, default=CALIBRATION_FILE,
                        help="Softmax registry (default: tmp_data/softmax_calibrations.json)")
    parser.add_argument("--dry-run", action="store_true", help="Report selections without writing")
    args = parser.parse_args(argv)
    try:
        config, messages = fill_config(json.loads(args.config.read_text()), calibration_file=args.calibration_file)
        for message in messages:
            print(message)
        if not args.dry_run:
            path = args.config if args.in_place else args.output
            if path is None:
                path = args.config.with_name(f"{args.config.stem}.filled.json")
            write_config(path, config)
            print(f"Wrote {path}")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
