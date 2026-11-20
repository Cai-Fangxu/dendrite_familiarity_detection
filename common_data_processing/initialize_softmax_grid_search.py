"""Start a new Softmax b/eta grid around maxima from an existing experiment.

Accept raw nd{nd}_ns{ns}/beta_select{value}.csv results or a Softmax summary.
Default output: tmp_data/tmp.csv with b_center,eta_reference,n_eff,nd,ns.
Optionally write a runner table with actual eta and the original beta_select.
"""

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common_data_processing.initialize_grid_search import _integer
from common_data_processing.plan_grid_search import Settings, SoftmaxGridKey, load_grids
from common_data_processing.plan_softmax_grid_search import load_eta_scales, rename_fields
from common_data_processing import softmax_calibration_registry as registry


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "tmp_data" / "tmp.csv"


def load_summary(path, total_synapses=None):
    """Read actual-eta summaries; infer missing ns only from an explicit total."""
    if total_synapses is not None:
        total_synapses = _integer(total_synapses, "total_synapses")
    grids = {}
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if not {"nd", "beta_select", "b", "eta", "capacity"} <= fields:
            raise ValueError("Summary requires nd, beta_select, b, eta, capacity")
        if "ns" not in fields and total_synapses is None:
            raise ValueError("Summary has no ns column; supply --total-synapses")
        for line_number, row in enumerate(reader, 2):
            try:
                nd = _integer(row["nd"], "nd")
                if "ns" in fields:
                    ns = _integer(row["ns"], "ns")
                else:
                    if total_synapses % nd:
                        raise ValueError("total_synapses must be divisible by nd to infer ns")
                    ns = _integer(total_synapses // nd, "ns")
                if total_synapses is not None and nd * ns != total_synapses:
                    raise ValueError("nd * ns does not match total_synapses")
                beta_select, b, eta, capacity = (
                    float(row[name]) for name in ("beta_select", "b", "eta", "capacity")
                )
                if not all(math.isfinite(value) for value in (beta_select, b, eta, capacity)):
                    raise ValueError("beta_select, b, eta, and capacity must be finite")
                if beta_select < 0 or eta <= 0 or capacity < 0:
                    raise ValueError("Require beta_select >= 0, eta > 0, and capacity >= 0")
                if "beta" in fields and float(row["beta"]) != -1:
                    raise ValueError("Unused decay beta must equal -1")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            key = SoftmaxGridKey(nd, ns, beta_select)
            grids.setdefault(key, []).append(dict(b=b, eta=eta, capacity=capacity))
    if not grids:
        raise ValueError(f"{path}: no summary rows")
    return grids


def load_source(source, settings, total_synapses=None):
    """Aggregate raw capacities or use existing summary means without modifying them."""
    source = Path(source)
    if not source.is_dir():
        return load_summary(source, total_synapses)
    grids = load_grids(source, settings, softmax=True)
    if total_synapses is not None:
        total = _integer(total_synapses, "total_synapses")
        if any(key.nd * key.ns != total for key in grids):
            raise ValueError("A source geometry does not match total_synapses")
    return {key: [dict(b=p.b, eta=p.bk, capacity=p.capacity) for p in points]
            for key, points in grids.items()}


def initial_grid(grids, settings, eta_scales):
    """Center a new grid in reference units, retaining each source beta_select.

    Capacity ties prefer larger b, then smaller actual eta. Rounding uses a new
    zero-anchored lattice. Nonpositive eta rows are omitted, not shifted away
    from the chosen optimum. Existing source points remain valid candidates for
    the new experiment.
    """
    reports = []
    for key, points in sorted(grids.items()):
        if not points:
            raise ValueError(f"{key}: empty grid")
        n_eff = eta_scales[key]
        if not math.isfinite(n_eff) or not 1 <= n_eff <= key.nd:
            raise ValueError(f"{key}: require finite target n_eff in [1, nd]")
        best = min(points, key=lambda p: (-p["capacity"], -p["b"], p["eta"]))
        report = dict(**asdict(key), n_eff=n_eff, source_best=best,
                      source_eta_reference=best["eta"] / n_eff, proposals=[])
        if best["capacity"] == 0:
            warnings.warn(f"{key}: all capacities are zero; skipping initial grid", stacklevel=2)
            report["status"] = "skipped_all_zero"
        else:
            b_index = round(max(0., best["b"]) / settings.b_spacing)
            eta_index = max(1, round((best["eta"] / n_eff) / settings.bk_spacing))
            b_center = round(b_index * settings.b_spacing, 12)
            eta_reference_center = round(eta_index * settings.bk_spacing, 12)
            report.update(
                status="initialized", b_center=b_center,
                eta_reference_center=eta_reference_center,
                eta_center=eta_reference_center * n_eff,
                b_min=round((b_index - settings.b_size // 2) * settings.b_spacing, 12),
                b_max=round((b_index + settings.b_size // 2) * settings.b_spacing, 12),
                eta_spacing=settings.bk_spacing * n_eff,
                omitted_nonpositive_eta=0,
            )
            for offset in range(-(settings.bk_size // 2), settings.bk_size // 2 + 1):
                if eta_index + offset <= 0:
                    report["omitted_nonpositive_eta"] += 1
                    continue
                reference = round((eta_index + offset) * settings.bk_spacing, 12)
                actual = reference * n_eff
                if reference <= 0 or not math.isfinite(actual) or actual <= 0:
                    raise ValueError("Eta spacing is too small or scaled eta is nonfinite")
                report["proposals"].append(dict(b_center=b_center, eta_reference=reference, eta=actual))
        reports.append(report)
    return reports


def validate_output_paths(source, output, params_output, calibration_sources):
    """Protect source data and calibration history from any generated output."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.suffix.lower() != ".csv":
        raise ValueError("--output must have a .csv extension")
    destinations = [output, output.with_suffix(".report.json")]
    if params_output is not None:
        params_output = Path(params_output).resolve()
        if params_output.suffix.lower() != ".csv":
            raise ValueError("--params-output must have a .csv extension")
        destinations.append(params_output)
    if len(destinations) != len(set(destinations)):
        raise ValueError("Output paths must be distinct")
    protected = {Path(path).resolve() for path in calibration_sources}
    if source.is_dir():
        protected.update(path.resolve() for path in source.glob("nd*_ns*/beta_select*.csv"))
    else:
        protected.add(source)
    if protected.intersection(destinations):
        raise ValueError("Outputs must not overwrite source data or calibration history")


def write_initial_grid(output, reports, settings, source, calibration_sources,
                       params_output=None, seed=0):
    """Replace reference candidates, optional runner rows, and the provenance report."""
    output = Path(output)
    settings_report = rename_fields(asdict(settings))
    settings_report["eta_reference_spacing"] = settings_report.pop("eta_spacing")
    audit = dict(
        source=str(Path(source).resolve()), settings=settings_report, eta_scaling="target_n",
        calibration_sources=[str(Path(path).resolve()) for path in calibration_sources],
        output_columns=["b_center", "eta_reference", "n_eff", "nd", "ns"],
        params_output=str(Path(params_output).resolve()) if params_output is not None else None,
        simulation_seed=seed if params_output is not None else None, grids=reports,
    )
    report_text = json.dumps(audit, indent=2, allow_nan=False) + "\n"
    reference_rows, parameter_rows = [], []
    for report in reports:
        for proposal in report["proposals"]:
            reference_rows.append([proposal["b_center"], proposal["eta_reference"],
                                   report["n_eff"], report["nd"], report["ns"]])
            parameter_rows.append([report["nd"], report["ns"], proposal["eta"],
                                   proposal["b_center"], report["beta_select"], -1, seed])
    outputs = [(output, reference_rows)]
    if params_output is not None:
        outputs.append((Path(params_output), parameter_rows))
    for path, rows in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            csv.writer(handle).writerows(rows)
    output.with_suffix(".report.json").write_text(report_text)


def main(argv=None):
    """Preview or write a new Softmax grid; dry runs perform no writes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Softmax raw-result directory or summary CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Reference candidate CSV (default: repository tmp_data/tmp.csv)")
    parser.add_argument("--params-output", type=Path,
                        help="Also write a ready-to-run table preserving source beta_select")
    parser.add_argument("--seed", type=int, default=0, help="Seed for optional runner rows")
    parser.add_argument("--total-synapses", type=int, help="Required for summaries without ns")
    parser.add_argument("--registry", type=Path, help="Override automatic calibration-registry discovery")
    parser.add_argument("--calibration-report", type=Path, action="append", help="Additional calibration report")
    parser.add_argument("--b-spacing", type=float, default=0.15)
    parser.add_argument("--eta-reference-spacing", type=float, default=0.25)
    parser.add_argument("--b-size", type=int, default=9, help="Scan width used to report the b range")
    parser.add_argument("--eta-size", type=int, default=5, help="Odd number of eta rows around the optimum")
    parser.add_argument("--n-trials", type=int, default=6, help="Capacity columns in each raw row")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not 0 <= args.seed <= 2**32 - 1:
            raise ValueError("seed must be in [0, 2**32 - 1]")
        settings = Settings(b_spacing=args.b_spacing, bk_spacing=args.eta_reference_spacing,
                            b_size=args.b_size, bk_size=args.eta_size, n_trials=args.n_trials)
        grids = load_source(args.source, settings, args.total_synapses)
        root = args.source if args.source.is_dir() else args.source.parent
        calibration_sources = registry.discover_sources(root, args.calibration_report or (), args.registry)
        if not calibration_sources:
            raise ValueError("No calibration registry found; supply --registry or --calibration-report")
        scales = load_eta_scales(grids, calibration_sources)
        validate_output_paths(args.source, args.output, args.params_output, calibration_sources)
        reports = initial_grid(grids, settings, scales)
        for report in reports:
            label = f"nd={report['nd']} ns={report['ns']} beta_select={report['beta_select']:.17g} n_eff={report['n_eff']:g}"
            if report["status"] == "skipped_all_zero":
                print(f"{label}: skipped_all_zero")
                continue
            print(f"{label}: b_center={report['b_center']:g}, "
                  f"eta_reference={[p['eta_reference'] for p in report['proposals']]}, "
                  f"eta={[p['eta'] for p in report['proposals']]}")
        count = sum(len(report["proposals"]) for report in reports)
        print(f"{count} candidate rows from {len(reports)} source grids")
        if not args.dry_run:
            write_initial_grid(args.output, reports, settings, args.source,
                               calibration_sources, args.params_output, args.seed)
            print(f"Wrote {args.output} and {args.output.with_suffix('.report.json')}")
            if args.params_output is not None:
                print(f"Wrote runner table {args.params_output}")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
