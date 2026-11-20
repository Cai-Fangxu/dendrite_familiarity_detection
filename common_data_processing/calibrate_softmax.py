"""Prepare Softmax simulation parameters from a batch of target effective counts.

Default: read tmp_data/tmp.csv (b_center, eta_reference, n, nd, ns) and write
         tmp_data/params_list_softmax.csv (nd, ns, eta, b_center, beta_select, -1, seed).
Example: python common_data_processing/calibrate_softmax.py input.csv --simulation-seed 123
Single-pair JSON mode remains available with --nd 100 --n 5.
Batch preparation uses eta = eta_reference * n (the target effective count).
No simulations are run.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

if __package__:
    from . import softmax_calibration_registry as registry
else:
    import softmax_calibration_registry as registry


def gaussian_ipr(nd, beta_select, samples, batch_size, seed):
    """Return mean sum(p**2) and its Monte Carlo standard error, using bounded memory."""
    rng = np.random.default_rng(seed)
    total = total_squared = 0.0
    for start in range(0, samples, batch_size):
        u = rng.standard_normal((min(batch_size, samples - start), nd))
        # Subtract before multiplying so even a large finite beta is well behaved.
        logits = beta_select * (u - u.max(axis=1, keepdims=True))
        shares = np.exp(logits)
        shares /= shares.sum(axis=1, keepdims=True)
        ipr = np.sum(shares**2, axis=1)
        total += float(ipr.sum())
        total_squared += float(ipr @ ipr)
    mean = total / samples
    variance = max(0., (total_squared - samples * mean**2) / (samples - 1))
    return mean, math.sqrt(variance / samples)


def calibrate(nd, n, samples=20000, batch_size=256, seed=0, rtol=1e-3, max_beta=1e6):
    """Solve E[sum(p**2)] = 1/n using common random samples at every trial beta.

    n=1 uses the finite approximation E[sum(p**2)] >= 1-rtol.
    rtol controls the empirical solve, not Monte Carlo uncertainty; an independent
    validation sample and standard error are reported separately.
    """
    if not isinstance(nd, int) or nd < 1 or not math.isfinite(n) or not 1 <= n <= nd:
        raise ValueError("Require integer nd >= 1 and 1 <= n <= nd")
    if not isinstance(samples, int) or samples < 2 or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Require samples >= 2 and batch_size >= 1")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not 0 < rtol < 1 or not math.isfinite(max_beta) or max_beta <= 0:
        raise ValueError("Require 0 < rtol < 1 and finite max_beta > 0")

    def evaluate(beta):
        return gaussian_ipr(nd, beta, samples, batch_size, seed)[0]

    target = 1 / n
    if n == nd:
        beta = 0.
    else:
        threshold = 1 - rtol if n == 1 else target
        low, high = 0., min(max_beta, max(1., math.sqrt(math.log(nd / n))))
        while evaluate(high) < threshold and high < max_beta:
            high = min(max_beta, 2 * high)
        if evaluate(high) < threshold:
            raise ValueError("Could not bracket calibration; increase max_beta or relax rtol")
        for _ in range(80):
            mid = (low + high) / 2
            value = evaluate(mid)
            if n != 1 and abs(value - target) <= rtol * target:
                low = high = mid
                break
            if value < threshold:
                low = mid
            else:
                high = mid
            if high - low <= 1e-12 * max(1., high):
                break
        beta = high if n == 1 else (low + high) / 2
    mean, error = gaussian_ipr(nd, beta, samples, batch_size, seed)
    validation_mean, validation_error = gaussian_ipr(nd, beta, samples, batch_size, seed + 1)
    return dict(
        nd=nd, target_n=n, beta_select=beta, target_ipr=target,
        measured_ipr=mean, ipr_standard_error=error, effective_n=1 / mean,
        validation_ipr=validation_mean, validation_standard_error=validation_error,
        samples=samples, batch_size=batch_size, seed=seed, rtol=rtol,
        max_beta=max_beta, approximate_n1=(n == 1 and nd > 1),
    )


def load_grid(path):
    """Validate headerless b_center, eta_reference, n, nd, ns rows, including fractional n."""
    rows = np.loadtxt(path, delimiter=",", ndmin=2)
    if rows.size == 0 or rows.shape[1] != 5:
        raise ValueError("Expected five columns: b_center, eta_reference, n, nd, ns")
    if not np.isfinite(rows).all():
        raise ValueError("Grid values must be finite")
    if not np.equal(rows[:, 3:], np.floor(rows[:, 3:])).all():
        raise ValueError("nd and ns must be integers")
    if np.any(rows[:, 3:] < 1) or np.any(rows[:, 1] <= 0):
        raise ValueError("Require nd, ns >= 1 and eta_reference > 0")
    if np.any(rows[:, 2] < 1) or np.any(rows[:, 2] > rows[:, 3]):
        raise ValueError("Target n must satisfy 1 <= n <= nd")
    return rows


def prepare_batch(input_path, output_path, *, simulation_seed=0, samples=20000,
                  batch_size=256, seed=0, rtol=1e-3, max_beta=1e6, registry_path=None):
    """Calibrate each distinct (nd, n) once and write a runnable table plus provenance.

    Input eta is a reference value; output eta = eta_reference * target n.
    Rows keep their input order. A common simulation seed matches random streams
    across parameter choices; it is separate from the Monte Carlo calibration seed.
    Validate and finish every calibration before replacing any existing output.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    report_path = output_path.with_suffix(".report.json")
    registry_path = Path(registry_path) if registry_path is not None else output_path.parent / registry.REGISTRY_NAME
    if registry_path.resolve() in {input_path.resolve(), output_path.resolve(), report_path.resolve()}:
        raise ValueError("Registry, input grid, output table, and batch report must have distinct paths")
    if input_path.resolve() in (output_path.resolve(), report_path.resolve()):
        raise ValueError("Output table and report must not overwrite the input grid")
    if not isinstance(simulation_seed, int) or not 0 <= simulation_seed <= 2**32 - 1:
        raise ValueError("simulation_seed must be an integer in [0, 2**32 - 1]")
    rows = load_grid(input_path)
    with np.errstate(over="ignore"):
        actual_etas = (rows[:, 1]) * rows[:, 2]
    if not np.isfinite(actual_etas).all():
        raise ValueError("Scaled eta = eta_reference * n must be finite")
    pairs = sorted({(int(row[3]), float(row[2])) for row in rows})
    results = {}
    for index, (nd, n) in enumerate(pairs):
        print(f"Calibrating {index + 1}/{len(pairs)}: nd={nd}, n={n:g}", flush=True)
        try:
            results[nd, n] = calibrate(nd, n, samples, batch_size, seed, rtol, max_beta)
        except ValueError as exc:
            raise ValueError(f"nd={nd}, n={n:g}: {exc}") from exc
    params = np.array([
        [nd, ns, eta, b_center, results[int(nd), float(n)]["beta_select"], -1, simulation_seed]
        for (b_center, eta_reference, n, nd, ns), eta in zip(rows, actual_etas)
    ])
    report = dict(
        input_file=str(input_path.resolve()), output_file=str(output_path.resolve()),
        input_columns=["b_center", "eta_reference", "n", "nd", "ns"],
        output_columns=["nd", "ns", "eta", "b_center", "beta_select", "beta", "seed"],
        input_rows=rows.tolist(), output_rows=params.tolist(),
        eta_scaling="target_n", simulation_seed=simulation_seed,
        calibration_registry=str(registry_path.resolve()),
        calibrations=list(results.values()),
    )
    report_text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    # Preserve the previous batch before replacing its report. Conflicts abort all writes.
    registry.update_registry(registry_path, list(results.values()), previous_report=report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, params, delimiter=",", fmt="%.17g")
    report_path.write_text(report_text)
    return report


def main(argv=None):
    """Prepare a batch table by default, or retain --nd/--n single-pair JSON mode."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv", type=Path, nargs="?",
                        help="Headerless b_center,eta_reference,n,nd,ns; default: repository tmp_data/tmp.csv")
    parser.add_argument("--nd", type=int, help="Single-pair JSON mode: number of dendrites")
    parser.add_argument("--n", type=float, help="Single-pair JSON mode: target effective count")
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0, help="Monte Carlo calibration seed (default: 0)")
    parser.add_argument("--simulation-seed", type=int, default=None,
                        help="Seed written to every runner row (batch mode only; default: 0)")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--max-beta", type=float, default=1e6)
    parser.add_argument("--registry", type=Path,
                        help="Persistent batch calibration history; default: softmax_calibrations.json beside output")
    parser.add_argument("--output", type=Path,
                        help="Batch CSV path (default: repository tmp_data/params_list_softmax.csv); JSON path in single-pair mode")
    args = parser.parse_args(argv)
    single_pair = args.nd is not None or args.n is not None
    if single_pair and (args.nd is None or args.n is None):
        parser.error("Single-pair mode requires both --nd and --n")
    if single_pair and (args.input_csv is not None or args.simulation_seed is not None or args.registry is not None):
        parser.error("Do not combine --nd/--n with a batch input, --simulation-seed, or --registry")
    try:
        if single_pair:
            result = calibrate(args.nd, args.n, args.samples, args.batch_size, args.seed, args.rtol, args.max_beta)
            report = json.dumps(result, indent=2, allow_nan=False) + "\n"
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(report)
            print(report, end="")
        else:
            tmp_data = Path(__file__).resolve().parents[1] / "tmp_data"
            output = args.output or tmp_data / "params_list_softmax.csv"
            report = prepare_batch(
                args.input_csv or tmp_data / "tmp.csv", output,
                simulation_seed=0 if args.simulation_seed is None else args.simulation_seed,
                samples=args.samples, batch_size=args.batch_size, seed=args.seed,
                rtol=args.rtol, max_beta=args.max_beta, registry_path=args.registry,
            )
            print(f"Wrote {len(report['input_rows'])} rows to {output}")
            print(f"Calibration report: {output.with_suffix('.report.json')}")
            print(f"Calibration registry: {args.registry or output.parent / registry.REGISTRY_NAME}")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    main()
