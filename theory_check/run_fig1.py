"""Run Figure 1 frozen-state experiments from an explicit JSON configuration.

Edit theory_check/fig1_config.example.json, then run from any working directory:
    python theory_check/run_fig1.py --config /path/to/config.json --validate-only
    python theory_check/run_fig1.py --config /path/to/config.json

Learning parameters are never imported from grid searches. Output is written to
a new run directory beneath /media/fangxu/T7/dendritic_computation by default.
Each configuration selects exactly one condition: independent or shared. Use
separate configurations when their learning parameters differ. No production
experiment can run until every enabled rule has its parameters.
"""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from theory_check.spherical import (
    RULE_NAMES, SUPPORTED_ND, Rule, fork_batch, indexed_key, overlaps, pattern,
    train_segment, initialize_weights,
)
from theory_check.statistics import (
    Moments, balanced_sizes, batch_se, choose_pairs, theory_moments,
)


DEFAULT_OUTPUT = "/media/fangxu/T7/dendritic_computation"
DEFAULT_SAMPLING = dict(
    forks=50000, fork_steps=10, fork_batch_size=32, uncertainty_groups=20,
    theory_samples=100000, theory_batch_size=4096, theory_seed=1729,
    train_chunk_steps=256, max_pairs=30,
)


def positive_integer(value, name, minimum=1):
    """Reject booleans and fractional counts rather than silently truncating."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def resolve_config(raw):
    """Validate fixed parameters and resolve reproducible experimental choices."""
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a JSON object")
    if "conditions" in raw:
        raise ValueError('Replace conditions with one condition: "independent" or "shared", and use a flat stages list')
    allowed = {"nd", "n", "rules", "sampling", "condition", "stages", "ages", "seeds", "dtype", "output_root"}
    if set(raw) - allowed:
        raise ValueError(f"Unknown configuration keys: {sorted(set(raw) - allowed)}")
    if not isinstance(raw.get("sampling", {}), dict):
        raise ValueError("sampling must be a JSON object")
    sampling = {**DEFAULT_SAMPLING, **raw.get("sampling", {})}
    if set(sampling) - set(DEFAULT_SAMPLING):
        raise ValueError("Unknown sampling option")
    for name, value in sampling.items():
        positive_integer(value, name, 0 if name in ("theory_seed", "max_pairs") else 1)
    groups = sampling["uncertainty_groups"]
    if groups < 2 or sampling["forks"] < 2 * groups or sampling["theory_samples"] < 2 * groups:
        raise ValueError("Use >= 2 uncertainty_groups, with >= 2 samples per group in forks and theory")
    if sampling["forks"] >= 2**32 or sampling["theory_seed"] >= 2**32:
        raise ValueError("Fork counts and theory_seed must fit uint32 stream indices")
    condition = raw.get("condition")
    if not isinstance(condition, str) or condition not in ("independent", "shared"):
        raise ValueError('Specify exactly one condition: "independent" or "shared"')
    stages = raw.get("stages", [6000] if condition == "independent" else [600, 3000, 12000])
    ages = raw.get("ages", [0, 10, 100, 300, 500])
    seeds = raw.get("seeds", [0, 1, 2])
    for name, values in (("ages", ages), ("seeds", seeds)):
        if not isinstance(values, list) or not values:
            raise ValueError(f"{name} must be a nonempty list")
        for value in values:
            positive_integer(value, name, 0)
            if value >= 2**32:
                raise ValueError(f"{name} must fit uint32 stream indices")
        if len(set(values)) != len(values):
            raise ValueError(f"{name} must be unique")
    if not isinstance(stages, list) or not stages:
        raise ValueError("stages must be a nonempty list for the selected condition")
    for t in stages:
        positive_integer(t, "stages")
        if t >= 2**32:
            raise ValueError("Each stage must fit uint32")
    if stages != sorted(set(stages)):
        raise ValueError("stages must be increasing and unique")
    dtype = raw.get("dtype", "float64")
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64")
    rules, missing, names = [], [], set()
    if not isinstance(raw.get("rules", []), list):
        raise ValueError("rules must be a list of JSON objects")
    for item in raw.get("rules", []):
        if not isinstance(item, dict) or not isinstance(item.get("enabled", True), bool):
            raise ValueError("Each rule must be an object with optional boolean enabled")
        if not item.get("enabled", True):
            continue
        unknown = set(item) - {"name", "enabled", "nd", "n", "b", "kappa", "eta", "beta_select", "beta_decay"}
        if unknown:
            raise ValueError(f"Unknown rule fields: {sorted(unknown)}")
        name = item.get("name")
        if not isinstance(name, str) or name not in RULE_NAMES or name in names:
            raise ValueError(f"Unknown or duplicate rule name: {name}")
        names.add(name)
        nd = positive_integer(item.get("nd", raw.get("nd", 100)), "nd")
        if nd not in SUPPORTED_ND:
            raise ValueError(f"Unsupported nd={nd}")
        n = item.get("n", raw.get("n", 10))
        if isinstance(n, bool) or not isinstance(n, (float, int)) or not np.isfinite(n) or not 0 < n <= nd:
            raise ValueError(f"{name}: n must be in (0, nd]")
        if name in ("nwta", "nwta_decay") and n != int(n):
            raise ValueError("nwta requires integer n")
        required = ("eta", "beta_select") if name.startswith("softmax") else ("b", "kappa")
        if name.endswith("_decay"):
            required += ("beta_decay",)
        params = {}
        for key in required:
            value = item.get(key)
            if value is None:
                missing.append(f"{name}.{key}")
            elif isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value):
                raise ValueError(f"{name}.{key} must be finite")
            else:
                params[key] = float(value)
        if "eta" in params and params["eta"] <= 0:
            raise ValueError("eta must be positive")
        if "beta_select" in params and params["beta_select"] < 0:
            raise ValueError("beta_select must be nonnegative")
        if "kappa" in params and params["kappa"] < 0:
            raise ValueError("kappa must be nonnegative")
        unused = set(item) & ({"b", "kappa", "eta", "beta_select", "beta_decay"} - set(required))
        if any(item[key] is not None for key in unused):
            raise ValueError(f"{name}: unused learning parameters {sorted(unused)}")
        rules.append(Rule(name, nd, 60000 // nd, n=n, **params))
    if not rules:
        raise ValueError("Enable at least one rule with explicit learning parameters")
    if missing:
        raise ValueError("Supply fixed learning parameters: " + ", ".join(missing))
    resolved = dict(sampling=sampling, condition=condition, stages=stages,
                    ages=sorted(ages), seeds=seeds, dtype=dtype,
                    output_root=str(raw.get("output_root", DEFAULT_OUTPUT)),
                    rules=[asdict(rule) for rule in rules])
    return resolved, rules


def write_json(path, value):
    """Write readable, standards-compliant run metadata."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def stream_key(seed, nd, condition, stream):
    """Use paired streams across rules, independent streams across purposes."""
    key = jax.random.PRNGKey(seed)
    for value in (nd, 0 if condition == "independent" else 1, stream):
        key = indexed_key(key, value)
    return key


def measure_forks(w, old, key, rule, condition, sampling):
    """Measure endpoint moments in bounded-memory batches with fixed fork count."""
    total = Moments()
    group_means, group_covariances, counts = [], [], []
    start = 0
    last_report = time.monotonic()
    for count in balanced_sizes(sampling["forks"], sampling["uncertainty_groups"]):
        group = Moments()
        for offset in range(0, count, sampling["fork_batch_size"]):
            size = min(sampling["fork_batch_size"], count - offset)
            indices = jnp.arange(start + offset, start + offset + size, dtype=jnp.uint32)
            values = fork_batch(w, old, key, indices, rule=rule, condition=condition,
                                steps=sampling["fork_steps"])
            group.add(np.asarray(values))
            if time.monotonic() - last_report > 30:
                print(f"    forks {start + offset + size}/{sampling['forks']}", flush=True)
                last_report = time.monotonic()
        start += count
        total.merge(group.count, group.mean, group.m2)
        group_means.append(group.mean)
        group_covariances.append(group.covariance)
        counts.append(count)
    return total, np.asarray(group_means), np.asarray(group_covariances), np.asarray(counts)


def point_rows(rule, condition, seed, t, ages, m, theory, empirical, pairs, steps):
    """Produce a common table, including age averages with shared-fork errors."""
    total, group_means, group_cov, counts = empirical
    scale = rule.ns / steps
    a, d = scale * total.mean, scale * total.covariance
    a_se = np.sqrt(np.maximum(np.diagonal(total.covariance, axis1=1, axis2=2), 0) / total.count) * scale
    d_se = batch_se(group_cov * scale, counts)
    th_a_se = batch_se(theory["batch_A"], theory["counts"])
    th_d_se = batch_se(theory["batch_C"], theory["counts"])
    base = dict(rule=rule.name, condition=condition, seed=seed, t=t, nd=rule.nd,
                ns=rule.ns, n=rule.n, b=rule.b, kappa=rule.kappa, eta=rule.eta,
                beta_select=rule.beta_select, beta_decay=rule.beta_decay,
                update="legacy_weight_decay" if rule.is_decay else "leading_angle_spherical",
                forks=total.count, fork_steps=steps)
    rows = []
    entries = [("diagonal", r, r) for r in range(rule.nd)] + [("cross", int(r), int(s)) for r, s in pairs]
    for p, age in enumerate(ages):
        for r in range(rule.nd):
            rows.append(dict(base, kind="drift", age=int(age), acquisition=t-int(age),
                             r=r, s=r, overlap=m[p, r], predicted=theory["A"][p, r],
                             observed=a[p, r], empirical_se=a_se[p, r], theory_se=th_a_se[p, r],
                             age_averaged=False))
        for kind, r, s in entries:
            rows.append(dict(base, kind=kind, age=int(age), acquisition=t-int(age),
                             r=r, s=s, overlap=m[p, r], predicted=theory["C"][r, s],
                             observed=d[p, r, s], empirical_se=d_se[p, r, s], theory_se=th_d_se[r, s],
                             age_averaged=False))
    # Average within each independent group first: old patterns share trajectories.
    avg_se = batch_se(group_cov.mean(axis=1) * scale, counts)
    for kind, r, s in entries:
        rows.append(dict(base, kind=kind, age=-1, acquisition=-1, r=r, s=s,
                         overlap=m[:, r].mean(), predicted=theory["C"][r, s],
                         observed=d[:, r, s].mean(), empirical_se=avg_se[r, s],
                         theory_se=th_d_se[r, s], age_averaged=True))
    return rows


def save_snapshot(directory, rule, condition, seed, t, old, w, ages, m, q, theory,
                  empirical, pairs, sampling):
    """Save reproducible state, full moments, MC groups, and point-level results."""
    directory.mkdir()
    total, group_means, group_cov, counts = empirical
    norms = np.linalg.norm(old, axis=-1)
    eligibility = rule.eligibility
    weight_norms = np.linalg.norm(w, axis=-1)
    metadata = dict(rule=asdict(rule), condition=condition, seed=seed, stage=t,
                    eligibility_threshold=eligibility if np.isfinite(eligibility) else "-Infinity",
                    update="legacy_weight_decay" if rule.is_decay else "leading_angle_spherical",
                    age_zero="immediately_after_learning",
                    fork_steps=sampling["fork_steps"], forks=total.count,
                    coefficient_reference="local_leading_decay_at_actual_norms" if rule.is_decay else "local_R4_R8_at_initial_frozen_state",
                    measured_coefficients="finite_duration_endpoint_moments",
                    full_sharing_coordinates="common_identity_order",
                    uncertainty="conditional on frozen weights; independent near-equal MC groups",
                    pair_selection="evenly_spaced_ranks_of_theoretical_C_off_diagonal",
                    R=float(q.sum() / rule.nd), rank=theory["rank"],
                    off_diagonal_rms=float(np.sqrt(np.sum((q - np.diag(np.diag(q)))**2) /
                                                   (rule.nd * (rule.nd - 1)))) if rule.nd > 1 else 0.0,
                    weight_norm_min=float(weight_norms.min()), weight_norm_mean=float(weight_norms.mean()),
                    weight_norm_max=float(weight_norms.max()),
                    max_norm_error=float(np.max(np.abs(weight_norms - 1))))
    write_json(directory / "metadata.json", metadata)
    np.savez_compressed(directory / "snapshot.npz", weights=w, Q=q, old_patterns=old,
                        old_norms=norms, weight_norms=weight_norms, overlaps=m, ages=ages, acquisition_times=t-np.asarray(ages),
                        acquisition_indices=t-np.asarray(ages)-1, pairs=pairs)
    np.savez_compressed(directory / "empirical.npz", K=total.count,
                        increment_mean=total.mean, increment_covariance=total.covariance,
                        group_counts=counts, group_mean=group_means, group_covariance=group_cov,
                        scale=rule.ns/sampling["fork_steps"])
    np.savez_compressed(directory / "theory.npz", **theory)
    rows = point_rows(rule, condition, seed, t, ages, m, theory, empirical, pairs, sampling["fork_steps"])
    with (directory / "points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # A plotting run ignores snapshots interrupted before this completion marker.
    write_json(directory / "complete.json", {"complete": True})


def run_experiment(config, rules, output):
    """Train once per seed/rule in the selected condition and inspect snapshots."""
    sampling = config["sampling"]
    condition = config["condition"]
    dtype = jnp.float64 if config["dtype"] == "float64" else jnp.float32
    independent_cache = {}
    for rule in rules:
        for seed in config["seeds"]:
            init_key = stream_key(seed, rule.nd, condition, 1)
            train_key = stream_key(seed, rule.nd, condition, 2)
            w = initialize_weights(init_key, rule, dtype)
            trained = 0
            for t in config["stages"]:
                print(f"{rule.name} / {condition} / seed {seed} / t={t}", flush=True)
                ages = [age for age in config["ages"] if age < t]
                skipped = [age for age in config["ages"] if age >= t]
                if skipped:
                    print(f"  skipping unavailable ages (age >= stage): {skipped}", flush=True)
                if not ages:
                    print("  skipping measurement: no valid probe ages", flush=True)
                    # The next measured stage still trains through all intervening inputs.
                    continue
                while trained < t:
                    size = min(sampling["train_chunk_steps"], t - trained)
                    w = train_segment(w, train_key, trained, rule=rule, condition=condition, steps=size)
                    trained += size
                # Regenerate exactly the actual acquisition patterns from indexed keys.
                old = jnp.stack([pattern(indexed_key(train_key, t-age-1), rule, condition, dtype)
                                 for age in ages])
                weights, old_np = np.asarray(w, dtype=float), np.asarray(old, dtype=float)
                if not np.isfinite(weights).all():
                    raise FloatingPointError("Training produced nonfinite weights")
                m = overlaps(weights, old_np, condition)
                # Disjoint supports give zero physical cross-overlaps, regardless of local ordering.
                q = weights @ weights.T if condition == "shared" else (
                    np.diag(np.sum(weights**2, axis=-1)) if rule.is_decay else np.eye(rule.nd))
                reusable_theory = condition == "independent" and not rule.is_decay
                if reusable_theory and rule in independent_cache:
                    theory = dict(independent_cache[rule])
                    theory["A"] = m @ theory["H"].T
                    theory["batch_A"] = -0.5 * theory["batch_C"][:, 0, 0, None, None] * m[None]
                else:
                    sequence = [sampling["theory_seed"], RULE_NAMES.index(rule.name), rule.nd,
                                0 if condition == "independent" else 1,
                                0 if reusable_theory else seed,
                                0 if reusable_theory else t]
                    theory = theory_moments(rule, condition, q, m, np.random.default_rng(sequence),
                                            sampling["theory_samples"], sampling["uncertainty_groups"],
                                            sampling["theory_batch_size"])
                    if reusable_theory:
                        independent_cache[rule] = theory
                pairs = choose_pairs(theory["C"], sampling["max_pairs"])
                fork_key = indexed_key(stream_key(seed, rule.nd, condition, 3), t)
                empirical = measure_forks(w, old, fork_key, rule, condition, sampling)
                snapshot = output / f"{rule.name}_{condition}_seed{seed}_t{t}"
                save_snapshot(snapshot, rule, condition, seed, t, old_np, weights,
                              ages, m, q, theory, empirical, pairs, sampling)
                print(f"  saved {snapshot.name}", flush=True)


def main(argv=None):
    """Validate before allocating output or launching any expensive computation."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path, help="New run directory; must not already exist")
    parser.add_argument("--forks", type=int, help="Override fixed fork count")
    parser.add_argument("--fork-steps", type=int, help="Override updates per fork (1, 10, 20, ...)")
    args = parser.parse_args(argv)
    try:
        raw = json.loads(args.config.read_text())
        for name in ("forks", "fork_steps"):
            if getattr(args, name) is not None:
                raw.setdefault("sampling", {})[name] = getattr(args, name)
        config, rules = resolve_config(raw)
        if args.validate_only:
            print(json.dumps(config, indent=2))
            return 0
        jax.config.update("jax_enable_x64", config["dtype"] == "float64")
        output = args.output or Path(config["output_root"]) / datetime.now(timezone.utc).strftime("fig1_%Y%m%dT%H%M%S_%fZ")
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "config.json", config)
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False)
        write_json(output / "environment.json", dict(python=platform.python_version(), numpy=np.__version__,
                   jax=jax.__version__, devices=[str(device) for device in jax.devices()],
                   git_revision=revision.stdout.strip(), git_status=status.stdout,
                   started_utc=datetime.now(timezone.utc).isoformat()))
        print(f"Output: {output}", flush=True)
        run_experiment(config, rules, output)
        write_json(output / "complete.json", {"complete": True})
    except (ValueError, OSError, FloatingPointError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
