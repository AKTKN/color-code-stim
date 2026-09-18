"""Small reproducible assessment; scores are computed only by the package API.

Run: python examples/evaluate_path_gap.py --smoke --output /tmp/path-gap-smoke
Larger runs: --distances 3 5 7 9 --probabilities .01 .03 .05 --shots 10000
"""

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import subprocess
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.proportion import proportion_confint

from color_code_stim import ColorCode, NoiseModel
from color_code_stim.metrics import ColorCodePathGap


def git(*args):
    return subprocess.check_output(
        ["git", *args], cwd=Path(__file__).resolve().parents[1], text=True
    ).strip()


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def physical_checks(cc):
    """Independent physical incidence for validation/exact class enumeration."""
    data = sorted(cc.qubit_groups["data"], key=lambda v: v["qid"])
    supports = [
        {v["qid"] for v in check.neighbors() if v["pauli"] is None}
        for check in cc.qubit_groups["anc_Z"]
    ]
    h = np.array(
        [[v["qid"] in support for v in data] for support in supports], dtype=np.uint8
    )
    logical = np.array([v["obs"] for v in data], dtype=np.uint8)
    return h, logical


def exact_quantities(errors, correction, weights, h, logical):
    """All 2^7 physical patterns, not graph paths, supply signed class minima."""
    patterns = np.array(
        list(itertools.product((0, 1), repeat=len(weights))), dtype=np.uint8
    )
    syndromes = patterns @ h.T % 2
    classes = patterns @ logical % 2
    minima = {}
    for syndrome, label, weight in zip(syndromes, classes, patterns @ weights):
        key = (tuple(syndrome), int(label))
        minima[key] = min(minima.get(key, float("inf")), float(weight))
    rows = []
    for syndrome, label, weight in zip(
        errors @ h.T % 2, correction @ logical % 2, correction @ weights
    ):
        same = minima[(tuple(syndrome), int(label))]
        opposite = minima[(tuple(syndrome), 1 - int(label))]
        rows.append(
            {
                "delta_class": opposite - same,
                "delta_corr": opposite - weight,
                "within_class_suboptimality": weight - same,
                "optimal_class_gap": abs(opposite - same),
            }
        )
    return rows


def rate_record(failures, probabilities, mask, exact):
    mass = float(probabilities[mask].sum())
    rate = float(probabilities[mask & failures].sum() / mass) if mass else None
    record = {
        "retained_fraction": mass,
        "abort_fraction": 1 - mass,
        "conditional_failure": rate,
        "selected_patterns_or_shots": int(mask.sum()),
    }
    if exact:
        record.update(ci_low=None, ci_high=None)
    else:
        low, high = (
            proportion_confint(
                int(failures[mask].sum()), int(mask.sum()), alpha=0.05, method="wilson"
            )
            if mask.any()
            else (None, None)
        )
        record.update(ci_low=low, ci_high=high)
    return record


def score_tables(scores, failures, probabilities, exact):
    thresholds, conditional, matched = [], [], []
    for name, score in scores.items():
        for threshold in np.unique(score):
            thresholds.append(
                {
                    "score": name,
                    "threshold": float(threshold),
                    **rate_record(failures, probabilities, score >= threshold, exact),
                }
            )
            conditional.append(
                {
                    "score": name,
                    "value": float(threshold),
                    **rate_record(failures, probabilities, score == threshold, exact),
                }
            )
        # MC shot ID is independent of sampled errors. For exact enumeration,
        # NEVER rank bit-pattern IDs within a tie: randomize the entire tied
        # score group with the same acceptance probability for every pattern.
        order = np.lexsort((np.arange(len(score)), -score))
        for retained in (1.0, 0.9, 0.75, 0.5):
            if exact:
                mass = np.zeros(len(score))
                remaining = retained
                for value in np.unique(score)[::-1]:
                    mask = score == value
                    group_mass = float(probabilities[mask].sum())
                    fraction = min(1.0, max(0.0, remaining / group_mass))
                    mass[mask] = fraction * probabilities[mask]
                    remaining -= fraction * group_mass
                actual = float(mass.sum())
                matched.append(
                    {
                        "score": name,
                        "target_retained": retained,
                        "retained_fraction": actual,
                        "abort_fraction": 1 - actual,
                        "conditional_failure": float(mass @ failures / actual),
                        "ci_low": None,
                        "ci_high": None,
                    }
                )
            else:
                mask = np.zeros(len(score), dtype=bool)
                mask[order[: max(1, int(retained * len(score)))]] = True
                record = rate_record(failures, probabilities, mask, False)
                record.pop("selected_patterns_or_shots")
                matched.append({"score": name, "target_retained": retained, **record})
    return thresholds, conditional, matched


def plot_assessment(
    folder, scores, thresholds, conditional, exact_rows, probabilities, title
):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for name, values in scores.items():
        rows = [r for r in thresholds if r["score"] == name]
        (line,) = axes[0].plot(
            [r["abort_fraction"] for r in rows],
            [r["conditional_failure"] for r in rows],
            ".-",
            label=name,
        )
        if rows[0]["ci_low"] is not None:
            axes[0].fill_between(
                [r["abort_fraction"] for r in rows],
                [r["ci_low"] for r in rows],
                [r["ci_high"] for r in rows],
                alpha=0.15,
                color=line.get_color(),
            )
        rows = [r for r in conditional if r["score"] == name]
        axes[1].plot(
            [r["value"] for r in rows],
            [r["conditional_failure"] for r in rows],
            ".-",
            label=name,
        )
        if rows[0]["ci_low"] is not None:
            axes[1].fill_between(
                [r["value"] for r in rows],
                [r["ci_low"] for r in rows],
                [r["ci_high"] for r in rows],
                alpha=0.15,
                color=line.get_color(),
            )
        unique = np.unique(values)
        axes[2].plot(
            unique, [probabilities[values == x].sum() for x in unique], ".-", label=name
        )
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    axes[0].set(
        xlabel="Abort fraction (whole score ties)",
        ylabel="Conditional logical failure",
    )
    axes[1].set(xlabel="Score", ylabel="Conditional logical failure")
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(bottom=0)
    axes[2].set(xlabel="Score", ylabel="Probability mass")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(folder / "assessment.png", dpi=140)
    plt.close(fig)
    if exact_rows:
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
        for ax, name in zip(axes, exact_rows[0]):
            ax.scatter(scores["phi"], [r[name] for r in exact_rows], s=12, alpha=0.5)
            ax.set(xlabel="phi", ylabel=name)
            ax.grid(alpha=0.2)
        fig.suptitle(
            "d=3 exact class diagnostics (points are patterns, not probability-weighted)"
        )
        fig.tight_layout()
        fig.savefig(folder / "exact_diagnostics.png", dpi=140)
        plt.close(fig)


def run_cohort(folder, d, p, seed, errors, exact, comparative):
    folder.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    cc = ColorCode(
        d=d,
        rounds=1,
        circuit_type="tri",
        cnot_schedule="tri_optimal",
        temp_bdry_type="Z",
        noise_model=NoiseModel(bitflip=p),
        comparative_decoding=comparative,
    )
    code_setup = perf_counter() - start
    start = perf_counter()
    adapter = ColorCodePathGap(cc)
    topology_mapping_setup = perf_counter() - start
    h, logical = physical_checks(cc)
    # Convert sampled PHYSICAL errors into decoder DEM coordinates. This is
    # simulation/validation only: the metric sees only returned corrections.
    start = perf_counter()
    dem_errors = np.empty_like(errors)
    dem_errors[:, adapter.mapping.qubit_to_column] = errors
    dets = np.asarray(dem_errors @ cc.H.T % 2, dtype=bool)
    obs = errors @ logical % 2
    if comparative:
        dets[:, -1] = False
    detector_setup = perf_counter() - start
    start = perf_counter()
    pred, extra = cc.decode(dets, full_output=True)
    decoding = perf_counter() - start
    saved_preds = extra["error_preds"].copy()
    start = perf_counter()
    correction = adapter.to_physical(extra).astype(np.uint8)
    mapping = perf_counter() - start
    start = perf_counter()
    result = adapter.evaluator.evaluate(correction)
    metric = perf_counter() - start
    np.testing.assert_array_equal(extra["error_preds"], saved_preds)
    np.testing.assert_array_equal(correction @ h.T % 2, errors @ h.T % 2)
    np.testing.assert_array_equal(correction @ logical % 2, pred)
    failures = pred != obs
    probabilities = (
        p ** errors.sum(axis=1) * (1 - p) ** (errors.shape[1] - errors.sum(axis=1))
        if exact
        else np.full(len(errors), 1 / len(errors))
    )
    if not np.isclose(probabilities.sum(), 1):
        raise AssertionError("Exhaustive physical probability does not sum to one")
    scores = {
        "phi": result.phi,
        "D_min": result.distance_by_color.min(axis=1),
        "minus_W": -result.correction_weight,
    }
    if "logical_gaps" in extra:
        scores["decoder_comparative_gap"] = extra["logical_gaps"]
    exact_rows = (
        exact_quantities(errors, correction, adapter.evaluator.weights, h, logical)
        if d == 3
        else []
    )
    rows = []
    for i in range(len(errors)):
        row = {
            "shot_id": i,
            "probability": float(probabilities[i]),
            "physical_error": "".join(map(str, errors[i])),
            "final_correction": "".join(map(str, correction[i])),
            "hard_prediction": int(pred[i]),
            "failure": int(failures[i]),
            "D_r": result.distance_by_color[i, 0],
            "D_g": result.distance_by_color[i, 1],
            "D_b": result.distance_by_color[i, 2],
            "W": result.correction_weight[i],
            "minimizing_color": result.minimizing_color[i],
        }
        row.update({name: values[i] for name, values in scores.items()})
        if exact_rows:
            row.update(exact_rows[i])
        rows.append(row)
    thresholds, conditional, matched = score_tables(
        scores, failures, probabilities, exact
    )
    write_csv(folder / "shots.csv", rows)
    write_csv(folder / "postselection.csv", thresholds)
    write_csv(folder / "conditional.csv", conditional)
    write_csv(folder / "equal_retention.csv", matched)
    mode = "comparative" if comparative else "ordinary"
    plot_assessment(
        folder,
        scores,
        thresholds,
        conditional,
        exact_rows,
        probabilities,
        f"d={d}, p={p}, {mode}; {'exact probabilities' if exact else 'smoke MC, 95% Wilson intervals'}",
    )
    summary = {
        "distance": d,
        "p": p,
        "seed": seed,
        "shots_or_patterns": len(errors),
        "exact": exact,
        "decoder_mode": mode,
        "rounds": 1,
        "circuit_type": "tri",
        "cnot_schedule": "tri_optimal",
        "memory": "Z",
        "qubit_ids": adapter.qubit_ids,
        "physical_errors_sha256": hashlib.sha256(errors.tobytes()).hexdigest(),
        "negative_phi_count": int((result.phi < 0).sum()),
        "negative_phi_probability": float(probabilities[result.phi < 0].sum()),
        **rate_record(failures, probabilities, np.ones(len(errors), dtype=bool), exact),
        "timing_seconds": {
            "code_setup": code_setup,
            "topology_and_mapping_setup": topology_mapping_setup,
            **adapter.setup_timings,
            "detector_and_decoder_setup": detector_setup,
            "decoding": decoding,
            "final_correction_mapping": mapping,
            "metric_evaluation": metric,
        },
        "exact_weighted_means": (
            {
                name: float(probabilities @ np.array([r[name] for r in exact_rows]))
                for name in exact_rows[0]
            }
            if exact_rows
            else None
        ),
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="d=3 exhaustive and d=5 256 shots, p=.03, both modes",
    )
    parser.add_argument("--distances", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--probabilities", type=float, nargs="+", default=[0.03])
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["ordinary", "comparative"],
        default=["ordinary", "comparative"],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.smoke:
        args.distances, args.probabilities, args.shots, args.modes = (
            [3, 5],
            [0.03],
            256,
            ["ordinary", "comparative"],
        )
    if (
        args.shots < 1
        or any(d < 3 or d % 2 == 0 for d in args.distances)
        or any(not 0 < p < 0.5 for p in args.probabilities)
    ):
        parser.error("Need positive shots, odd distances >=3, and 0<p<1/2")
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for d, p in itertools.product(args.distances, args.probabilities):
        n = (3 * d * d + 1) // 4
        rng = np.random.default_rng(
            np.random.SeedSequence([args.seed, d, int(round(p * 1e9))])
        )
        errors = (
            np.array(list(itertools.product((0, 1), repeat=n)), dtype=np.uint8)
            if d == 3
            else (rng.random((args.shots, n)) < p).astype(np.uint8)
        )
        for mode in args.modes:
            name = f"d{d}_p{p:g}_{mode}"
            summaries.append(
                run_cohort(
                    args.output / name,
                    d,
                    p,
                    args.seed,
                    errors,
                    d == 3,
                    mode == "comparative",
                )
            )
    source = (
        Path(__file__).resolve().parents[1]
        / "src/color_code_stim/metrics/monochromatic_path_gap.py"
    )
    metadata = {
        "base_commit": git("merge-base", "HEAD", "origin/main"),
        "implementation_commit": git("rev-parse", "HEAD"),
        "source_dirty": bool(
            git(
                "status",
                "--porcelain",
                "--",
                "src",
                "tests",
                "examples/evaluate_path_gap.py",
            )
        ),
        "metric_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "stim",
                "pymatching",
                "igraph",
                "statsmodels",
                "color-code-stim",
            )
        },
        "configuration": {**vars(args), "output": str(args.output)},
        "cohorts": summaries,
        "interpretation": "Smoke evaluation only; no statistically resolved superiority claim. Comparative gap is decoder-derived, not exact or posterior odds.",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cohorts": [
                    {
                        k: s[k]
                        for k in (
                            "distance",
                            "decoder_mode",
                            "conditional_failure",
                            "negative_phi_count",
                        )
                    }
                    for s in summaries
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
