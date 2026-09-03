from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from co2ipsimulator.inference import (
    build_forward_model,
    load_run_config,
    posterior_samples,
    run_abc_smc,
    summarize_posterior,
)

from .experiment import (
    build_forward_config,
    build_observed,
    calibration_years,
    prior_distributions_and_log,
    truth_parameters,
)
from .plot import plot_recovery
from .scenarios import get_scenario

OUTPUT_ROOT = Path("examples/synthetic/output")


def output_dir(scenario, run_config) -> Path:
    experiment = run_config.extras.get("experiment", scenario.name)
    return OUTPUT_ROOT / scenario.name / experiment


def run(scenario, run_config) -> Path:
    end_year = max(calibration_years(scenario, run_config))
    config = build_forward_config(scenario, run_config, end_year)
    truth = truth_parameters(scenario, run_config)
    statistics, observed, epsilon = build_observed(config, scenario, run_config)
    forward_model = build_forward_model(config, statistics)

    idata = run_abc_smc(
        run_config.inference, run_config.pymc, forward_model, epsilon, observed
    )
    samples = posterior_samples(idata)
    names = run_config.inference.parameter_names
    summary = summarize_posterior(idata, names)

    outdir = output_dir(scenario, run_config)
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "abc_posterior_parameter_samples.npy", samples)
    np.save(outdir / "abc_observed_summary.npy", observed)
    np.save(outdir / "abc_epsilon.npy", epsilon)
    np.save(outdir / "true_parameters.npy", truth)
    (outdir / "abc_summary_names.txt").write_text("\n".join(statistics.names))
    (outdir / "abc_parameter_names.txt").write_text("\n".join(names))
    np.savetxt(
        outdir / "abc_posterior_parameter_samples.csv",
        samples,
        delimiter=",",
        header=",".join(names),
        comments="",
    )

    diagnostics = {
        "scenario": scenario.name,
        "n_samples": int(samples.shape[0]),
        "parameter_names": list(names),
        "truth": truth.tolist(),
        "estimates": [
            {
                "name": estimate.name,
                "truth": float(truth[index]),
                "mean": estimate.mean,
                "median": estimate.median,
                "std": estimate.standard_deviation,
                "lower": estimate.lower,
                "upper": estimate.upper,
            }
            for index, estimate in enumerate(summary.estimates)
        ],
    }
    (outdir / "abc_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))

    print(f"\n{scenario.name} posterior ({samples.shape[0]} samples):")
    _, is_log = prior_distributions_and_log(run_config)
    for index, estimate in enumerate(summary.estimates):
        t, med, lo, hi = (truth[index], estimate.median, estimate.lower, estimate.upper)
        if is_log[index]:
            t, med, lo, hi = (np.log10(v) for v in (t, med, lo, hi))
            label = f"log10({estimate.name})"
        else:
            label = estimate.name
        print(
            f"  {label:>14}: truth={t:8.3f}  median={med:8.3f}  90% CI [{lo:8.3f}, {hi:8.3f}]"
        )

    distributions, is_log = prior_distributions_and_log(run_config)
    plot_recovery(
        names,
        samples,
        distributions,
        is_log,
        truth=truth,
        output=outdir / "abc_recovery_hist.pdf",
    )
    print(f"  -> {outdir}")
    return outdir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit one synthetic ABC-SMC experiment."
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Experiment TOML file."
    )
    args = parser.parse_args()
    run_config = load_run_config(args.config)
    scenario = get_scenario(run_config.extras["scenario"])
    run(scenario, run_config)


if __name__ == "__main__":
    main()
