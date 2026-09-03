from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from co2ipsimulator.inference import (
    build_forward_model,
    load_run_config,
    posterior_samples,
    prior_draws,
    run_abc_smc,
    scan_forward_model,
    summarize_posterior,
)

from examples.synthetic.experiment import prior_distributions_and_log
from examples.synthetic.plot import plot_recovery

from .experiment import build_forward_config, build_observed
from .posterior_correlation import write_posterior_correlation


def output_dir(run_config) -> Path:
    return Path(run_config.extras.get("output_dir", "examples/sleipner/output/abc"))


SCAN_DRAWS = 64


def run(run_config) -> Path:
    config = build_forward_config(run_config)
    statistics, observed, epsilon, _ = build_observed(config, run_config)
    forward_model = build_forward_model(config, statistics)

    # Before sampling, check the forward model over the prior: a draw that runs
    # out of substeps, or that loses mass out of the model, would otherwise be
    # accepted by the ABC without any sign that it was not solved.
    scan = scan_forward_model(
        config,
        statistics.snapshot_years,
        prior_draws(run_config.inference, SCAN_DRAWS, seed=run_config.pymc.seed),
    )
    print(scan.report())

    idata = run_abc_smc(
        run_config.inference, run_config.pymc, forward_model, epsilon, observed
    )
    samples = posterior_samples(idata)
    names = run_config.inference.parameter_names
    summary = summarize_posterior(idata, names)

    outdir = output_dir(run_config)
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "abc_posterior_parameter_samples.npy", samples)
    np.save(outdir / "abc_observed_summary.npy", observed)
    np.save(outdir / "abc_epsilon.npy", epsilon)
    (outdir / "abc_summary_names.txt").write_text("\n".join(statistics.names))
    (outdir / "abc_parameter_names.txt").write_text("\n".join(names))
    np.savetxt(
        outdir / "abc_posterior_parameter_samples.csv",
        samples,
        delimiter=",",
        header=",".join(names),
        comments="",
    )

    _, is_log = prior_distributions_and_log(run_config)
    diagnostics = {
        "n_samples": int(samples.shape[0]),
        "parameter_names": list(names),
        "estimates": [
            {
                "name": estimate.name,
                "median": estimate.median,
                "std": estimate.standard_deviation,
                "lower": estimate.lower,
                "upper": estimate.upper,
            }
            for estimate in summary.estimates
        ],
    }
    (outdir / "abc_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    (outdir / "abc_forward_diagnostics.json").write_text(
        json.dumps(scan.as_dict(), indent=2)
    )

    print(f"\nsleipner posterior ({samples.shape[0]} samples):")
    for index, estimate in enumerate(summary.estimates):
        med, lo, hi = estimate.median, estimate.lower, estimate.upper
        if is_log[index]:
            med, lo, hi = (np.log10(v) for v in (med, lo, hi))
            label = f"log10({estimate.name})"
        else:
            label = estimate.name
        print(f"  {label:>14}: median={med:8.3f}  90% CI [{lo:8.3f}, {hi:8.3f}]")

    distributions, is_log = prior_distributions_and_log(run_config)
    plot_recovery(
        names,
        samples,
        distributions,
        is_log,
        ncols=3,
        output=outdir / "abc_posterior_hist.pdf",
        threshold_xlim=(
            (0.0, 200.0)
            if run_config.inference.rate_limit_prior is None
            else (0.0, 150.0)
        ),
    )
    write_posterior_correlation(samples, names, is_log, output_dir=outdir)
    print(f"  -> {outdir}")
    return outdir


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit one Sleipner ABC-SMC experiment.")
    parser.add_argument(
        "--config", type=Path, required=True, help="Experiment TOML file."
    )
    args = parser.parse_args()
    run(load_run_config(args.config))


if __name__ == "__main__":
    main()
