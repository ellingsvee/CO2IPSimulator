from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from co2ipsimulator.inference import load_run_config, run_forward_model

from .experiment import build_forward_config, build_observed

PTH_BOUNDS_KPA = (10.0, 200.0)
LOG10_MOBILITY_BOUNDS = (-14.0, -9.0)


def _distance(simulated, observed, epsilon):
    return float(np.sum(((simulated - observed) / epsilon) ** 2))


def optimize(run_config, *, maxiter: int = 25, popsize: int = 10, seed: int = 0):
    config = build_forward_config(run_config)
    statistics, observed, epsilon, _ = build_observed(config, run_config)
    n_pth = len(run_config.inference.pth_priors)
    estimate_t = run_config.inference.rate_limit_prior is not None
    bounds = [PTH_BOUNDS_KPA] * n_pth + ([LOG10_MOBILITY_BOUNDS] if estimate_t else [])

    def to_params(theta):
        pth = theta[:n_pth]
        return np.array(list(pth) + ([10.0 ** theta[n_pth]] if estimate_t else []))

    def objective(theta):
        return _distance(
            run_forward_model(config, statistics, to_params(theta)), observed, epsilon
        )

    result = differential_evolution(
        objective,
        bounds,
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        tol=1e-4,
        polish=True,
        workers=1,
    )
    best = to_params(result.x)
    simulated = run_forward_model(config, statistics, best)
    return result, best, simulated, observed, epsilon, statistics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find a diagnostic best fit for a Sleipner configuration."
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Experiment TOML file."
    )
    parser.add_argument(
        "--maxiter", type=int, default=25, help="Optimizer iteration limit."
    )
    args = parser.parse_args()
    run_config = load_run_config(args.config)
    result, best, simulated, observed, epsilon, statistics = optimize(
        run_config, maxiter=args.maxiter
    )

    names = run_config.inference.parameter_names

    print(f"\nbest distance = {result.fun:.1f}  ({result.nfev} forward evals)")
    print("best parameters:")
    for name, value in zip(names, best):
        shown = (
            f"log10(lambda)={np.log10(value):.3f}"
            if name == "lambda_s"
            else f"{value:.1f} kPa"
        )
        print(f"  {name:>12}: {shown}")
    print(f"\n{'summary':>34}  {'sim':>10}  {'obs':>10}")
    for name, sim, obs in zip(statistics.names, simulated, observed):
        print(f"  {name:>32}  {sim:10.3f}  {obs:10.3f}")

    outdir = Path(run_config.extras.get("output_dir", "examples/sleipner/output/abc"))
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "optimum_parameters.npy", best)
    (outdir / "optimum.json").write_text(
        json.dumps(
            {
                "distance": float(result.fun),
                "parameters": {name: float(v) for name, v in zip(names, best)},
            },
            indent=2,
        )
    )
    print(f"\n  -> {outdir}")


if __name__ == "__main__":
    main()
