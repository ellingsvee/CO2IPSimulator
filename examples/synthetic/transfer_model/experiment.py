from __future__ import annotations

from dataclasses import replace

from co2ipsimulator.inference import RunConfig

from ..comparison import Variant, experiment_root
from ..experiment import FINITE_RATE, QUASI_STATIC

EXPERIMENT_ROOT = "transfer_model"

TRANSFERS = (FINITE_RATE, QUASI_STATIC)

TRANSFER_KEY = {FINITE_RATE: "rate_truth", QUASI_STATIC: "quasi_truth"}
TRANSFER_LABEL = {
    FINITE_RATE: "Finite-rate transfer present",
    QUASI_STATIC: "Finite-rate transfer absent",
}

MODELS = (
    Variant("rate_model", "Finite-rate model"),
    Variant("quasi_model", "Quasi-static IP model"),
)


def model_run_config(base: RunConfig, transfer: str, model: Variant) -> RunConfig:
    """One cell of the (data-generating transfer law, fitted model) grid.

    The truth is generated with ``transfer`` whatever the fitted model is, so a
    quasi-static fit to rate-limited data is a genuine misspecification rather
    than a relabelled truth. Both cells of a row share priors, survey schedule
    and tolerance, so only the fitted physics differs.
    """
    if base.inference.rate_limit_prior is None:
        raise SystemExit("the base config must declare [inference.rate_limit]")
    root = experiment_root(base, EXPERIMENT_ROOT)
    extras = {
        **base.extras,
        "truth_transfer": transfer,
        "experiment": f"{root}/{TRANSFER_KEY[transfer]}/{model.key}",
    }
    rate_limit = base.inference.rate_limit_prior if model.key == "rate_model" else None
    inference = replace(base.inference, rate_limit_prior=rate_limit)
    return replace(base, inference=inference, extras=extras)


def variant_configs(base: RunConfig, transfer: str) -> dict:
    return {
        model.key: (model, model_run_config(base, transfer, model)) for model in MODELS
    }
