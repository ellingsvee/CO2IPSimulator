from __future__ import annotations

import pytest

from co2ipsimulator.inference import (
    LogNormalPrior,
    MassMode,
    NormalPrior,
    SummaryMode,
    load_run_config,
)


def test_run_config_loads_inference_summary_and_sampler_settings(tmp_path):
    path = tmp_path / "inference.toml"
    path.write_text(
        "[pymc]\n"
        "draws = 25\n"
        "progressbar = false\n"
        "[inference.threshold_pressure]\n"
        'layer_names = ["A", "B"]\n'
        'distribution = "normal"\n'
        "mu = [45.0, 55.0]\n"
        "sigma = [4.0, 6.0]\n"
        "[inference.rate_limit]\n"
        'name = "lambda_s"\n'
        'distribution = "lognormal"\n'
        "mu = -25.0\n"
        "sigma = 1.0\n"
        "[summary]\n"
        'mode = "mass-and-footprints"\n'
        'mass_mode = "mt"\n'
        "mass_relative_epsilon = 0.2\n"
        "[extras]\n"
        'scenario = "custom"\n'
    )

    config = load_run_config(path)

    assert config.pymc.draws == 25
    assert config.pymc.progressbar is False
    assert config.summary.mode is SummaryMode.MASS_AND_FOOTPRINTS
    assert config.summary.mass_mode is MassMode.MT
    assert isinstance(config.inference.pth_priors[0].distribution, NormalPrior)
    assert config.inference.pth_priors[0].distribution.mu == 45.0
    assert config.inference.pth_priors[1].distribution.sigma == 6.0
    assert isinstance(config.inference.rate_limit_prior.distribution, LogNormalPrior)
    assert config.inference.parameter_names == ("A", "B", "lambda_s")
    assert config.extras == {"scenario": "custom"}


def test_run_config_rejects_unknown_settings(tmp_path):
    path = tmp_path / "stale.toml"
    path.write_text(
        "[inference.threshold_pressure]\n"
        'layer_names = ["A"]\n'
        'distribution = "lognormal"\n'
        "mu = [4.0]\n"
        "sigma = [0.5]\n"
        "[summary]\n"
        "area_epsilon = 0.05\n"
    )

    with pytest.raises(ValueError, match="area_epsilon"):
        load_run_config(path)
