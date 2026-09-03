# Experiments

Run all commands from the repository root. The tracked TOML files are the configurations used for the experiments. Generated arrays, diagnostics, and figures are written below `examples/synthetic/output` or `examples/sleipner/output`.

## Synthetic experiments

The dome and Gaussian-random-field (GRF) scenarios use the same two-stage workflow. Replace `grf` with `dome` to run the other scenario.

```bash
# First fit the model to estimate parameters
uv run python -m examples.synthetic.fit --config examples/synthetic/configs/grf.toml

# Use the fitted parameters to forecast the plume evolution
uv run python -m examples.synthetic.forecast --config examples/synthetic/configs/grf.toml
```

The transfer-model study compares finite-rate and quasi-static model choices:

```bash
uv run python -m examples.synthetic.transfer_model.run --config examples/synthetic/transfer_model/configs/grf.toml
```

The plume-detection study compares assumed seismic detection thresholds:

```bash
uv run python -m examples.synthetic.plume_detection.run --config examples/synthetic/plume_detection/configs/grf.toml
```


## Sleipner experiments

Fit the Sleipner model and then produce its posterior forecast using the same configuration:

```bash
uv run python -m examples.sleipner.fit --config examples/sleipner/configs/abc_mass_and_footprint.toml
uv run python -m examples.sleipner.forecast --config examples/sleipner/configs/abc_mass_and_footprint.toml
```


The observation-time comparison runs its required fits and forecasts together:

```bash
uv run python -m examples.sleipner.observation_time.run --config examples/sleipner/configs/observation_time.toml \
  --reuse-existing
```

