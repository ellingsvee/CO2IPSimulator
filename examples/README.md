# Experiments

Run all commands from the repository root. The tracked TOML files are the configurations used for the experiments. Generated arrays, diagnostics, and figures are written below `examples/synthetic/output` or `examples/sleipner/output`.

## Synthetic experiments

The dome and Gaussian-random-field (GRF) scenarios use the same two-stage workflow. Replace `grf` with `dome` to run the other scenario.

```bash
uv run python -m examples.synthetic.fit \
  --config examples/synthetic/configs/grf.toml
uv run python -m examples.synthetic.forecast \
  --config examples/synthetic/configs/grf.toml
```

The transfer-model study compares finite-rate and quasi-static model choices:

```bash
uv run python -m examples.synthetic.transfer_model.run \
  --config examples/synthetic/transfer_model/configs/grf.toml
```

The plume-detection study compares assumed seismic detection thresholds:

```bash
uv run python -m examples.synthetic.plume_detection.run \
  --config examples/synthetic/plume_detection/configs/grf.toml
```

Both comparison commands accept `--plot-only` to redraw figures from completed runs. The plume-detection command also accepts `--operator-only` to generate only the detection-operator diagnostics.

## Sleipner experiments

Fit the Sleipner model and then produce its posterior forecast using the same configuration:

```bash
uv run python -m examples.sleipner.fit \
  --config examples/sleipner/configs/abc_mass_and_footprint.toml
uv run python -m examples.sleipner.forecast \
  --config examples/sleipner/configs/abc_mass_and_footprint.toml
```

Alternative mass-and-footprint and quasi-static configurations are in the same directory.

The observation-time comparison runs its required fits and forecasts together:

```bash
uv run python -m examples.sleipner.observation_time.run \
  --config examples/sleipner/configs/observation_time.toml \
  --reuse-existing
```

Use `--plot-only` to redraw the comparison from completed outputs. Individual saved runs can be replotted with `examples.synthetic.replot` or `examples.sleipner.replot`. Use `--help` for their input options.
