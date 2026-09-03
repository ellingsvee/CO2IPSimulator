# CO2IPSimulator

CO2IPSimulator is a reduced-order invasion-percolation simulator for CO2 migration in layered reservoirs. The repository also contains an approximate Bayesian computation (ABC-SMC) inference framework for parameter estimation. Synthetic and Sleipner experiments are included, and the Sleipner data is included in the repository.

## Repository layout

Some of the key directories are:

- `src/co2ipsimulator/model/`: Python simulator interface and grid construction.
- `src/rust/`: Performance-critical spill and trap-filling code.
- `src/co2ipsimulator/inference/`: Forward-model, summary-statistic, prior, and ABC-SMC components.
- `examples/`: Synthetic and Sleipner experiments. See [`examples/README.md`](examples/README.md) for how to run.

## Requirements and setup

The project requires:

- [Python 3.13](https://www.python.org/downloads/) or newer
- [uv](https://docs.astral.sh/uv/)
- [Rust](https://rust-lang.org/tools/install/)

From the repository root:

```bash
uv sync --locked
uv run maturin develop --release
```

The release build is recommended for the inference experiments, which evaluate the simulator many times. The committed `uv.lock` and `Cargo.lock` files define the dependency versions used by the project.

## Validation

```bash
# Python tests, linter, and formatter
uv run pytest
uv run ruff check src examples tests
uv run ruff format --check src examples tests

# Rust tests, linter, and formatter
cargo fmt --all -- --check
cargo clippy --all-targets --all-features
cargo test --all-features
```

Run `uv run pytest -m slow` to validate the Sleipner geometry and mass balance when the separately distributed field data are available.

The experiment TOML files contain the sampling settings, random seeds, model parameters, and output locations used for the paper.

## Citation and license

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). The source code is released under the [BSD 3-Clause License](LICENSE). The bundled Sleipner data have separate terms; see [`examples/sleipner/data/README.md`](examples/sleipner/data/README.md).
