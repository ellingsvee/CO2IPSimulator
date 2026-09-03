from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from co2ipsimulator.inference import RunConfig


@dataclass(frozen=True)
class ObservationTimeModel:
    key: str
    label: str
    snapshot_years: tuple[int, ...]


MODELS = (
    # ObservationTimeModel("both", "2010 and 2023 observations", (2010, 2023)),
    # ObservationTimeModel("2010_only", "2010 observation only", (2010,)),
    ObservationTimeModel("both", "2010 and 2023", (2010, 2023)),
    ObservationTimeModel("2010_only", "Only 2010", (2010,)),
)


def experiment_root(base: RunConfig) -> Path:
    return Path(
        base.extras.get(
            "observation_time_output_dir",
            "examples/sleipner/output/observation_time",
        )
    )


def model_run_config(base: RunConfig, model: ObservationTimeModel) -> RunConfig:
    root = experiment_root(base)
    output_dir = root / model.key
    if model.key == "both":
        output_dir = Path(base.extras.get("both_observation_fit_dir", output_dir))
    extras = {
        **base.extras,
        "snapshot_years": list(model.snapshot_years),
        "output_dir": str(output_dir),
    }
    return replace(base, extras=extras)


def model_configs(
    base: RunConfig,
) -> tuple[tuple[ObservationTimeModel, RunConfig], ...]:
    return tuple((model, model_run_config(base, model)) for model in MODELS)
