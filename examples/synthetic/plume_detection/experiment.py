from __future__ import annotations

from dataclasses import dataclass, replace

from co2ipsimulator.inference import RunConfig

from ..comparison import Variant, experiment_root

EXPERIMENT_ROOT = "plume_detection"

DEFAULT_TRUTH_THRESHOLD_M = 3.0


@dataclass(frozen=True)
class DetectionCase:
    """One analysis of one synthetic survey.

    The survey registers a cell only once its CO2 column exceeds the truth
    threshold, which is a property of the data. ``model_threshold_m`` is what the
    analyst assumes instead; ``None`` marginalises over it, which needs an
    ``[inference.detection]`` prior in the config.
    """

    variant: Variant
    model_threshold_m: float | None

    @property
    def infers(self) -> bool:
        return self.model_threshold_m is None


def default_cases(truth_threshold_m: float) -> tuple[DetectionCase, ...]:
    return (
        DetectionCase(Variant("m0", "Assume 0 m"), 0.0),
        DetectionCase(
            Variant("m3", f"Assume {truth_threshold_m:g} m"), truth_threshold_m
        ),
        DetectionCase(Variant("m8", "Assume 8 m"), 8.0),
        DetectionCase(Variant("inf", "Infer"), None),
    )


def truth_threshold(base: RunConfig) -> float:
    return float(
        base.extras.get("truth_detection_threshold_m", DEFAULT_TRUTH_THRESHOLD_M)
    )


def cases_from_config(base: RunConfig) -> tuple[DetectionCase, ...]:
    truth = truth_threshold(base)
    raw = base.extras.get("detection_cases")
    if not raw:
        return default_cases(truth)
    return tuple(
        DetectionCase(
            variant=Variant(str(entry["key"]), str(entry["label"])),
            model_threshold_m=(
                None if entry.get("model") is None else float(entry["model"])
            ),
        )
        for entry in raw
    )


def case_run_config(base: RunConfig, case: DetectionCase) -> RunConfig:
    """Specialise the run config to one case.

    A case that assumes a threshold drops the detection prior even when the
    config declares one, so the same file can hold both the prior the inferring
    case uses and the fixed values the others assume.
    """
    root = experiment_root(base, EXPERIMENT_ROOT)
    extras = {
        **base.extras,
        "truth_detection_threshold_m": truth_threshold(base),
        "experiment": f"{root}/{case.variant.key}",
    }
    if case.infers:
        if base.inference.detection_prior is None:
            raise SystemExit(
                f"case {case.variant.key!r} infers the detection threshold but the "
                "config has no [inference.detection] section"
            )
        return replace(base, extras=extras)
    extras["detection_threshold_m"] = case.model_threshold_m
    inference = replace(base.inference, detection_prior=None)
    return replace(base, inference=inference, extras=extras)


def variant_configs(base: RunConfig, cases: tuple[DetectionCase, ...]) -> dict:
    return {
        case.variant.key: (case.variant, case_run_config(base, case)) for case in cases
    }
