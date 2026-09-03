from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from .configs import InferenceConfig
from .forward import ForwardDiagnostics, ForwardModelConfig, forward_snapshots


def prior_draws(inference: InferenceConfig, count: int, seed: int = 0) -> np.ndarray:
    """``count`` independent draws from the prior, one row per draw."""
    distributions = [prior.distribution for prior in inference.pth_priors]
    if inference.rate_limit_prior is not None:
        distributions.append(inference.rate_limit_prior.distribution)
    if inference.detection_prior is not None:
        distributions.append(inference.detection_prior.distribution)
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [distribution.sample(rng, count) for distribution in distributions]
    )


@dataclass(frozen=True)
class ForwardScan:
    """What a batch of forward runs could not account for.

    A draw is only usable if the time integration met its tolerance and the
    simulator held on to the mass it was given. Reporting these over the prior
    before sampling is what lets a run that reaches the substep cap, or that
    loses mass out of the model, be identified rather than silently accepted by
    the ABC.
    """

    draws: int
    unconverged_draws: int
    leaking_draws: int
    stalled_draws: int
    worst_relative_error: float
    worst_escaped_fraction: float
    worst_stalled_fraction: float

    @property
    def is_clean(self) -> bool:
        return (
            self.unconverged_draws == 0
            and self.leaking_draws == 0
            and self.stalled_draws == 0
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"is_clean": self.is_clean}

    def report(self) -> str:
        if self.is_clean:
            return (
                f"forward scan: {self.draws} prior draws all converged "
                f"(worst error {self.worst_relative_error:.2e}) and lost no mass"
            )
        return (
            f"forward scan: {self.draws} prior draws, "
            f"{self.unconverged_draws} hit the substep cap "
            f"(worst error {self.worst_relative_error:.2e}), "
            f"{self.leaking_draws} lost mass "
            f"(worst {100 * self.worst_escaped_fraction:.2f}%), "
            f"{self.stalled_draws} stalled in the spill graph "
            f"(worst {100 * self.worst_stalled_fraction:.2f}%)"
        )


def summarize_forward_diagnostics(
    diagnostics: Sequence[ForwardDiagnostics],
) -> ForwardScan:
    return ForwardScan(
        draws=len(diagnostics),
        unconverged_draws=sum(1 for d in diagnostics if not d.converged),
        leaking_draws=sum(1 for d in diagnostics if d.escaped_fraction > 0.0),
        stalled_draws=sum(1 for d in diagnostics if d.stalled_fraction > 0.0),
        worst_relative_error=max(
            (d.worst_relative_error for d in diagnostics), default=0.0
        ),
        worst_escaped_fraction=max(
            (d.escaped_fraction for d in diagnostics), default=0.0
        ),
        worst_stalled_fraction=max(
            (d.stalled_fraction for d in diagnostics), default=0.0
        ),
    )


def scan_forward_model(
    config: ForwardModelConfig,
    snapshot_years: tuple[int, ...],
    draws: np.ndarray,
) -> ForwardScan:
    return summarize_forward_diagnostics(
        [
            forward_snapshots(config, parameters, snapshot_years)[1]
            for parameters in draws
        ]
    )
