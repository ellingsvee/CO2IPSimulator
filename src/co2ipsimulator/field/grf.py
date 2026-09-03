from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

KernelName = Literal["gaussian", "matern"]


@dataclass(frozen=True)
class GRFKernel:
    sigma: float
    correlation_length_m: float
    kernel: KernelName = "matern"
    nu: float = 1.5

    def __post_init__(self) -> None:
        if self.sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {self.sigma!r}")
        if self.correlation_length_m <= 0:
            raise ValueError(
                f"correlation_length_m must be > 0, got {self.correlation_length_m!r}"
            )
        if self.kernel not in ("gaussian", "matern"):
            raise ValueError(f"unknown kernel {self.kernel!r}")
        if self.kernel == "matern" and self.nu <= 0:
            raise ValueError(f"nu must be > 0, got {self.nu!r}")


def _covariance_values(
    r: np.ndarray,
    *,
    kernel: KernelName,
    nu: float,
    correlation_length_m: float,
) -> np.ndarray:
    L = float(correlation_length_m)
    if kernel == "gaussian":
        return np.exp(-0.5 * (r / L) ** 2)
    if kernel == "matern":
        if np.isclose(nu, 0.5):
            return np.exp(-r / L)
        if np.isclose(nu, 1.5):
            s = np.sqrt(3.0) * r / L
            return (1.0 + s) * np.exp(-s)
        if np.isclose(nu, 2.5):
            s = np.sqrt(5.0) * r / L
            return (1.0 + s + s * s / 3.0) * np.exp(-s)
        from scipy.special import gamma, kv

        s = np.sqrt(2.0 * nu) * r / L
        out = np.empty_like(r)
        out[r == 0] = 1.0
        mask = r > 0
        out[mask] = (2.0 ** (1.0 - nu) / gamma(nu)) * (s[mask] ** nu) * kv(nu, s[mask])
        return out
    raise ValueError(f"unknown kernel {kernel!r}")


def _grid_lags(
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    pad_factor: int = 2,
) -> tuple[np.ndarray, tuple[int, int]]:
    if pad_factor < 1:
        raise ValueError(f"pad_factor must be >= 1, got {pad_factor!r}")
    NX = nx * pad_factor
    NY = ny * pad_factor
    ix = np.arange(NX) * dx
    iy = np.arange(NY) * dy
    ix = np.minimum(ix, NX * dx - ix)
    iy = np.minimum(iy, NY * dy - iy)
    X, Y = np.meshgrid(ix, iy, indexing="xy")
    return np.sqrt(X * X + Y * Y), (NY, NX)


def _embedding_eigenvalues(
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    kernel: GRFKernel,
    pad_factor: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    r, shape = _grid_lags(nx, ny, dx, dy, pad_factor=pad_factor)
    cov = _covariance_values(
        r,
        kernel=kernel.kernel,
        nu=kernel.nu,
        correlation_length_m=kernel.correlation_length_m,
    )
    return np.fft.fft2(cov).real, shape


def sample_grf_2d(
    nx: int,
    ny: int,
    *,
    dx: float,
    dy: float,
    kernel: GRFKernel,
    rng: np.random.Generator | int | None = None,
    pad_factor: int = 2,
    auto_pad: bool = True,
    max_pad_factor: int = 8,
    rel_eigenvalue_tol: float = 1.0e-5,
) -> np.ndarray:
    if kernel.sigma == 0.0:
        return np.zeros((ny, nx), dtype=np.float64)

    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    pad = max(1, int(pad_factor))
    eig, (NY, NX) = _embedding_eigenvalues(nx, ny, dx, dy, kernel, pad)
    while auto_pad and pad < int(max_pad_factor):
        min_eig, max_eig = float(eig.min()), float(eig.max())
        if min_eig >= -rel_eigenvalue_tol * max(max_eig, 1.0):
            break
        pad += 1
        eig, (NY, NX) = _embedding_eigenvalues(nx, ny, dx, dy, kernel, pad)

    min_eig, max_eig = float(eig.min()), float(eig.max())
    if min_eig < -rel_eigenvalue_tol * max(max_eig, 1.0):
        import warnings

        warnings.warn(
            f"circulant embedding still non-PSD at pad_factor={pad} "
            f"(min eigenvalue {min_eig:.3e} vs max {max_eig:.3e})",
            RuntimeWarning,
            stacklevel=2,
        )
    eig_clipped = np.maximum(eig, 0.0)

    xi = (gen.standard_normal((NY, NX)) + 1j * gen.standard_normal((NY, NX))) / np.sqrt(
        2.0
    )
    spectrum = np.sqrt(eig_clipped) * xi
    field = np.fft.ifft2(spectrum) * np.sqrt(NY * NX)
    sample = field.real[:ny, :nx]

    empirical_std = float(sample.std())
    if empirical_std > 0.0:
        sample = sample * (kernel.sigma / empirical_std)
    return np.ascontiguousarray(sample, dtype=np.float64)


def empirical_correlogram(
    field: np.ndarray, *, dx: float, dy: float, max_lag_m: float, n_bins: int = 12
) -> tuple[np.ndarray, np.ndarray]:
    f = field - field.mean()
    ny, nx = f.shape
    F = np.fft.fft2(f)
    acov = np.fft.ifft2(np.abs(F) ** 2).real / (nx * ny)
    iy = np.arange(ny)
    ix = np.arange(nx)
    iy = np.minimum(iy, ny - iy)
    ix = np.minimum(ix, nx - ix)
    X, Y = np.meshgrid(ix * dx, iy * dy, indexing="xy")
    r = np.sqrt(X * X + Y * Y)
    bins = np.linspace(0.0, float(max_lag_m), n_bins + 1)
    centres = 0.5 * (bins[:-1] + bins[1:])
    rho = np.zeros(n_bins, dtype=np.float64)
    var = acov.flat[0]
    if var <= 0:
        return centres, rho
    for b in range(n_bins):
        mask = (r >= bins[b]) & (r < bins[b + 1])
        if mask.any():
            rho[b] = float(acov[mask].mean() / var)
    return centres, rho
