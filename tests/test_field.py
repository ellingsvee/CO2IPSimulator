import numpy as np

from co2ipsimulator.field import GRFKernel, sample_grf_2d


def test_grf_sampler_is_reproducible_with_requested_shape_and_scale():
    kernel = GRFKernel(
        sigma=10.0,
        correlation_length_m=500.0,
        kernel="matern",
        nu=1.5,
    )

    first = sample_grf_2d(48, 64, dx=50.0, dy=50.0, kernel=kernel, rng=7)
    second = sample_grf_2d(48, 64, dx=50.0, dy=50.0, kernel=kernel, rng=7)

    assert first.shape == (64, 48)
    assert np.array_equal(first, second)
    assert np.isclose(first.std(), 10.0)
    assert abs(float(first.mean())) < 5.0
