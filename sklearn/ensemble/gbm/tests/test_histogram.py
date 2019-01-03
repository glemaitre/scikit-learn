import numpy as np
import pytest

from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal

from sklearn.ensemble.gbm.histogram import _build_histogram_naive
from sklearn.ensemble.gbm.histogram import _build_histogram
from sklearn.ensemble.gbm.histogram import _build_histogram_no_hessian
from sklearn.ensemble.gbm.histogram import _build_histogram_root_no_hessian
from sklearn.ensemble.gbm.histogram import _build_histogram_root
from sklearn.ensemble.gbm.histogram import _subtract_histograms
from sklearn.ensemble.gbm.types import HISTOGRAM_DTYPE


@pytest.mark.parametrize(
    'build_func', [_build_histogram_naive, _build_histogram])
def test_build_histogram(build_func):
    binned_feature = np.array([0, 2, 0, 1, 2, 0, 2, 1], dtype=np.uint8)

    # Small sample_indices (below unrolling threshold)
    ordered_gradients = np.array([0, 1, 3], dtype=np.float32)
    ordered_hessians = np.array([1, 1, 2], dtype=np.float32)

    sample_indices = np.array([0, 2, 3], dtype=np.uint32)
    hist = np.zeros(3, dtype=HISTOGRAM_DTYPE)
    build_func(3, sample_indices, binned_feature, ordered_gradients,
               ordered_hessians, hist)
    assert_array_equal(hist['count'], [2, 1, 0])
    assert_allclose(hist['sum_gradients'], [1, 3, 0])
    assert_allclose(hist['sum_hessians'], [2, 2, 0])

    # Larger sample_indices (above unrolling threshold)
    sample_indices = np.array([0, 2, 3, 6, 7], dtype=np.uint32)
    ordered_gradients = np.array([0, 1, 3, 0, 1], dtype=np.float32)
    ordered_hessians = np.array([1, 1, 2, 1, 0], dtype=np.float32)

    hist = np.zeros(3, dtype=HISTOGRAM_DTYPE)
    build_func(3, sample_indices, binned_feature, ordered_gradients,
               ordered_hessians, hist)
    assert_array_equal(hist['count'], [2, 2, 1])
    assert_allclose(hist['sum_gradients'], [1, 4, 0])
    assert_allclose(hist['sum_hessians'], [2, 2, 1])


def test_histogram_sample_order_independence():
    rng = np.random.RandomState(42)
    n_sub_samples = 100
    n_samples = 1000
    n_bins = 256

    binned_feature = rng.randint(0, n_bins - 1, size=n_samples, dtype=np.uint8)
    sample_indices = rng.choice(np.arange(n_samples, dtype=np.uint32),
                                n_sub_samples, replace=False)
    ordered_gradients = rng.randn(n_sub_samples).astype(np.float32)
    hist_gc = np.zeros(n_bins, dtype=HISTOGRAM_DTYPE)
    _build_histogram_no_hessian(n_bins, sample_indices, binned_feature,
                                ordered_gradients, hist_gc)

    ordered_hessians = rng.exponential(size=n_sub_samples).astype(np.float32)
    hist_ghc = np.zeros(n_bins, dtype=HISTOGRAM_DTYPE)
    _build_histogram(n_bins, sample_indices, binned_feature,
                     ordered_gradients, ordered_hessians, hist_ghc)

    permutation = rng.permutation(n_sub_samples)
    hist_gc_perm = np.zeros(n_bins, dtype=HISTOGRAM_DTYPE)
    _build_histogram_no_hessian(n_bins, sample_indices[permutation],
                                binned_feature, ordered_gradients[permutation],
                                hist_gc_perm)

    hist_ghc_perm = np.zeros(n_bins, dtype=HISTOGRAM_DTYPE)
    _build_histogram(n_bins, sample_indices[permutation], binned_feature,
                     ordered_gradients[permutation],
                     ordered_hessians[permutation], hist_ghc_perm)

    assert_allclose(hist_gc['sum_gradients'], hist_gc_perm['sum_gradients'])
    assert_array_equal(hist_gc['count'], hist_gc_perm['count'])

    assert_allclose(hist_ghc['sum_gradients'], hist_ghc_perm['sum_gradients'])
    assert_allclose(hist_ghc['sum_hessians'], hist_ghc_perm['sum_hessians'])
    assert_array_equal(hist_ghc['count'], hist_ghc_perm['count'])


@pytest.mark.parametrize("constant_hessian", [True, False])
def test_unrolled_equivalent_to_naive(constant_hessian):
    # Make sure the different unrolled histogram computations give the same
    # results as the naive one.
    rng = np.random.RandomState(42)
    n_samples = 10
    n_bins = 5
    sample_indices = np.arange(n_samples).astype(np.uint32)
    binned_feature = rng.randint(0, n_bins - 1, size=n_samples, dtype=np.uint8)
    ordered_gradients = rng.randn(n_samples).astype(np.float32)
    if constant_hessian:
        ordered_hessians = np.ones(n_samples, dtype=np.float32)
    else:
        ordered_hessians = rng.lognormal(size=n_samples).astype(np.float32)

    hist_gc_root = np.zeros(n_bins, dtype=HISTOGRAM_DTYPE)
    hist_ghc_root = np.zeros(n_bins, dtype=HISTOGRAM_DTYPE)
    hist_gc = np.zeros(n_bins, dtype=HISTOGRAM_DTYPE)
    hist_ghc = np.zeros(n_bins, dtype=HISTOGRAM_DTYPE)
    hist_naive = np.zeros(n_bins, dtype=HISTOGRAM_DTYPE)

    _build_histogram_root_no_hessian(n_bins, binned_feature, ordered_gradients,
                                     hist_gc_root)
    _build_histogram_root(n_bins, binned_feature, ordered_gradients,
                          ordered_hessians, hist_ghc_root)
    _build_histogram_no_hessian(n_bins, sample_indices, binned_feature,
                                ordered_gradients, hist_gc)
    _build_histogram(n_bins, sample_indices, binned_feature,
                     ordered_gradients, ordered_hessians, hist_ghc)
    _build_histogram_naive(n_bins, sample_indices, binned_feature,
                           ordered_gradients, ordered_hessians, hist_naive)

    for hist in (hist_gc_root, hist_ghc_root, hist_gc, hist_gc, hist_ghc):
        assert_array_equal(hist['count'], hist_naive['count'])
        assert_allclose(hist['sum_gradients'], hist_naive['sum_gradients'])
    for hist in (hist_ghc_root, hist_ghc):
        assert_allclose(hist['sum_hessians'], hist_naive['sum_hessians'])
    for hist in (hist_gc_root, hist_gc):
        assert_array_equal(hist['sum_hessians'], np.zeros(n_bins))


@pytest.mark.parametrize("constant_hessian", [True, False])
def test_hist_subtraction(constant_hessian):
    # Make sure the histogram subtraction trick gives the same result as the
    # classical method.
    rng = np.random.RandomState(42)
    n_samples = 10
    n_bins = 5
    sample_indices = np.arange(n_samples).astype(np.uint32)
    binned_feature = rng.randint(0, n_bins - 1, size=n_samples, dtype=np.uint8)
    ordered_gradients = rng.randn(n_samples).astype(np.float32)
    if constant_hessian:
        ordered_hessians = np.ones(n_samples, dtype=np.float32)
    else:
        ordered_hessians = rng.lognormal(size=n_samples).astype(np.float32)

    hist_parent = np.zeros(n_bins, dtype=HISTOGRAM_DTYPE)
    if constant_hessian:
        _build_histogram_no_hessian(n_bins, sample_indices, binned_feature,
                                    ordered_gradients, hist_parent)
    else:
         _build_histogram(n_bins, sample_indices, binned_feature,
                          ordered_gradients, ordered_hessians, hist_parent)

    mask = rng.randint(0, 2, n_samples).astype(np.bool)

    sample_indices_left = sample_indices[mask]
    ordered_gradients_left = ordered_gradients[mask]
    ordered_hessians_left = ordered_hessians[mask]
    hist_left = np.zeros(n_bins, dtype=HISTOGRAM_DTYPE)
    if constant_hessian:
        _build_histogram_no_hessian(n_bins, sample_indices_left,
                                    binned_feature, ordered_gradients_left,
                                    hist_left)
    else:
        _build_histogram(n_bins, sample_indices_left, binned_feature,
                         ordered_gradients_left, ordered_hessians_left,
                         hist_left)

    sample_indices_right = sample_indices[~mask]
    ordered_gradients_right = ordered_gradients[~mask]
    ordered_hessians_right = ordered_hessians[~mask]
    hist_right = np.zeros(n_bins, dtype=HISTOGRAM_DTYPE)
    if constant_hessian:
        _build_histogram_no_hessian(n_bins, sample_indices_right,
                                    binned_feature, ordered_gradients_right,
                                    hist_right)
    else:
        _build_histogram(n_bins, sample_indices_right, binned_feature,
                         ordered_gradients_right, ordered_hessians_right,
                         hist_right)

    hist_left_sub = np.zeros(n_bins, dtype=HISTOGRAM_DTYPE)
    hist_right_sub = np.zeros(n_bins, dtype=HISTOGRAM_DTYPE)
    _subtract_histograms(n_bins, hist_parent, hist_right, hist_left_sub)
    _subtract_histograms(n_bins, hist_parent, hist_left, hist_right_sub)

    for key in ('count', 'sum_hessians', 'sum_gradients'):
        assert_allclose(hist_left[key], hist_left_sub[key], rtol=1e-6)
        assert_allclose(hist_right[key], hist_right_sub[key], rtol=1e-6)
