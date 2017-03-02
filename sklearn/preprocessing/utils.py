# Authors: Denis Engemann <denis-alexander.engemann@inria.fr>
#          Guillaume Lemaitre <guillaume.lemaitre@inria.fr>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Raghav RV <rvraghav93@gmail.com>
#          Thierry Guillemot <thierry.guillemot.work@gmail.com>
# License: BSD 3 clause

from __future__ import division

import numpy as np
from scipy import sparse


# FIXME move these functions somewhere else
# the numpy version of these functions do not support sparse matrices
def _hist_bin_sqrt(x):
    """Square root histogram bin estimator

    Bin width is inversely proportional to the data size. Used by many
    programs for its simplicity.

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    if sparse.issparse(x):
        return (max(x.data) - min(x.data)) / np.sqrt(len(x.data))
    else:
        return x.ptp() / np.sqrt(x.size)


def _hist_bin_sturges(x):
    """Sturges histogram bin estimator

    A very simplistic estimator based on the assumption of normality of
    the data. This estimator has poor performance for non-normal data,
    which becomes especially obvious for large data sets. The estimate
    depends only on size of the data.

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    if sparse.issparse(x):
        return (max(x.data) - min(x.data)) / (np.log2(len(x.data)) + 1)
    else:
        return x.ptp() / (np.log2(x.size) + 1)


def _hist_bin_rice(x):
    """Rice histogram bin estimator

    Another simple estimator with no normality assumption. It has better
    performance for large data than Sturges, but tends to overestimate
    the number of bins. The number of bins is proportional to the cube
    root of data size (asymptotically optimal). The estimate depends
    only on size of the data.

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    if sparse.issparse(x):
        return (max(x.data) - min(x.data)) / (2 * len(x.data) ** (1 / 3))
    else:
        return x.ptp() / (2 * x.size ** (1 / 3))


def _hist_bin_scott(x):
    """Scott histogram bin estimator

    The binwidth is proportional to the standard deviation of the data
    and inversely proportional to the cube root of data size
    (asymptotically optimal).

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    if sparse.issparse(x):
        return (24 * np.pi ** 0.5 / len(x.data)) ** (1 / 3) * np.std(x.data)
    else:
        return (24 * np.pi**0.5 / x.size) ** (1 / 3) * np.std(x)


def _hist_bin_doane(x):
    """Doane's histogram bin estimator

    Improved version of Sturges' formula which works better for
    non-normal data.

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    if sparse.issparse(x):
        if len(x.data) > 2:
            x_size = len(x.data)
            sg1 = np.sqrt(6 * (x_size - 2) / ((x_size + 1) * (x_size + 3)))
            sigma = np.std(x.data)
            if sigma > 0:
                # These three operations add up to
                # g1 = np.mean(((x - np.mean(x)) / sigma)**3)
                # but use only one temp array instead of three
                x_mean = np.mean(x.data)
                temp = x.data - x_mean
                np.true_divide(temp, sigma, temp)
                np.power(temp, 3, temp)
                g1 = np.mean(temp)
                return ((max(x.data) - min(x.data)) /
                        (1 + np.log2(x_size) +
                         np.log2(1 + np.absolute(g1) / sg1)))
        return 0.0
    else:
        if x.size > 2:
            sg1 = np.sqrt(6 * (x.size - 2) / ((x.size + 1) * (x.size + 3)))
            sigma = np.std(x)
            if sigma > 0:
                # These three operations add up to
                # g1 = np.mean(((x - np.mean(x)) / sigma)**3)
                # but use only one temp array instead of three
                temp = x - np.mean(x)
                np.true_divide(temp, sigma, temp)
                np.power(temp, 3, temp)
                g1 = np.mean(temp)
                return x.ptp() / (1 + np.log2(x.size) +
                                  np.log2(1 + np.absolute(g1) / sg1))
        return 0.0


def _hist_bin_fd(x):
    """The Freedman-Diaconis histogram bin estimator

    The Freedman-Diaconis rule uses interquartile range (IQR) to
    estimate binwidth. It is considered a variation of the Scott rule
    with more robustness as the IQR is less affected by outliers than
    the standard deviation. However, the IQR depends on fewer points
    than the standard deviation, so it is less accurate, especially for
    long tailed distributions.
    If the IQR is 0, this function returns 1 for the number of bins.
    Binwidth is inversely proportional to the cube root of data size
    (asymptotically optimal).

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    if sparse.issparse(x):
        iqr = np.subtract(*np.percentile(x.data, [75, 25]))
        return 2 * iqr * len(x.data) ** (-1 / 3)
    else:
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        return 2 * iqr * x.size ** (-1 / 3)


def _hist_bin_auto(x):
    """Histogram bin estimator that uses the minimum width of the
    Freedman-Diaconis and Sturges estimators.

    The FD estimator is usually the most robust method, but its width
    estimate tends to be too large for small `x`. The Sturges estimator
    is quite good for small (<1000) datasets and is the default in the R
    language. This method gives good off the shelf behaviour.

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    See Also
    --------
    _hist_bin_fd, _hist_bin_sturges
    """
    # There is no need to check for zero here. If ptp is, so is IQR and
    # vice versa. Either both are zero or neither one is.
    return min(_hist_bin_fd(x), _hist_bin_sturges(x))
