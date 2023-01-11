"""Compatibility fixes for older version of python, numpy and scipy

If you add content to this file, please give the version of the package
at which the fix is no longer needed.
"""
# Authors: Emmanuelle Gouillart <emmanuelle.gouillart@normalesup.org>
#          Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Fabian Pedregosa <fpedregosa@acm.org>
#          Lars Buitinck
#
# License: BSD 3 clause

from functools import update_wrapper
from importlib import resources
import functools
import sys
import warnings

import sklearn
import numpy as np
import joblib
import scipy
import scipy.stats
import threadpoolctl
from .._config import config_context, get_config
from ..externals._packaging.version import parse as parse_version


np_version = parse_version(np.__version__)
sp_version = parse_version(scipy.__version__)


if sp_version >= parse_version("1.4"):
    from scipy.sparse.linalg import lobpcg
else:
    # Backport of lobpcg functionality from scipy 1.4.0, can be removed
    # once support for sp_version < parse_version('1.4') is dropped
    # mypy error: Name 'lobpcg' already defined (possibly by an import)
    from ..externals._lobpcg import lobpcg  # type: ignore  # noqa

try:
    from scipy.optimize._linesearch import line_search_wolfe2, line_search_wolfe1
except ImportError:  # SciPy < 1.8
    from scipy.optimize.linesearch import line_search_wolfe2, line_search_wolfe1  # type: ignore  # noqa


def _object_dtype_isnan(X):
    return X != X


class loguniform(scipy.stats.reciprocal):
    """A class supporting log-uniform random variables.

    Parameters
    ----------
    low : float
        The minimum value
    high : float
        The maximum value

    Methods
    -------
    rvs(self, size=None, random_state=None)
        Generate log-uniform random variables

    The most useful method for Scikit-learn usage is highlighted here.
    For a full list, see
    `scipy.stats.reciprocal
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.reciprocal.html>`_.
    This list includes all functions of ``scipy.stats`` continuous
    distributions such as ``pdf``.

    Notes
    -----
    This class generates values between ``low`` and ``high`` or

        low <= loguniform(low, high).rvs() <= high

    The logarithmic probability density function (PDF) is uniform. When
    ``x`` is a uniformly distributed random variable between 0 and 1, ``10**x``
    are random variables that are equally likely to be returned.

    This class is an alias to ``scipy.stats.reciprocal``, which uses the
    reciprocal distribution:
    https://en.wikipedia.org/wiki/Reciprocal_distribution

    Examples
    --------

    >>> from sklearn.utils.fixes import loguniform
    >>> rv = loguniform(1e-3, 1e1)
    >>> rvs = rv.rvs(random_state=42, size=1000)
    >>> rvs.min()  # doctest: +SKIP
    0.0010435856341129003
    >>> rvs.max()  # doctest: +SKIP
    9.97403052786026
    """


# TODO: remove when the minimum scipy version is >= 1.5
if sp_version >= parse_version("1.5"):
    from scipy.linalg import eigh as _eigh  # noqa
else:

    def _eigh(*args, **kwargs):
        """Wrapper for `scipy.linalg.eigh` that handles the deprecation of `eigvals`."""
        eigvals = kwargs.pop("subset_by_index", None)
        return scipy.linalg.eigh(*args, eigvals=eigvals, **kwargs)


def _with_config(delayed_func, config):
    """Helper function that intends to attach a config to a delayed function."""
    if hasattr(delayed_func, "with_config"):
        return delayed_func.with_config(config)
    else:
        warnings.warn(
            "You are using `sklearn.utils.fixes.Parallel` that intend to attached a "
            "configuration to a delayed function. However, the function used for "
            "delaying the function does not expose `with_config`. Use "
            "`sklearn.utils.fixes.delayed` for this purpose. A default configuration "
            "is used instead.",
            UserWarning,
        )
        return delayed_func


class Parallel(joblib.Parallel):
    # A `Parallel` tweaked class that allows attaching a configuration to each task
    # to be run in parallel.

    def __call__(self, iterable):
        # Capture the thread-local scikit-learn configuration at the time
        # Parallel.__call__ is issued since the tasks can be dispatched
        # in a different thread depending on the backend and on the value of
        # pre_dispatch and n_jobs.
        config = get_config()
        iterable_with_config = (
            (_with_config(delayed_func, config), args, kwargs)
            for delayed_func, args, kwargs in iterable
        )
        return super().__call__(iterable_with_config)


# remove when https://github.com/joblib/joblib/issues/1071 is fixed
def delayed(function):
    """Decorator used to capture the arguments of a function."""

    @functools.wraps(function)
    def delayed_function(*args, **kwargs):
        return _FuncWrapper(function), args, kwargs

    return delayed_function


class _FuncWrapper:
    """ "Load the global configuration before calling the function."""

    def __init__(self, function):
        self.function = function
        update_wrapper(self, self.function)

    def with_config(self, config):
        self.config = config
        return self

    def __call__(self, *args, **kwargs):
        config = getattr(self, "config", None)
        if config is None:
            warnings.warn(
                "You are using `sklearn.utils.fixes.delayed` without using "
                "`sklearn.utils.fixes.Parallel`. A default configuration is used "
                "instead of propagating the user defined configuration.",
                UserWarning,
            )
            config = get_config()
        with config_context(**config):
            return self.function(*args, **kwargs)


# Rename the `method` kwarg to `interpolation` for NumPy < 1.22, because
# `interpolation` kwarg was deprecated in favor of `method` in NumPy >= 1.22.
def _percentile(a, q, *, method="linear", **kwargs):
    return np.percentile(a, q, interpolation=method, **kwargs)


if np_version < parse_version("1.22"):
    percentile = _percentile
else:  # >= 1.22
    from numpy import percentile  # type: ignore  # noqa


# compatibility fix for threadpoolctl >= 3.0.0
# since version 3 it's possible to setup a global threadpool controller to avoid
# looping through all loaded shared libraries each time.
# the global controller is created during the first call to threadpoolctl.
def _get_threadpool_controller():
    if not hasattr(threadpoolctl, "ThreadpoolController"):
        return None

    if not hasattr(sklearn, "_sklearn_threadpool_controller"):
        sklearn._sklearn_threadpool_controller = threadpoolctl.ThreadpoolController()

    return sklearn._sklearn_threadpool_controller


def threadpool_limits(limits=None, user_api=None):
    controller = _get_threadpool_controller()
    if controller is not None:
        return controller.limit(limits=limits, user_api=user_api)
    else:
        return threadpoolctl.threadpool_limits(limits=limits, user_api=user_api)


threadpool_limits.__doc__ = threadpoolctl.threadpool_limits.__doc__


def threadpool_info():
    controller = _get_threadpool_controller()
    if controller is not None:
        return controller.info()
    else:
        return threadpoolctl.threadpool_info()


threadpool_info.__doc__ = threadpoolctl.threadpool_info.__doc__


# TODO: Remove when SciPy 1.9 is the minimum supported version
def _mode(a, axis=0):
    if sp_version >= parse_version("1.9.0"):
        return scipy.stats.mode(a, axis=axis, keepdims=True)
    return scipy.stats.mode(a, axis=axis)


###############################################################################
# Backport of Python 3.9's importlib.resources
# TODO: Remove when Python 3.9 is the minimum supported version


def _open_text(data_module, data_file_name):
    if sys.version_info >= (3, 9):
        return resources.files(data_module).joinpath(data_file_name).open("r")
    else:
        return resources.open_text(data_module, data_file_name)


def _open_binary(data_module, data_file_name):
    if sys.version_info >= (3, 9):
        return resources.files(data_module).joinpath(data_file_name).open("rb")
    else:
        return resources.open_binary(data_module, data_file_name)


def _read_text(descr_module, descr_file_name):
    if sys.version_info >= (3, 9):
        return resources.files(descr_module).joinpath(descr_file_name).read_text()
    else:
        return resources.read_text(descr_module, descr_file_name)


def _path(data_module, data_file_name):
    if sys.version_info >= (3, 9):
        return resources.as_file(resources.files(data_module).joinpath(data_file_name))
    else:
        return resources.path(data_module, data_file_name)


def _is_resource(data_module, data_file_name):
    if sys.version_info >= (3, 9):
        return resources.files(data_module).joinpath(data_file_name).is_file()
    else:
        return resources.is_resource(data_module, data_file_name)
