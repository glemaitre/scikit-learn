import warnings

from ..base import BaseEstimator, TransformerMixin
from ..utils import check_array, safe_indexing
from ..externals.six import string_types


def _identity(X):
    """The identity function.
    """
    return X


class FunctionTransformer(BaseEstimator, TransformerMixin):
    """Constructs a transformer from an arbitrary callable.

    A FunctionTransformer forwards its X (and optionally y) arguments to a
    user-defined function or function object and returns the result of this
    function. This is useful for stateless transformations such as taking the
    log of frequencies, doing custom scaling, etc.

    Note: If a lambda is used as the function, then the resulting
    transformer will not be pickleable.

    .. versionadded:: 0.17

    Read more in the :ref:`User Guide <function_transformer>`.

    Parameters
    ----------
    func : callable, optional default=None
        The callable to use for the transformation. This will be passed
        the same arguments as transform, with args and kwargs forwarded.
        If validate is True, func needs to return a 2-dimensional array.
        If func is None, then func will be the identity function.

    inverse_func : callable, optional default=None

        The callable to use for the inverse transformation. This will be
        passed the same arguments as inverse transform, with args and
        kwargs forwarded. If validate is True, inverse_func needs to return
        a 2-dimensional array. If inverse_func is None, then inverse_func will
        be the identity function.

    validate : bool, optional default=True
        Indicate that the input X and transformed output arrays should be
        checked before calling ``transform``. If validate is false, there will
        be no input validation. If it is true, then X will be converted to a
        2-dimensional NumPy array or sparse matrix. If this conversion is not
        possible or X contains NaN or infinity, an exception is raised. In
        addition, the output of func and inverse_func are checked to return a
        2-dimensional array.

    accept_sparse : boolean, optional
        Indicate that func accepts a sparse matrix as input. If validate is
        False, this has no effect. Otherwise, if accept_sparse is false,
        sparse matrix inputs will cause an exception to be raised.

    pass_y : bool, optional default=False
        Indicate that transform should forward the y argument to the
        inner callable.

        .. deprecated::0.19

    kw_args : dict, optional
        Dictionary of additional keyword arguments to pass to func.

    inv_kw_args : dict, optional
        Dictionary of additional keyword arguments to pass to inverse_func.

    """
    def __init__(self, func=None, inverse_func=None, validate=True,
                 accept_sparse=False, pass_y='deprecated',
                 kw_args=None, inv_kw_args=None):
        self.func = func
        self.inverse_func = inverse_func
        self.validate = validate
        self.accept_sparse = accept_sparse
        self.pass_y = pass_y
        self.kw_args = kw_args
        self.inv_kw_args = inv_kw_args

    def _check_functions(self, X):
        """Check that transform and inverse_transform lead to a 2D array"""
        idx_selected = slice(None, None, max(1, X.shape[0] // 100))
        msg = (" transforms a 2D array into a 1D array"
               " which do not follow the estimator API of"
               " scikit-learn. If you are sure you want to"
               " proceed, set 'validate=False'. This warning will"
               " be turned to an error in 0.22")
        if (self.func is not None and
                self.transform(safe_indexing(X, idx_selected)).ndim != 2):
            warnings.warn("'func'" + msg, FutureWarning)
        if (self.inverse_func is not None and
            self.inverse_transform(
                self.transform(safe_indexing(X, idx_selected))).ndim != 2):
            warnings.warn("'inverse_func'" + msg, FutureWarning)

    def fit(self, X, y=None):
        """Fit transformer by checking X.

        If ``validate`` is ``True``, ``X`` will be checked.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input array.

        Returns
        -------
        self
        """
        if self.validate:
            X = check_array(X, self.accept_sparse)
            self._check_functions(X)
        return self

    def transform(self, X, y='deprecated'):
        """Transform X using the forward function.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input array.

        y : (ignored)
            .. deprecated::0.19

        Returns
        -------
        X_out : array-like, shape (n_samples, n_features)
            Transformed input.
        """
        if not isinstance(y, string_types) or y != 'deprecated':
            warnings.warn("The parameter y on transform() is "
                          "deprecated since 0.19 and will be removed in 0.21",
                          DeprecationWarning)

        return self._transform(X, y=y, func=self.func, kw_args=self.kw_args)

    def inverse_transform(self, X, y='deprecated'):
        """Transform X using the inverse function.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input array.

        y : (ignored)
            .. deprecated::0.19

        Returns
        -------
        X_out : array-like, shape (n_samples, n_features)
            Transformed input.
        """
        if not isinstance(y, string_types) or y != 'deprecated':
            warnings.warn("The parameter y on inverse_transform() is "
                          "deprecated since 0.19 and will be removed in 0.21",
                          DeprecationWarning)
        return self._transform(X, y=y, func=self.inverse_func,
                               kw_args=self.inv_kw_args)

    def _transform(self, X, y=None, func=None, kw_args=None):
        if self.validate:
            X = check_array(X, self.accept_sparse)

        if func is None:
            func = _identity

        if (not isinstance(self.pass_y, string_types) or
                self.pass_y != 'deprecated'):
            # We do this to know if pass_y was set to False / True
            pass_y = self.pass_y
            warnings.warn("The parameter pass_y is deprecated since 0.19 and "
                          "will be removed in 0.21", DeprecationWarning)
        else:
            pass_y = False

        return func(X, *((y,) if pass_y else ()),
                    **(kw_args if kw_args else {}))
