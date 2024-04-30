from abc import abstractmethod
from collections.abc import MutableMapping
from inspect import signature
from numbers import Integral, Real

import numpy as np

from ..base import (
    BaseEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
    _fit_context,
    clone,
)
from ..exceptions import NotFittedError
from ..metrics import (
    check_scoring,
    get_scorer_names,
    make_scorer,
    precision_recall_curve,
    roc_curve,
)
from ..metrics._scorer import _BaseScorer
from ..utils import _safe_indexing
from ..utils._param_validation import HasMethods, Interval, RealNotInt, StrOptions
from ..utils._response import _get_response_values_binary
from ..utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    _raise_for_params,
    process_routing,
)
from ..utils.metaestimators import available_if
from ..utils.multiclass import type_of_target
from ..utils.parallel import Parallel, delayed
from ..utils.validation import (
    _check_method_params,
    _num_samples,
    check_is_fitted,
    indexable,
)
from ._split import StratifiedShuffleSplit, check_cv


def _estimator_has(attr):
    """Check if we can delegate a method to the underlying estimator.

    First, we check the fitted estimator if available, otherwise we
    check the unfitted estimator.
    """

    def check(self):
        if hasattr(self, "estimator_"):
            getattr(self.estimator_, attr)
        else:
            getattr(self.estimator, attr)
        return True

    return check


class BaseThresholdClassifier(ClassifierMixin, MetaEstimatorMixin, BaseEstimator):
    """Base class for classifiers that set a non-default decision threshold.

    In this base class, we define the following interface:

    - the validation of common parameters in `fit`;
    - the different prediction methods that can be used with the classifier.

    .. versionadded:: 1.5

    Parameters
    ----------
    estimator : estimator instance
        The classifier, fitted or not, for which we want to optimize
        the decision threshold used during `predict`.

    pos_label : int, float, bool or str, default=None
        The label of the positive class. Used when `objective_metric` is
        `"max_tnr_at_tpr_constraint"`"`, `"max_tpr_at_tnr_constraint"`, or a dictionary.
        When `pos_label=None`, if `y_true` is in `{-1, 1}` or `{0, 1}`,
        `pos_label` is set to 1, otherwise an error will be raised. When using a
        scorer, `pos_label` can be passed as a keyword argument to
        :func:`~sklearn.metrics.make_scorer`.

    response_method : {"auto", "decision_function", "predict_proba"}, default="auto"
        Methods by the classifier `base_estimator` corresponding to the
        decision function for which we want to find a threshold. It can be:

        * if `"auto"`, it will try to invoke, for each classifier,
          `"predict_proba"` or `"decision_function"` in that order.
        * otherwise, one of `"predict_proba"` or `"decision_function"`.
          If the method is not implemented by the classifier, it will raise an
          error.
    """

    _required_parameters = ["estimator"]
    _parameter_constraints: dict = {
        "estimator": [
            HasMethods(["fit", "predict_proba"]),
            HasMethods(["fit", "decision_function"]),
        ],
        "pos_label": [Real, str, "boolean", None],
        "response_method": [StrOptions({"auto", "predict_proba", "decision_function"})],
    }

    def __init__(self, estimator, *, pos_label=None, response_method="auto"):
        self.estimator = estimator
        self.pos_label = pos_label
        self.response_method = response_method

    @_fit_context(
        # *TunedClassifier*.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y, **params):
        """Fit the classifier.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        **params : dict
            Parameters to pass to the `fit` method of the underlying
            classifier and to the `objective_metric` scorer.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        _raise_for_params(params, self, None)

        X, y = indexable(X, y)

        y_type = type_of_target(y, input_name="y")
        if y_type != "binary":
            raise ValueError(
                f"Only binary classification is supported. Unknown label type: {y_type}"
            )

        if self.response_method == "auto":
            self._response_method = ["predict_proba", "decision_function"]
        else:
            self._response_method = self.response_method

        return self._fit(X, y, **params)

    @abstractmethod
    def _get_pos_label(self):
        """Get the positive label."""
        pass

    @abstractmethod
    def _get_decision_threshold(self):
        """Get the decision threshold."""
        pass

    @property
    def classes_(self):
        """Classes labels."""
        return self.estimator_.classes_

    def predict(self, X):
        """Predict the target of new samples.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The samples, as accepted by `estimator.predict`.

        Returns
        -------
        class_labels : ndarray of shape (n_samples,)
            The predicted class.
        """
        check_is_fitted(self, "estimator_")
        pos_label = self._get_pos_label()  # defined in subclasses
        y_score, _ = _get_response_values_binary(
            self.estimator_, X, self._response_method, pos_label=pos_label
        )

        return _threshold_scores_to_class_labels(
            y_score, self._get_decision_threshold(), self.classes_, pos_label
        )

    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X):
        """Predict class probabilities for `X` using the fitted estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        probabilities : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)

    @available_if(_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X):
        """Predict logarithm class probabilities for `X` using the fitted estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        log_probabilities : ndarray of shape (n_samples, n_classes)
            The logarithm class probabilities of the input samples.
        """
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_log_proba(X)

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        """Decision function for samples in `X` using the fitted estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        decisions : ndarray of shape (n_samples,)
            The decision function computed the fitted estimator.
        """
        check_is_fitted(self, "estimator_")
        return self.estimator_.decision_function(X)

    def _more_tags(self):
        return {
            "binary_only": True,
            "_xfail_checks": {
                "check_classifiers_train": "Threshold at probability 0.5 does not hold",
                "check_sample_weights_invariance": (
                    "Due to the cross-validation and sample ordering, removing a sample"
                    " is not strictly equal to putting is weight to zero. Specific unit"
                    " tests are added for TunedThresholdClassifierCV specifically."
                ),
            },
        }


class FixedThresholdClassifier(BaseThresholdClassifier):
    """Classifier that manually sets the decision threshold.

    This classifier allows to change the default decision threshold used for
    converting posterior probability estimates (i.e. output of `predict_proba`) or
    decision scores (i.e. output of `decision_function`) into a class label.

    Here, the threshold is not optimized and is set to a constant value.

    Read more in the :ref:`User Guide <FixedThresholdClassifier>`.

    .. versionadded:: 1.5

    Parameters
    ----------
    estimator : estimator instance
        The classifier, fitted or not, for which we want to optimize
        the decision threshold used during `predict`.

    threshold_value : float, default=0.5
        The decision threshold to use when converting posterior probability estimates
        (i.e. output of `predict_proba`) or decision scores (i.e. output of
        `decision_function`) into a class label.

    pos_label : int, float, bool or str, default=None
        The label of the positive class. Used when `objective_metric` is
        `"max_tnr_at_tpr_constraint"`"`, `"max_tpr_at_tnr_constraint"`, or a dictionary.
        When `pos_label=None`, if `y_true` is in `{-1, 1}` or `{0, 1}`,
        `pos_label` is set to 1, otherwise an error will be raised. When using a
        scorer, `pos_label` can be passed as a keyword argument to
        :func:`~sklearn.metrics.make_scorer`.

    response_method : {"auto", "decision_function", "predict_proba"}, default="auto"
        Methods by the classifier `base_estimator` corresponding to the
        decision function for which we want to find a threshold. It can be:

        * if `"auto"`, it will try to invoke, for each classifier,
          `"predict_proba"` or `"decision_function"` in that order.
        * otherwise, one of `"predict_proba"` or `"decision_function"`.
          If the method is not implemented by the classifier, it will raise an
          error.

    Attributes
    ----------
    estimator_ : estimator instance
        The fitted classifier used when predicting.

    classes_ : ndarray of shape (n_classes,)
        The class labels.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

    See Also
    --------
    sklearn.model_selection.TunedThresholdClassifierCV : Classifier that post-tunes
        the decision threshold based on some metrics and using cross-validation.
    sklearn.calibration.CalibratedClassifierCV : Estimator that calibrates
        probabilities.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.metrics import confusion_matrix
    >>> from sklearn.model_selection import FixedThresholdClassifier, train_test_split
    >>> X, y = make_classification(
    ...     n_samples=1_000, weights=[0.9, 0.1], class_sep=0.8, random_state=42
    ... )
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, stratify=y, random_state=42
    ... )
    >>> classifier = LogisticRegression(random_state=0).fit(X_train, y_train)
    >>> print(confusion_matrix(y_test, classifier.predict(X_test)))
    [[217   7]
     [ 19   7]]
    >>> classifier_other_threshold = FixedThresholdClassifier(
    ...     classifier, threshold_value=0.1, response_method="predict_proba"
    ... ).fit(X_train, y_train)
    >>> print(confusion_matrix(y_test, classifier_other_threshold.predict(X_test)))
    [[184  40]
     [  6  20]]
    """

    _parameter_constraints: dict = {
        **BaseThresholdClassifier._parameter_constraints,
        "threshold_value": [Real],
    }

    def __init__(
        self,
        estimator,
        *,
        threshold_value=0.5,
        pos_label=None,
        response_method="auto",
    ):
        super().__init__(
            estimator=estimator, pos_label=pos_label, response_method=response_method
        )
        self.threshold_value = threshold_value

    def _fit(self, X, y, **params):
        """Fit the classifier.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        **params : dict
            Parameters to pass to the `fit` method of the underlying
            classifier and to the `objective_metric` scorer.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        self.estimator_ = clone(self.estimator).fit(X, y, **params)
        return self

    def _get_pos_label(self):
        """Get the positive label."""
        return self.pos_label

    def _get_decision_threshold(self):
        """Get the decision threshold."""
        return self.threshold_value

    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        router = MetadataRouter(owner=self.__class__.__name__).add(
            estimator=self.estimator,
            method_mapping=MethodMapping().add(callee="fit", caller="fit"),
        )
        return router


def _threshold_scores_to_class_labels(y_score, threshold, classes, pos_label):
    """Threshold `y_score` and return the associated class labels."""
    if pos_label is None:
        map_thresholded_score_to_label = np.array([0, 1])
    else:
        pos_label_idx = np.flatnonzero(classes == pos_label)[0]
        neg_label_idx = np.flatnonzero(classes != pos_label)[0]
        map_thresholded_score_to_label = np.array([neg_label_idx, pos_label_idx])

    return classes[map_thresholded_score_to_label[(y_score >= threshold).astype(int)]]


class _CurveScorer(_BaseScorer):
    """Scorer taking a continuous response and output a score for each threshold.

    Parameters
    ----------
    score_func : callable
        The score function to use. It will be called as
        `score_func(y_true, y_pred, **kwargs)`.

    sign : int
        Either 1 or -1 to returns the score with `sign * score_func(estimator, X, y)`.
        Thus, `sign` defined if higher scores are better or worse.

    n_thresholds : int or array-like
        Related to the number of decision thresholds for which we want to compute the
        score. If an integer, it will be used to generate `n_thresholds` thresholds
        uniformly distributed between the minimum and maximum predicted scores. If an
        array-like, it will be used as the thresholds.

    kwargs : dict
        Additional parameters to pass to the score function.

    response_method : str
        The method to call on the estimator to get the response values.
    """

    def __init__(self, score_func, sign, kwargs, n_thresholds, response_method):
        super().__init__(
            score_func=score_func,
            sign=sign,
            kwargs=kwargs,
            response_method=response_method,
        )
        self._n_thresholds = n_thresholds

    @classmethod
    def from_scorer(cls, scorer, response_method, n_thresholds, pos_label):
        """Create a continuous scorer from a normal scorer."""
        # add `pos_label` if requested by the scorer function
        scorer_kwargs = {**scorer._kwargs}
        signature_scoring_func = signature(scorer._score_func)
        if (
            "pos_label" in signature_scoring_func.parameters
            and "pos_label" not in scorer_kwargs
        ):
            if pos_label is None:
                # Since the provided `pos_label` is the default, we need to
                # use the default value of the scoring function that can be either
                # `None` or `1`.
                scorer_kwargs["pos_label"] = signature_scoring_func.parameters[
                    "pos_label"
                ].default
            else:
                scorer_kwargs["pos_label"] = pos_label
        # transform a binary metric into a curve metric for all possible decision
        # thresholds
        instance = cls(
            score_func=scorer._score_func,
            sign=scorer._sign,
            response_method=response_method,
            n_thresholds=n_thresholds,
            kwargs=scorer_kwargs,
        )
        # transfer the metadata request
        instance._metadata_request = scorer._get_metadata_request()
        return instance

    def _score(self, method_caller, estimator, X, y_true, **kwargs):
        """Evaluate predicted target values for X relative to y_true.

        Parameters
        ----------
        method_caller : callable
            Returns predictions given an estimator, method name, and other
            arguments, potentially caching results.

        estimator : object
            Trained estimator to use for scoring.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Test data that will be fed to estimator.predict.

        y_true : array-like of shape (n_samples,)
            Gold standard target values for X.

        **kwargs : dict
            Other parameters passed to the scorer. Refer to
            :func:`set_score_request` for more details.

        Returns
        -------
        scores : ndarray of shape (n_thresholds,)
            The scores associated to each threshold.

        potential_thresholds : ndarray of shape (n_thresholds,)
            The potential thresholds used to compute the scores.
        """
        pos_label = self._get_pos_label()
        y_score = method_caller(
            estimator, self._response_method, X, pos_label=pos_label
        )

        scoring_kwargs = {**self._kwargs, **kwargs}
        if isinstance(self._n_thresholds, Integral):
            potential_thresholds = np.linspace(
                np.min(y_score), np.max(y_score), self._n_thresholds
            )
        else:
            potential_thresholds = np.asarray(self._n_thresholds)
        score_thresholds = [
            self._sign
            * self._score_func(
                y_true,
                _threshold_scores_to_class_labels(
                    y_score, th, estimator.classes_, pos_label
                ),
                **scoring_kwargs,
            )
            for th in potential_thresholds
        ]
        return np.array(score_thresholds), potential_thresholds


def _fit_and_score_over_thresholds(
    classifier,
    X,
    y,
    *,
    fit_params,
    train_idx,
    val_idx,
    curve_scorer,
    score_params,
):
    """Fit a classifier and compute the scores for different decision thresholds.

    Parameters
    ----------
    classifier : estimator instance
        The classifier to fit and use for scoring. If `classifier` is already fitted,
        it will be used as is.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The entire dataset.

    y : array-like of shape (n_samples,)
        The entire target vector.

    fit_params : dict
        Parameters to pass to the `fit` method of the underlying classifier.

    train_idx : ndarray of shape (n_train_samples,) or None
        The indices of the training set. If `None`, `classifier` is expected to be
        already fitted.

    val_idx : ndarray of shape (n_val_samples,)
        The indices of the validation set used to score `classifier`. If `train_idx`,
        the entire set will be used.

    curve_scorer : scorer instance
        The scorer taking `classifier` and the validation set as input and outputting
        decision thresholds and scores as a curve. Note that this is different from
        the usual scorer that output a single score value:

        * when `score_method` is one of the four constraint metrics, the curve scorer
          will output a curve of two scores parametrized by the decision threshold, e.g.
          TPR/TNR or precision/recall curves for each threshold;
        * otherwise, the curve scorer will output a single score value for each
          threshold.

    score_params : dict
        Parameters to pass to the `score` method of the underlying scorer.

    Returns
    -------
    potential_thresholds : ndarray of shape (n_thresholds,)
        The decision thresholds used to compute the scores. They are returned in
        ascending order.

    scores : ndarray of shape (n_thresholds,) or tuple of such arrays
        The scores computed for each decision threshold. When TPR/TNR or precision/
        recall are computed, `scores` is a tuple of two arrays.
    """

    if train_idx is not None:
        X_train, X_val = _safe_indexing(X, train_idx), _safe_indexing(X, val_idx)
        y_train, y_val = _safe_indexing(y, train_idx), _safe_indexing(y, val_idx)
        fit_params_train = _check_method_params(X, fit_params, indices=train_idx)
        score_params_val = _check_method_params(X, score_params, indices=val_idx)
        classifier.fit(X_train, y_train, **fit_params_train)
    else:  # prefit estimator, only a validation set is provided
        X_val, y_val, score_params_val = X, y, score_params

    if curve_scorer is roc_curve or (
        isinstance(curve_scorer, _BaseScorer) and curve_scorer._score_func is roc_curve
    ):
        fpr, tpr, potential_thresholds = curve_scorer(
            classifier, X_val, y_val, **score_params_val
        )
        # For fpr=0/tpr=0, the threshold is set to `np.inf`. We need to remove it.
        fpr, tpr, potential_thresholds = fpr[1:], tpr[1:], potential_thresholds[1:]
        # thresholds are in decreasing order
        return potential_thresholds[::-1], ((1 - fpr)[::-1], tpr[::-1])
    elif curve_scorer is precision_recall_curve or (
        isinstance(curve_scorer, _BaseScorer)
        and curve_scorer._score_func is precision_recall_curve
    ):
        precision, recall, potential_thresholds = curve_scorer(
            classifier, X_val, y_val, **score_params_val
        )
        # thresholds are in increasing order
        # the last element of the precision and recall is not associated with any
        # threshold and should be discarded
        return potential_thresholds, (precision[:-1], recall[:-1])
    else:
        scores, potential_thresholds = curve_scorer(
            classifier, X_val, y_val, **score_params_val
        )
    return potential_thresholds, scores


def _mean_interpolated_score(target_thresholds, cv_thresholds, cv_scores):
    """Compute the mean interpolated score across folds by defining common thresholds.

    Parameters
    ----------
    target_thresholds : ndarray of shape (n_thresholds,)
        The thresholds to use to compute the mean score.

    cv_thresholds : ndarray of shape (n_folds, n_thresholds_fold)
        The thresholds used to compute the scores for each fold.

    cv_scores : ndarray of shape (n_folds, n_thresholds_fold)
        The scores computed for each threshold for each fold.

    Returns
    -------
    mean_score : ndarray of shape (n_thresholds,)
        The mean score across all folds for each target threshold.
    """
    return np.mean(
        [
            np.interp(target_thresholds, split_thresholds, split_score)
            for split_thresholds, split_score in zip(cv_thresholds, cv_scores)
        ],
        axis=0,
    )


class TunedThresholdClassifierCV(BaseThresholdClassifier):
    """Classifier that post-tunes the decision threshold using cross-validation.

    This estimator post-tunes the decision threshold (cut-off point) that is
    used for converting posterior probability estimates (i.e. output of
    `predict_proba`) or decision scores (i.e. output of `decision_function`)
    into a class label. The tuning is done by optimizing a binary metric,
    potentially constrained by a another metric.

    Read more in the :ref:`User Guide <TunedThresholdClassifierCV>`.

    .. versionadded:: 1.5

    Parameters
    ----------
    estimator : estimator instance
        The classifier, fitted or not, for which we want to optimize
        the decision threshold used during `predict`.

    objective_metric : {"max_tpr_at_tnr_constraint", "max_tnr_at_tpr_constraint", \
            "max_precision_at_recall_constraint, "max_recall_at_precision_constraint"} \
            , str, dict or callable, default="balanced_accuracy"
        The objective metric to be optimized. Can be one of:

        * a string associated to a scoring function (see model evaluation
          documentation);
        * a scorer callable object created with :func:`~sklearn.metrics.make_scorer`;
        * `"max_tnr_at_tpr_constraint"`: find the decision threshold for a true
          positive ratio (TPR) of `constraint_value`;
        * `"max_tpr_at_tnr_constraint"`: find the decision threshold for a true
          negative ratio (TNR) of `constraint_value`.
        * `"max_precision_at_recall_constraint"`: find the decision threshold for a
          recall of `constraint_value`;
        * `"max_recall_at_precision_constraint"`: find the decision threshold for a
          precision of `constraint_value`.

    constraint_value : float, default=None
        The value associated with the `objective_metric` metric for which we
        want to find the decision threshold when `objective_metric` is either
        `"max_tnr_at_tpr_constraint"`, `"max_tpr_at_tnr_constraint"`,
        `"max_precision_at_recall_constraint"`, or
        `"max_recall_at_precision_constraint"`.

    pos_label : int, float, bool or str, default=None
        The label of the positive class. Used when `objective_metric` is
        `"max_tnr_at_tpr_constraint"`"`, `"max_tpr_at_tnr_constraint"`, or a dictionary.
        When `pos_label=None`, if `y_true` is in `{-1, 1}` or `{0, 1}`,
        `pos_label` is set to 1, otherwise an error will be raised. When using a
        scorer, `pos_label` can be passed as a keyword argument to
        :func:`~sklearn.metrics.make_scorer`.

    response_method : {"auto", "decision_function", "predict_proba"}, default="auto"
        Methods by the classifier `base_estimator` corresponding to the
        decision function for which we want to find a threshold. It can be:

        * if `"auto"`, it will try to invoke, for each classifier,
          `"predict_proba"` or `"decision_function"` in that order.
        * otherwise, one of `"predict_proba"` or `"decision_function"`.
          If the method is not implemented by the classifier, it will raise an
          error.

    n_thresholds : int or array-like, default=100
        The number of decision threshold to use when discretizing the output of the
        classifier `method`. Pass an array-like to manually specify the thresholds
        to use.

    cv : int, float, cross-validation generator, iterable or "prefit", default=None
        Determines the cross-validation splitting strategy to train classifier.
        Possible inputs for cv are:

        * `None`, to use the default 5-fold stratified K-fold cross validation;
        * An integer number, to specify the number of folds in a stratified k-fold;
        * A float number, to specify a single shuffle split. The floating number should
          be in (0, 1) and represent the size of the validation set;
        * An object to be used as a cross-validation generator;
        * An iterable yielding train, test splits;
        * `"prefit"`, to bypass the cross-validation.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. warning::
            Using `cv="prefit"` and passing the same dataset for fitting `estimator`
            and tuning the cut-off point is subject to undesired overfitting. You can
            refer to :ref:`TunedThresholdClassifierCV_no_cv` for an example.

            This option should only be used when the set used to fit `estimator` is
            different from the one used to tune the cut-off point (by calling
            :meth:`TunedThresholdClassifierCV.fit`).

    refit : bool, default=True
        Whether or not to refit the classifier on the entire training set once
        the decision threshold has been found.
        Note that forcing `refit=False` on cross-validation having more
        than a single split will raise an error. Similarly, `refit=True` in
        conjunction with `cv="prefit"` will raise an error.

    n_jobs : int, default=None
        The number of jobs to run in parallel. When `cv` represents a
        cross-validation strategy, the fitting and scoring on each data split
        is done in parallel. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors. See :term:`Glossary <n_jobs>` for more details.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of cross-validation when `cv` is a float.
        See :term:`Glossary <random_state>`.

    store_cv_results : bool, default=False
        Whether to store all scores and thresholds computed during the cross-validation
        process.

    Attributes
    ----------
    estimator_ : estimator instance
        The fitted classifier used when predicting.

    best_threshold_ : float
        The new decision threshold.

    best_score_ : float or None
        The optimal score of the objective metric, evaluated at `best_threshold_`.

    constrained_score_ : float or None
        When `objective_metric` is one of `"max_tpr_at_tnr_constraint"`,
        `"max_tnr_at_tpr_constraint"`, `"max_precision_at_recall_constraint"`,
        `"max_recall_at_precision_constraint"`, it will corresponds to the score of the
        metric which is constrained. It should be close to `constraint_value`. If
        `objective_metric` is not one of the above, `constrained_score_` is None.

    cv_results_ : dict or None
        A dictionary containing the scores and thresholds computed during the
        cross-validation process. Only exist if `store_cv_results=True`.
        The keys are different depending on the `objective_metric` used:

        * when `objective_metric` is one of `"max_tpr_at_tnr_constraint"`,
          `"max_tnr_at_tpr_constraint"`, `"max_precision_at_recall_constraint"`,
          `"max_recall_at_precision_constraint"`, the keys are `"thresholds"`,
          `"constrained_scores"`, and `"maximized_scores"`;
        * otherwise, for score computing a single values, the keys are `"thresholds"`
          and `"scores"`.

    classes_ : ndarray of shape (n_classes,)
        The class labels.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

    See Also
    --------
    sklearn.model_selection.FixedThresholdClassifier : Classifier that uses a
        constant threshold.
    sklearn.calibration.CalibratedClassifierCV : Estimator that calibrates
        probabilities.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.metrics import classification_report
    >>> from sklearn.model_selection import TunedThresholdClassifierCV, train_test_split
    >>> X, y = make_classification(
    ...     n_samples=1_000, weights=[0.9, 0.1], class_sep=0.8, random_state=42
    ... )
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, stratify=y, random_state=42
    ... )
    >>> classifier = RandomForestClassifier(random_state=0).fit(X_train, y_train)
    >>> print(classification_report(y_test, classifier.predict(X_test)))
                  precision    recall  f1-score   support
    <BLANKLINE>
               0       0.94      0.99      0.96       224
               1       0.80      0.46      0.59        26
    <BLANKLINE>
        accuracy                           0.93       250
       macro avg       0.87      0.72      0.77       250
    weighted avg       0.93      0.93      0.92       250
    <BLANKLINE>
    >>> classifier_tuned = TunedThresholdClassifierCV(
    ...     classifier, objective_metric="max_precision_at_recall_constraint",
    ...     constraint_value=0.7,
    ... ).fit(X_train, y_train)
    >>> print(
    ...     f"Cut-off point found at {classifier_tuned.best_threshold_:.3f} for a "
    ...     f"recall of {classifier_tuned.constrained_score_:.3f} and a precision of "
    ...     f"{classifier_tuned.best_score_:.3f}."
    ... )
    Cut-off point found at 0.3... for a recall of 0.7... and a precision of 0.7...
    >>> print(classification_report(y_test, classifier_tuned.predict(X_test)))
                  precision    recall  f1-score   support
    <BLANKLINE>
               0       0.96      0.96      0.96       224
               1       0.68      0.65      0.67        26
    <BLANKLINE>
        accuracy                           0.93       250
       macro avg       0.82      0.81      0.81       250
    weighted avg       0.93      0.93      0.93       250
    <BLANKLINE>
    """

    _parameter_constraints: dict = {
        **BaseThresholdClassifier._parameter_constraints,
        "objective_metric": [
            StrOptions(
                set(get_scorer_names())
                | {
                    "max_tnr_at_tpr_constraint",
                    "max_tpr_at_tnr_constraint",
                    "max_precision_at_recall_constraint",
                    "max_recall_at_precision_constraint",
                }
            ),
            callable,
            MutableMapping,
        ],
        "constraint_value": [Real, None],
        "n_thresholds": [Interval(Integral, 1, None, closed="left"), "array-like"],
        "cv": [
            "cv_object",
            StrOptions({"prefit"}),
            Interval(RealNotInt, 0.0, 1.0, closed="neither"),
        ],
        "refit": ["boolean"],
        "n_jobs": [Integral, None],
        "random_state": ["random_state"],
        "store_cv_results": ["boolean"],
    }

    def __init__(
        self,
        estimator,
        *,
        objective_metric="balanced_accuracy",
        constraint_value=None,
        pos_label=None,
        response_method="auto",
        n_thresholds=100,
        cv=None,
        refit=True,
        n_jobs=None,
        random_state=None,
        store_cv_results=False,
    ):
        super().__init__(
            estimator=estimator, response_method=response_method, pos_label=pos_label
        )
        self.objective_metric = objective_metric
        self.constraint_value = constraint_value
        self.n_thresholds = n_thresholds
        self.cv = cv
        self.refit = refit
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.store_cv_results = store_cv_results

    def _fit(self, X, y, **params):
        """Fit the classifier and post-tune the decision threshold.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        **params : dict
            Parameters to pass to the `fit` method of the underlying
            classifier and to the `objective_metric` scorer.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        if isinstance(self.cv, Real) and 0 < self.cv < 1:
            cv = StratifiedShuffleSplit(
                n_splits=1, test_size=self.cv, random_state=self.random_state
            )
        elif self.cv == "prefit":
            if self.refit is True:
                raise ValueError("When cv='prefit', refit cannot be True.")
            try:
                check_is_fitted(self.estimator, "classes_")
            except NotFittedError as exc:
                raise NotFittedError(
                    """When cv='prefit', `estimator` must be fitted."""
                ) from exc
            cv = self.cv
        else:
            cv = check_cv(self.cv, y=y, classifier=True)
            if self.refit is False and cv.get_n_splits() > 1:
                raise ValueError("When cv has several folds, refit cannot be False.")

        if isinstance(self.objective_metric, str) and self.objective_metric in {
            "max_tpr_at_tnr_constraint",
            "max_tnr_at_tpr_constraint",
            "max_precision_at_recall_constraint",
            "max_recall_at_precision_constraint",
        }:
            if self.constraint_value is None:
                raise ValueError(
                    "When `objective_metric` is 'max_tpr_at_tnr_constraint', "
                    "'max_tnr_at_tpr_constraint', 'max_precision_at_recall_constraint',"
                    " or 'max_recall_at_precision_constraint', `constraint_value` must "
                    "be provided. Got None instead."
                )
            constrained_metric = True
        else:
            constrained_metric = False

        routed_params = process_routing(self, "fit", **params)
        self._curve_scorer = self._get_curve_scorer()

        # in the following block, we:
        # - define the final classifier `self.estimator_` and train it if necessary
        # - define `classifier` to be used to post-tune the decision threshold
        # - define `split` to be used to fit/score `classifier`
        if cv == "prefit":
            self.estimator_ = self.estimator
            classifier = self.estimator_
            splits = [(None, range(_num_samples(X)))]
        else:
            self.estimator_ = clone(self.estimator)
            classifier = clone(self.estimator)
            splits = cv.split(X, y, **routed_params.splitter.split)

            if self.refit:
                # train on the whole dataset
                X_train, y_train, fit_params_train = X, y, routed_params.estimator.fit
            else:
                # single split cross-validation
                train_idx, _ = next(cv.split(X, y, **routed_params.splitter.split))
                X_train = _safe_indexing(X, train_idx)
                y_train = _safe_indexing(y, train_idx)
                fit_params_train = _check_method_params(
                    X, routed_params.estimator.fit, indices=train_idx
                )

            self.estimator_.fit(X_train, y_train, **fit_params_train)

        if hasattr(self.estimator_, "n_features_in_"):
            self.n_features_in_ = self.estimator_.n_features_in_
        if hasattr(self.estimator_, "feature_names_in_"):
            self.feature_names_in_ = self.estimator_.feature_names_in_

        cv_thresholds, cv_scores = zip(
            *Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_and_score_over_thresholds)(
                    clone(classifier) if cv != "prefit" else classifier,
                    X,
                    y,
                    fit_params=routed_params.estimator.fit,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    curve_scorer=self._curve_scorer,
                    score_params=routed_params.scorer.score,
                )
                for train_idx, val_idx in splits
            )
        )

        if any(np.isclose(th[0], th[-1]) for th in cv_thresholds):
            raise ValueError(
                "The provided estimator makes constant predictions. Therefore, it is "
                "impossible to optimize the decision threshold."
            )

        # find the global min and max thresholds across all folds
        min_threshold = min(
            split_thresholds.min() for split_thresholds in cv_thresholds
        )
        max_threshold = max(
            split_thresholds.max() for split_thresholds in cv_thresholds
        )
        if isinstance(self.n_thresholds, Integral):
            decision_thresholds = np.linspace(
                min_threshold, max_threshold, num=self.n_thresholds
            )
        else:
            decision_thresholds = np.asarray(self.n_thresholds)

        if not constrained_metric:  # find best score that is the highest value
            objective_scores = _mean_interpolated_score(
                decision_thresholds, cv_thresholds, cv_scores
            )
            best_idx = objective_scores.argmax()
            self.best_score_ = objective_scores[best_idx]
            self.best_threshold_ = decision_thresholds[best_idx]
            self.constrained_score_ = None
            if self.store_cv_results:
                self.cv_results_ = {
                    "thresholds": decision_thresholds,
                    "scores": objective_scores,
                }
        else:
            if "tpr" in self.objective_metric:  # tpr/tnr
                mean_tnr, mean_tpr = [
                    _mean_interpolated_score(decision_thresholds, cv_thresholds, sc)
                    for sc in zip(*cv_scores)
                ]
            else:  # precision/recall
                mean_precision, mean_recall = [
                    _mean_interpolated_score(decision_thresholds, cv_thresholds, sc)
                    for sc in zip(*cv_scores)
                ]

            def _get_best_idx(constrained_score, maximized_score):
                """Find the index of the best score constrained by another score."""
                mask = constrained_score >= self.constraint_value
                mask_idx = maximized_score[mask].argmax()
                return np.flatnonzero(mask)[mask_idx]

            if self.objective_metric == "max_tpr_at_tnr_constraint":
                constrained_scores, maximized_scores = mean_tnr, mean_tpr
            elif self.objective_metric == "max_tnr_at_tpr_constraint":
                constrained_scores, maximized_scores = mean_tpr, mean_tnr
            elif self.objective_metric == "max_precision_at_recall_constraint":
                constrained_scores, maximized_scores = mean_recall, mean_precision
            else:  # max_recall_at_precision_constraint
                constrained_scores, maximized_scores = mean_precision, mean_recall

            best_idx = _get_best_idx(constrained_scores, maximized_scores)
            self.best_score_ = maximized_scores[best_idx]
            self.constrained_score_ = constrained_scores[best_idx]
            self.best_threshold_ = decision_thresholds[best_idx]
            if self.store_cv_results:
                self.cv_results_ = {
                    "thresholds": decision_thresholds,
                    "constrained_scores": constrained_scores,
                    "maximized_scores": maximized_scores,
                }

        return self

    def _get_pos_label(self):
        """Get the positive label."""
        # `pos_label` has been validated and is stored in the scorer
        return self._curve_scorer._get_pos_label()

    def _get_decision_threshold(self):
        """Get the decision threshold."""
        return self.best_threshold_

    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        router = (
            MetadataRouter(owner=self.__class__.__name__)
            .add(
                estimator=self.estimator,
                method_mapping=MethodMapping().add(callee="fit", caller="fit"),
            )
            .add(
                splitter=self.cv,
                method_mapping=MethodMapping().add(callee="split", caller="fit"),
            )
            .add(
                scorer=self._get_curve_scorer(),
                method_mapping=MethodMapping().add(callee="score", caller="fit"),
            )
        )
        return router

    def _get_curve_scorer(self):
        """Get the curve scorer based on the objective metric used.

        Here, we reuse the conventional "scorer API" via `make_scorer` or
        `_CurveScorer`. Note that the use here is unconventional because `make_scorer`
        or the "scorer API" is expected to return a single score value when calling
        `scorer(estimator, X, y)`. Here the score function used are both returning
        scores and thresholds representing a curve.
        """
        if self.objective_metric in {
            "max_tnr_at_tpr_constraint",
            "max_tpr_at_tnr_constraint",
            "max_precision_at_recall_constraint",
            "max_recall_at_precision_constraint",
        }:
            if "tpr" in self.objective_metric:  # tpr/tnr
                score_curve_func = roc_curve
            else:  # precision/recall
                score_curve_func = precision_recall_curve
            curve_scorer = make_scorer(
                score_curve_func,
                response_method=self._response_method,
                pos_label=self.pos_label,
            )
        else:
            scoring = check_scoring(self.estimator, scoring=self.objective_metric)
            curve_scorer = _CurveScorer.from_scorer(
                scoring, self._response_method, self.n_thresholds, self.pos_label
            )
        return curve_scorer
