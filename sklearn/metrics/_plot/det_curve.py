import scipy as sp

from .base import _get_response

from .. import detection_error_tradeoff_curve

from ...utils import check_matplotlib_support


class DetCurveDisplay:
    """DET Curve visualization.

    It is recommend to use :func:`~sklearn.metrics.plot_det_curve` to create a
    visualizer. All parameters are stored as attributes.

    Read more in the :ref:`User Guide <visualizations>`.

    .. versionadded:: 0.24

    Parameters
    ----------
    fpr : ndarray
        False positive rate.

    tpr : ndarray
        True positive rate.

    estimator_name : str, default=None
        Name of estimator. If None, the estimator name is not shown.

    pos_label : str or int, default=None
        The label of the positive class.
        When `pos_label=None`, if `y_true` is in {-1, 1} or {0, 1},
        `pos_label` is set to 1, otherwise an error will be raised.

    Attributes
    ----------
    line_ : matplotlib Artist
        DET Curve.

    ax_ : matplotlib Axes
        Axes with DET Curve.

    figure_ : matplotlib Figure
        Figure containing the curve.

    Examples
    --------
    """
    def __init__(self, *, fpr, fnr, estimator_name=None, pos_label=None):
        self.fpr = fpr
        self.fnr = fnr
        self.estimator_name = estimator_name
        self.pos_label = pos_label

    def plot(self, ax=None, *, name=None, **kwargs):
        """Plot visualization.

        Parameters
        ----------
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        name : str, default=None
            Name of DET Curve for labeling. If `None`, use the name of the
            estimator.

        Returns
        -------
        display : :class:`~sklearn.metrics.plot.DetCurveDisplay`
            Object that stores computed values.
        """
        check_matplotlib_support('RocCurveDisplay.plot')

        name = self.estimator_name if name is None else name
        line_kwargs = {} if name is None else {"label": name}
        line_kwargs.update(**kwargs)

        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        self.line_, = ax.plot(
            sp.stats.norm.ppf(self.fpr),
            sp.stats.norm.ppf(self.fnr),
            **line_kwargs,
        )
        info_pos_label = (f" (Positive label: {self.pos_label})"
                          if self.pos_label is not None else "")

        xlabel = "False Positive Rate" + info_pos_label
        ylabel = "False Negative Rate" + info_pos_label
        ax.set(xlabel=xlabel, ylabel=ylabel)

        if "label" in line_kwargs:
            ax.legend(loc="lower right")

        ticks = [0.001, 0.01, 0.05, 0.20, 0.5, 0.80, 0.95, 0.99, 0.999]
        tick_locations = sp.stats.norm.ppf(ticks)
        tick_labels = [
            '{:.0%}'.format(s) if (100*s).is_integer() else '{:.1%}'.format(s)
            for s in ticks
        ]
        ax.set_xticks(tick_locations)
        ax.set_xticklabels(tick_labels)
        ax.set_xlim(-3, 3)
        ax.set_yticks(tick_locations)
        ax.set_yticklabels(tick_labels)
        ax.set_ylim(-3, 3)

        self.ax_ = ax
        self.figure_ = ax.figure
        return self


def plot_det_curve(
    estimator,
    X,
    y,
    *,
    sample_weight=None,
    response_method="auto",
    name=None,
    ax=None,
    pos_label=None,
    **kwargs
):
    """Plot detection error tradeoff (DET) curve.

    Extra keyword arguments will be passed to matplotlib's `plot`.

    Read more in the :ref:`User Guide <visualizations>`.

    .. versionadded:: 0.24

    Parameters
    ----------
    estimator : estimator instance
        Fitted classifier or a fitted :class:`~sklearn.pipeline.Pipeline`
        in which the last estimator is a classifier.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Input values.

    y : array-like of shape (n_samples,)
        Target values.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    response_method : {'predict_proba', 'decision_function', 'auto'} \
            default='auto'
        Specifies whether to use :term:`predict_proba` or
        :term:`decision_function` as the target response. If set to 'auto',
        :term:`predict_proba` is tried first and if it does not exist
        :term:`decision_function` is tried next.

    name : str, default=None
        Name of ROC Curve for labeling. If `None`, use the name of the
        estimator.

    ax : matplotlib axes, default=None
        Axes object to plot on. If `None`, a new figure and axes is created.

    pos_label : str or int, default=None
        The label of the positive class.
        When `pos_label=None`, if `y_true` is in {-1, 1} or {0, 1},
        `pos_label` is set to 1, otherwise an error will be raised.

    Returns
    -------
    display : :class:`~sklearn.metrics.DetCurveDisplay`
        Object that stores computed values.

    See Also
    --------
    roc_auc_score : Compute the area under the ROC curve

    roc_curve : Compute Receiver operating characteristic (ROC) curve

    Examples
    --------
    """
    check_matplotlib_support('plot_roc_curve')

    y_pred, pos_label = _get_response(
        X, estimator, response_method, pos_label=pos_label
    )

    fpr, fnr, _ = detection_error_tradeoff_curve(
        y,
        y_pred,
        pos_label=pos_label,
        sample_weight=sample_weight,
    )

    name = estimator.__class__.__name__ if name is None else name

    viz = DetCurveDisplay(
        fpr=fpr,
        fnr=fnr,
        estimator_name=name,
        pos_label=pos_label
    )

    return viz.plot(ax=ax, name=name, **kwargs)
