import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm, gridspec
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing.data import QuantileNormalizer

from sklearn.datasets import fetch_california_housing

dataset = fetch_california_housing()
X, y = dataset.data, dataset.target

# Take only 2 features to make visualization easier
# Feature 0 has a tapering distribution of outliers
# Feature 5 has a few but widely separated outliers
X = X[:, [0, 5]]

X_min_max_scaled = MinMaxScaler().fit_transform(X)
X_max_abs_scaled = MaxAbsScaler().fit_transform(X)
X_standard_scaled = StandardScaler().fit_transform(X)
X_robust_scaled = RobustScaler(quantile_range=(25, 75)).fit_transform(X)
X_l2_normalized = Normalizer().fit_transform(X)
X_quantile_normalized = QuantileNormalizer().fit_transform(X)

y = minmax_scale(y)  # To make colors corresponding to the target


def plot_distribution(axes, X, y, hist_nbins=50, plot_title="", size=(15, 10)):
    # |          :
    # |    DATA  : hist_X1
    # |  ........:.........
    # |  hist_X0 : empty
    # |          :
    # |____________________

    ax, hist_X1, hist_X0, empty = axes
    empty.axis('off')

    ax.set_title(plot_title, fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # The scatter plot
    colors = cm.plasma(np.array(y, dtype=float))
    ax.scatter(X[:, 0], X[:, 1],
               marker='.', s=20, lw=0, alpha=0.5, c=colors)

    # The histogram for axis 0 and axis 1
    for hist_ax, X_feat in ((hist_X1, X[:, 1]),
                            (hist_X0, X[:, 0])):
        if hist_ax is hist_X0:
            orientation = 'vertical'
            hist_ax.set_xlim(ax.get_xlim())
            hist_ax.invert_yaxis()
        else:
            orientation = 'horizontal'
            hist_ax.set_ylim(ax.get_ylim())

        hist_ax.hist(X_feat, bins=hist_nbins, orientation=orientation,
                     color='grey', ec='grey')
        hist_ax.axis('off')

# gs2 = gridspec.GridSpec(28, 6, width_ratios=[5, 1, 5, 1, 5, 1], wspace=0.3,
#                         height_ratios=[5, 1, 5, 1], hspace=0.3)
# subplots2 = list(plt.subplot(g) for g in gs2)

# Seven rows of 2 x 4 grids
gs = gridspec.GridSpec(14, 4, left=3, right=4,
                       width_ratios=[5, 1, 5, 1], wspace=0.3,
                       height_ratios=[5, 1] * 7, hspace=0.3)
subplots = list(plt.subplot(g) for g in gs)
subplots[0].figure.set_size_inches(8, 42)

for i, (X, title) in enumerate((
        (X, "Unscaled data"),
        (X_min_max_scaled, "Data after min-max scaling"),
        (X_robust_scaled, "Data after robust scaling"),
        (X_max_abs_scaled, "Data after max-abs scaling"),
        (X_standard_scaled, "Data after standard scaling"),
        (X_l2_normalized, "Data after sample-wise L2 normalizing"),
        (X_quantile_normalized, "Data after quantile normalizing"))):
    offset = 8 * i
    axes = subplots[offset: offset + 2] + subplots[offset + 4: offset + 6]
    plot_distribution(axes, X, y, hist_nbins=50,
                      plot_title=title + "\n(including outliers)")

    X0_min, X0_99th_pc = np.percentile(X[:, 0], [0, 99])
    X1_min, X1_99th_pc = np.percentile(X[:, 1], [0, 99])

    non_outliers = np.all(X < [X0_99th_pc, X1_99th_pc], axis=1)
    axes = subplots[offset + 2: offset + 4] + subplots[offset + 6: offset + 8]
    plot_distribution(axes, X[non_outliers], y[non_outliers], hist_nbins=50,
                      plot_title=(title +
                                  "\n(Zoomed-in at quantile range [0, 99))"))

plt.show()
