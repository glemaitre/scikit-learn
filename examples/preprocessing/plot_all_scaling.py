import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm, gridspec

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing.data import QuantileNormalizer

from sklearn.datasets import fetch_california_housing

dataset = fetch_california_housing()
X_full, y_full = dataset.data, dataset.target

# Take only 2 features to make visualization easier
# Feature 0 has a tapering distribution of outliers
# Feature 5 has a few but very large outliers
X = X_full[:, [0, 5]]

X_min_max_scaled = MinMaxScaler().fit_transform(X)
X_max_abs_scaled = MaxAbsScaler().fit_transform(X)
X_standard_scaled = StandardScaler().fit_transform(X)
X_robust_scaled = RobustScaler(quantile_range=(25, 75)).fit_transform(X)
X_l2_normalized = Normalizer().fit_transform(X)
X_quantile_normalized = QuantileNormalizer().fit_transform(X)

y = minmax_scale(y_full)  # To make colors corresponding to the target


def plot_distribution(axes, X, y, hist_nbins=50, plot_title="", size=(15, 10),
                      X_label="", y_label=""):
    ax, hist_X1, hist_X0, empty = axes
    empty.axis('off')

    ax.set_title(plot_title, fontsize=12)
    ax.set_xlabel(X_label)
    ax.set_ylabel(y_label)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # The scatter plot
    colors = cm.plasma_r(y)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.5, marker='o', s=5, lw=0, c=colors)

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

gs = gridspec.GridSpec(7 * 4, 2 * 2, right=4, left=3,
                       width_ratios=[5, 1, 5, 1], wspace=0.3,
                       height_ratios=[5, 1, 0.2, 0.5] * 7, hspace=0.5)
subplots = list(plt.subplot(g) for g in gs)
subplots[0].figure.set_size_inches(10, 50)

for i, (X, title) in enumerate((
        (X, "Unscaled data"),
        (X_min_max_scaled, "Data after min-max scaling"),
        (X_robust_scaled, "Data after robust scaling"),
        (X_max_abs_scaled, "Data after max-abs scaling"),
        (X_standard_scaled, "Data after standard scaling"),
        (X_l2_normalized, "Data after sample-wise L2 normalizing"),
        (X_quantile_normalized, "Data after quantile normalizing"))):
    offset = 16 * i
    axes = subplots[offset: offset + 2] + subplots[offset + 4: offset + 6]
    plot_distribution(axes, X, y, hist_nbins=50,
                      plot_title=title + "\n(including outliers)",
                      X_label="Median Income", y_label="Number of households")

    X0_min, X0_99th_pc = np.percentile(X[:, 0], [0, 99])
    X1_min, X1_99th_pc = np.percentile(X[:, 1], [0, 99])

    non_outliers = np.all(X < [X0_99th_pc, X1_99th_pc], axis=1)
    axes = subplots[offset + 2: offset + 4] + subplots[offset + 6: offset + 8]
    plot_distribution(axes, X[non_outliers], y[non_outliers], hist_nbins=50,
                      plot_title=(title +
                                  "\n(Zoomed-in at quantile range [0, 99))"),
                      X_label="Median Income", y_label="Number of households")

    # Plot a heatmap legend for the y, combining a row of 4 cols
    heatmap_legend_ax = plt.subplot(gs[offset+8: offset+12])
    norm = mpl.colors.Normalize(y_full.min(), y_full.max())
    mpl.colorbar.ColorbarBase(heatmap_legend_ax, cmap=cm.plasma_r,
                              norm=norm, orientation='horizontal',
                              label='color mapping for values of y')

    # Blank space to avoid overlapping of plots;
    # plt.tight_layout does not work with gridspec, height of this row is
    # adjusted at `height_ratios` param given to `GridSpec`
    plt.subplot(gs[offset+12: offset+16]).axis('off')

plt.show()
