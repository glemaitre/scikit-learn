import numpy as np

from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt
from matplotlib import cm, gridspec

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing.data import QuantileNormalizer

from sklearn.preprocessing import LabelEncoder

from sklearn.datasets import fetch_california_housing

dataset = fetch_california_housing()
X, y = dataset.data, dataset.target #  .astype(int)
# Binarize
y = np.where(y < np.median(y), 0, 1)

X_min_max_scaled = MinMaxScaler().fit_transform(X)
X_max_abs_scaled = MaxAbsScaler().fit_transform(X)
X_standard_scaled = StandardScaler().fit_transform(X)
X_robust_scaled = RobustScaler(quantile_range=(25, 75)).fit_transform(X)
X_l2_normalized = Normalizer().fit_transform(X)
X_quantile_normalized = QuantileNormalizer().fit_transform(X)

# Take only 2 features to make visualization easier
# Feature 0 has a tapering distribution of outliers
# Feature 5 has a few but widely separated outliers
X = X[:, [0, 5]]

def plot_distribution(X, y, hist_nbins=50, plot_title=""):
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(plot_title, fontsize=12)

    gs = gridspec.GridSpec(2, 2, width_ratios=[5, 1], wspace=0.02,
                           height_ratios=[5, 1], hspace=0.1)
    ax, hist_X1, hist_X0, empty = (plt.subplot(g) for g in gs)
    empty.axis('off')

    # The scatter plot
    colors = cm.bwr(np.array(y, dtype=float))
    ax.scatter(X[:, 0], X[:, 1],
               marker='.', s=10, lw=0, alpha=0.7, c=colors)

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
                     color='grey')
        hist_ax.axis('off')

plot_distribution(X, y, hist_nbins=50,
                  plot_title="The original distribution of data with outliers")

X0_min, X0_99th_pc = np.percentile(X[:, 0], [0, 99])
X1_min, X1_99th_pc = np.percentile(X[:, 1], [0, 99])

non_outliers = np.all(X < [X0_99th_pc, X1_99th_pc], axis=1)
plot_distribution(X[non_outliers], y[non_outliers], hist_nbins=50,
                  plot_title="Data with extreme outliers (>99th percentile) removed")

for X, technique in ((X_min_max_scaled, "min-max scaling"),
                     (X_robust_scaled, "robust scaling"),
                     (X_max_abs_scaled, "max abs scaling"),
                     (X_standard_scaled, "standard scaling"),
                     (X_l2_normalized, "L2 Normalizing"),
                     (X_quantile_normalized, "Quantile Normalizing")):
    plot_distribution(X, y, hist_nbins=50, plot_title="Distribution after %s" %
                      technique)

plt.show()
