# Authors: Denis Engemann <denis-alexander.engemann@inria.fr>
#          Guillaume Lemaitre <guillaume.lemaitre@inria.fr>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Raghav RV <rvraghav93@gmail.com>
#          Thierry Guillemot <thierry.guillemot.work@gmail.com>
# License: BSD 3 clause

import numpy as np
from scipy import sparse

from sklearn.utils.testing import assert_equal
from sklearn.preprocessing.utils import (_hist_bin_auto,
                                         _hist_bin_doane,
                                         _hist_bin_fd,
                                         _hist_bin_rice,
                                         _hist_bin_scott,
                                         _hist_bin_sqrt,
                                         _hist_bin_sturges)

RND_SEED = np.random.RandomState(42)
X = RND_SEED.randn(100)
X_sparse = sparse.csc_matrix(X)


def test_hist_bin_methods():
    assert_equal(_hist_bin_auto(X), _hist_bin_auto(X_sparse))
    assert_equal(_hist_bin_doane(X), _hist_bin_doane(X_sparse))
    assert_equal(_hist_bin_fd(X), _hist_bin_fd(X_sparse))
    assert_equal(_hist_bin_rice(X), _hist_bin_rice(X_sparse))
    assert_equal(_hist_bin_scott(X), _hist_bin_scott(X_sparse))
    assert_equal(_hist_bin_sqrt(X), _hist_bin_sqrt(X_sparse))
    assert_equal(_hist_bin_sturges(X), _hist_bin_sturges(X_sparse))
