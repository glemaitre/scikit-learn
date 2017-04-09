# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

cimport numpy as np

ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters

from .stats_node cimport StatsNode
from .split_record cimport SplitRecord

cdef SIZE_t MSE_CRITERION = 0

cpdef _impurity_mse_py(stats_node):
    return _impurity_mse(stats_node)

cdef inline DOUBLE_t _impurity_mse(NodeStats* stats) nogil except -1:
    """Compute the impurity in MSE sense

    Parameters
    ----------
    stats_node: StatsNode,
        The split record from which to compute the impurity

    Returns
    -------
    impurity: float,
        The impurity.
    """
    cdef DOUBLE_t impurity
    impurity = (stats[0].sum_sq_y /
                stats[0].sum_weighted_samples)
    impurity -= ((stats[0].sum_y /
                  stats[0].sum_weighted_samples) ** 2.0)
    return impurity


cdef inline DOUBLE_t impurity_improvement(
        SplitRecord c_split_record,
        DOUBLE_t sum_total_weighted_samples,
        SIZE_t criterion=MSE_CRITERION) nogil except -1:
    """Compute the impurity improvement.

    Parameters
    ----------
    c_split_record: SplitRecord,
        Split record of the current node.

    sum_total_weighted_samples: float,
        The sum of all the weights for all samples.

    criterion: str, optional(default='mse')
        The criterion to use to compute the impurity improvement.

    Returns
    -------
    impurity_improvement: float,
        Impurity improvement
    """
    # check that there is more sample in the root nodes than
    # in the current nodes
    if ((sum_total_weighted_samples <
         c_split_record.c_stats.sum_weighted_samples) or
        (sum_total_weighted_samples <
         c_split_record.l_stats.sum_weighted_samples) or
        (sum_total_weighted_samples <
         c_split_record.r_stats.sum_weighted_samples)):
        return -1

    # impurity current node, left child and right child
    cdef DOUBLE_t c_impurity = _impurity_mse(c_split_record.c_stats)
    cdef DOUBLE_t l_impurity = _impurity_mse(c_split_record.l_stats)
    cdef DOUBLE_t r_impurity = _impurity_mse(c_split_record.r_stats)

    return ((c_split_record.c_stats.sum_weighted_samples /
             sum_total_weighted_samples) *
            (c_impurity -
             (c_split_record.l_stats.sum_weighted_samples /
              sum_total_weighted_samples * l_impurity) -
             (c_split_record.r_stats.sum_weighted_samples /
              sum_total_weighted_samples * r_impurity)))
