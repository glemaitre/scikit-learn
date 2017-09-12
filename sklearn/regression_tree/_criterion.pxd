# cython: cdivision=True

# from ._stats_node cimport StatsNode


# cdef double _impurity_mse(StatsNode stats_node)

# cpdef double impurity_mse(double sum_y, double sum_sq_y, int n_samples,
#                           double sum_weighted_samples)

# cpdef double impurity_improvement(StatsNode c_stats,
#                                   StatsNode r_stats,
#                                   StatsNode l_stats,
#                                   double sum_total_weighted_samples,
#                                   criterion)

from ._stats_node cimport StatsNode


cdef inline double _impurity_mse(StatsNode stats_node):
    cdef double impurity
    impurity = (stats_node.sum_sq_y /
                stats_node.sum_weighted_samples)
    impurity -= ((stats_node.sum_y /
                  stats_node.sum_weighted_samples) ** 2.0)

    return impurity


cpdef double impurity_mse(double sum_y, double sum_sq_y, int n_samples,
                          double sum_weighted_samples)


cpdef double impurity_improvement(StatsNode c_stats,
                                  StatsNode r_stats,
                                  StatsNode l_stats,
                                  double sum_total_weighted_samples,
                                  criterion)
