# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Authors: Raghav RV <rvraghav93@gmail.com>
#          Guillaume Lemaitre <guillaume.lemaitre@inria.fr>
#
# License: BSD 3 clause

import numpy as np
cimport numpy as np

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer


cdef struct SplitRecord:
    # Data to track sample split
    SIZE_t feature         # Which feature to split on.
    SIZE_t pos             # Split samples array at the given position,
                           # i.e. count of samples below threshold for feature.
                           # pos is >= end if the node is a leaf.
    double threshold       # Threshold to split at.
    double improvement     # Impurity improvement given parent node.
    double proxy_improvement  # Proxy for impurity improvement to speed up
                              # computation times

    # stats for left partition
    SIZE_t n_left
    double weighted_n_left
    double sum_y_left      # sum of target values of left split
    double sum_ysq_left    # sum of square of target values of left
    double impurity_left   # Impurity of the left split.

    # stats for right partition
    SIZE_t n_right
    double weighted_n_right
    double sum_y_right     # sum of target values of right split
    double sum_ysq_right   # sum of square of target values of right
    double impurity_right  # Impurity of the right split.


cdef inline int init_SplitRecord_from_parent(SplitRecord* s,
                                             Node* parent_node,
                                             double* parent_node_stats) nogil except -1:
    s.feature = 0
    s.pos = 0  # Indicates an invalid split
    s.threshold = np.nan
    s.improvement = 0
    s.proxy_improvement = 0

    # All samples are on the right
    s.n_left = 0
    s.weighted_n_left = 0
    s.sum_y_left = 0
    s.sum_ysq_left = 0
    s.impurity_left = 0

    s.n_right = parent_node.n_node_samples
    s.weighted_n_right = parent_node.weighted_n_node_samples
    s.sum_y_right = parent_node_stats[1]
    s.sum_ysq_right = parent_node_stats[2]
    s.impurity_right = parent_node.impurity
    return 0

cdef inline int copy_SplitRecord(SplitRecord* src, SplitRecord* dest) nogil except -1:
    # XXX: Should we do a simpler memcopy?
    dest.feature = src.feature
    dest.pos = src.pos
    dest.threshold = src.threshold
    dest.improvement = src.improvement
    dest.proxy_improvement = src.proxy_improvement
    dest.n_left = src.n_left
    dest.n_right = src.n_right
    dest.sum_y_left = src.sum_y_left
    dest.sum_y_right = src.sum_y_right
    dest.sum_ysq_left = src.sum_ysq_left
    dest.sum_ysq_right = src.sum_ysq_right
    dest.impurity_left = src.impurity_left
    dest.impurity_right = src.impurity_right
    return 0