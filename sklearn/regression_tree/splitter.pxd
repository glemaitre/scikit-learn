# distutils: language = c++

from libcpp.

from libc.math cimport fabs
cimport numpy as np
from .criterion import impurity_improvement
from .stats_node cimport StatsNode
from .split_record cimport SplitRecord

ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters

cdef DOUBLE_t NAN = <DOUBLE_t>np.nan
cdef DOUBLE_t INF = <DOUBLE_t>np.inf
cdef DOUBLE_t FEATURE_THRESHOLD = 1e-7

cdef struct NodeStats:
    DOUBLE_t sum_y
    DOUBLE_t sum_y_sq
    SIZE_t n_samples
    DOUBLE_t sum_sample_weights

cdef inline void init_NodeStats(NodeStats* stats) nogil:
    stats[0].sum_y = 0.0
    stats[0].sum_y_sq = 0.0
    stats[0].n_samples = 0
    stats[0].sum_sample_weights = 0.0

cdef struct Node:
    cdef NodeStats stats
    cdef DOUBLE_t impurity
    cdef SIZE_t parent_node_id
    cdef SIZE_t right_node_id
    cdef SIZE_t left_node_id

cdef struct Split:
    cdef SIZE_t feature
    cdef DOUBLE_t threshold
    cdef DOUBLE_t impurity_improvement

cdef struct SplitRecord:
    cdef Split split

    cdef SIZE_t split_idx
    cdef SIZE_t prev_idx
    cdef SIZE_t node_id

    cdef NodeStats left_stats
    cdef NodeStats right_stats

cdef inline void init_SplitRecord(SplitRecord* split_record) nogil:
    split_record[0].feature = 0
    split_record[0].split_idx = 0
    split_record[0].prev_idx = 0
    split_record[0].threshold = NAN
    split_record[0].impurity = INF
    split_record[0].impurity_improvement = -INF
    split_record[0].node_id = 0

    init_NodeStats(&split_record[0].left_stats)
    init_NodeStats(&split_record[0].right_stats)

cdef inline void copy_NodeStats(NodeStats* src, NodeStats* dest) nogil:
    dest[0].sum_y = src[0].sum_y
    dest[0].sum_y_sq = src[0].sum_y_sq
    dest[0].n_samples = src[0].n_samples
    dest[0].sum_sample_weights = src[0].sum_sample_weights
