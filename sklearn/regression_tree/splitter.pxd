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

cdef class Splitter:
    cdef np.ndarray X
    cdef np.ndarray y
    cdef np.ndarray sample_weight
    cdef DOUBLE_t sum_total_weighted_samples
    cdef SIZE_t feature_idx
    cdef SIZE_t start_idx
    cdef SIZE_t prev_idx
    cdef SplitRecord split_record
    cdef public SplitRecord best_split_record
    cdef SIZE_t min_samples_leaf
    cdef DOUBLE_t min_weight_leaf

    cdef StatsNode temp

    cpdef reset(self, SIZE_t feature_idx, SIZE_t start_idx,
                SplitRecord split_record)
    cpdef update_stats(self, SIZE_t sample_idx)
    cpdef node_evaluate_split(self, SIZE_t sample_idx)

cdef struct SplitRecord:
    cdef public SIZE_t feature
    cdef public SIZE_t pos
    cdef public DOUBLE_t threshold
    cdef public DOUBLE_t impurity
    cdef public DOUBLE_t impurity_improvement
    cdef public SIZE_t nid

    cdef public StatsNode c_stats
    cdef public StatsNode l_stats
    cdef public StatsNode r_stats

cdef struct NodeStats:
    DOUBLE_t sum_y
    DOUBLE_t sum_sq_y
    SIZE_t n_samples
    DOUBLE_t sum_weighted_samples

cdef inline void set_SplitRecord(
        SplitRecord* split_record, SIZE_t feature=0, SIZE_t pos=0,
        DOUBLE_t threshold=0.0, DOUBLE_t impurity=0.0,
        DOUBLE_t impurity_improvement=0.0,
        SIZE_t nid=0, StatsNode *c_stats=NULL, StatsNode *l_stats=NULL,
        StatsNode *r_stats=NULL):
    split_record[0].feature = feature
    split_record[0].pos = pos
    split_record[0].threshold = threshold
    split_record[0].impurity = impurity
    split_record[0].impurity_improvement = impurity_improvement
    split_record[0].nid = nid

    if c_stats != NULL:
        copy_NodeStats(c_stats, &split_record[0].c_stats)
    if l_stats != NULL:
        copy_NodeStats(l_stats, &split_record[0].l_stats)
    if r_stats != NULL:
        copy_NodeStats(r_stats, &split_record[0].r_stats)

cdef inline void init_NodeStats(NodeStats* node):
    node[0].sum_y = 0.0
    node[0].sum_sq_y = 0.0
    node[0].n_samples = 0
    node[0].sum_weighted_samples = 0.0

cdef inline void copy_NodeStats(NodeStats* src, NodeStats* dest):
    dest[0].sum_y = src[0].sum_y
    dest[0].sum_sq_y = src[0].sum_sq_y
    dest[0].n_samples = src[0].n_samples
    dest[0].sum_weighted_samples = src[0].sum_weighted_samples
