#cython: boundscheck=False
#cython: cdivision=True
#cython: warparound=False

from libcpp.map cimport map
from libc.stdlib cimport malloc, free
from libc.math cimport NAN, INFINITY, isnan

from ._tree cimport Node

from ._splitter cimport Splitter
from ._splitter cimport splitter_init
from ._splitter cimport splitter_reset
from ._splitter cimport splitter_node_evaluate_split
from ._splitter cimport splitter_expand
from ._splitter cimport splitter_set_nid
from ._splitter cimport splitter_copy_to

from ._split_record cimport SplitRecord
from ._split_record cimport split_record_reset

from ._stats_node cimport StatsNode
from ._stats_node cimport stats_node_reset
from ._stats_node cimport stats_node_clear

from ._criterion cimport _impurity_mse

# from sklearn.utils import check_random_state


cdef:
    int TREE_UNDEFINED = -2
    int FEAT_UNKNOWN = -3
    int TREE_LEAF = -1
    bint TREE_NOT_LEAF = 0


cdef void weighted_sum_y(double[::1] y, double[::1] sample_weight,
                         double* p_sum_y, double* p_sum_sq_y):
    cdef int i
    p_sum_y[0] = 0.0
    p_sum_sq_y[0] = 0.0
    for i in range(y.shape[0]):
        p_sum_y[0] += y[i] * sample_weight[i]
        p_sum_sq_y[0] += y[i] ** 2 * sample_weight[i]


cdef class ExactTreeBuilder(TreeBuilder):

    def __cinit__(self, int min_samples_split,
                  int min_samples_leaf, double min_weight_leaf,
                  double min_impurity_split,
                  int max_depth, int max_features):
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.min_impurity_split = min_impurity_split
        self.max_depth = max_depth
        self.max_features = max_features

    cpdef build(self, Tree tree, float[:, ::1] X, int[::1, :] X_idx_sorted,
                double[::1] y, double[::1] sample_weight,
                double sum_total_weighted_samples):

        # FIXME: don't set the random state here
        # rng = check_random_state(0)

        # we don't do any checking for the moment

        cdef:
            int n_samples = X.shape[0]
            int n_features = X.shape[1]
            int init_capacity
            int n_splitter
            int max_n_splitter
            map[int, int] nid_to_splitter_idx_
            Splitter* splitters_
            int next_n_splitter
            int next_max_n_splitter
            Splitter* next_splitters_

        if tree.max_depth <= 10:
            init_capacity = (2 ** (tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047

        tree._resize(init_capacity)

        ##################################################################
        # Initialization
        ##################################################################
        n_splitter = 1
        max_n_splitter = 1
        splitters_ = <Splitter*> malloc(max_n_splitter * sizeof(Splitter))

        # compute the stats for the root node
        cdef:
            double root_sum_y = 0.0
            double root_sum_sq_y = 0.0
            int root_n_samples = n_samples
            double root_sum_weighted_samples = sum_total_weighted_samples
        weighted_sum_y(y, sample_weight, &root_sum_y, &root_sum_sq_y)
        # init a stats_node and split_record which will be used to create
        # the splitter
        cdef:
            SplitRecord split_record
            StatsNode left_stats_node, right_stats_node
        stats_node_clear(&left_stats_node)
        stats_node_reset(&right_stats_node, root_sum_y,
                         root_sum_sq_y, root_n_samples,
                         root_sum_weighted_samples)
        split_record_reset(&split_record, 0, 0, NAN,
                           _impurity_mse(&right_stats_node),
                           -INFINITY, 0,
                           &right_stats_node, &left_stats_node,
                           &right_stats_node)
        # create the tree node
        split_record.nid = tree._add_node_with_value(
            TREE_UNDEFINED,
            1, TREE_NOT_LEAF,
            FEAT_UNKNOWN,
            TREE_UNDEFINED,
            split_record.impurity,
            n_samples,
            sum_total_weighted_samples,
            root_sum_y / root_sum_weighted_samples)
        # init the splitter
        splitter_init(&splitters_[0],
                      FEAT_UNKNOWN, TREE_UNDEFINED,
                      &split_record, self.min_samples_leaf,
                      self.min_weight_leaf)
        # add the correspondence nid to splitter idx
        nid_to_splitter_idx_[split_record.nid] = 0
        # initialize an array storing the correspondance between
        # each sample and node id
        cdef:
            int i
            int* X_nid = <int*> malloc(n_samples * sizeof(int))
            bint* X_nid_visited = <bint*> malloc(n_samples * sizeof(bint))
            int* count_X_nid_label = <int*> malloc(max_n_splitter * sizeof(int))
        for i in range(n_samples):
            X_nid[i] = 0
            X_nid_visited[i] = 0
        for i in range(max_n_splitter):
            count_X_nid_label[i] = 0

        cdef:
            int j
            int current_depth = 0
            int n_visited_feature = 0
            int feat_idx
            int[::1] shuffled_feature_idx
            int sample_idx_sorted
            int[::1] X_col
            bint b_grow
            int X_idx
            int parent_n_left_samples
            int splitter_idx
            bint b_impurity
            bint b_samples_split
            bint b_samples_leaf
        while current_depth < self.max_depth:
            #shuffled_feature_idx = rng.permutation(range(n_features)).astype(np.int32)

            n_visited_feature = 0

            for i in range(n_features):
                # feat_idx = shuffled_feature_idx[i]
                feat_idx = i

                if n_visited_feature >= self.max_features:
                    break

                X_col = X_idx_sorted[:, feat_idx]

                # reset the split record for the current feature
                # keeping the best split record untouched
                for j in range(n_splitter):
                    splitter_reset(&splitters_[j], feat_idx, X_col[0])

                # evaluate all possible split for the different samples
                for j in range(X_col.shape[0]):
                    sample_idx_sorted = X_col[j]
                    if X_nid[sample_idx_sorted] != -1:
                        # FIXME: need to incorporate constant feature
                        splitter_node_evaluate_split(
                            &splitters_[
                                nid_to_splitter_idx_[X_nid[sample_idx_sorted]]],
                            X, y, sample_weight, sum_total_weighted_samples,
                            sample_idx_sorted)

                n_visited_feature += 1

            # expand the splitter if needed
            next_max_n_splitter = 2 * n_splitter
            next_n_splitter = 0
            next_splitters_ = <Splitter*> malloc(next_max_n_splitter * sizeof(Splitter))
            b_grow = 0
            for i in range(n_splitter):
                if isnan(splitters_[i].best_split_record.threshold):
                    tree._update_node(
                        splitters_[i].split_record.nid,
                        TREE_LEAF, TREE_LEAF,
                        TREE_UNDEFINED,
                        splitters_[i].original_split_record.impurity,
                        TREE_UNDEFINED,
                        splitters_[i].original_split_record.c_stats.n_samples,
                        splitters_[i].original_split_record.c_stats.sum_weighted_samples)
                else:
                    splitter_expand(&splitters_[i],
                                    &next_splitters_[next_n_splitter],
                                    &next_splitters_[next_n_splitter + 1])
                    # add the left node
                    left_nid = tree._add_node_with_value(
                        splitters_[i].split_record.nid,
                        1, TREE_NOT_LEAF, FEAT_UNKNOWN, TREE_UNDEFINED,
                        next_splitters_[next_n_splitter].split_record.impurity,
                        next_splitters_[next_n_splitter].split_record.c_stats.n_samples,
                        next_splitters_[next_n_splitter].split_record.c_stats.sum_weighted_samples,
                        next_splitters_[next_n_splitter].split_record.c_stats.sum_y /
                        next_splitters_[next_n_splitter].split_record.c_stats.sum_weighted_samples)
                    # add the right node
                    right_nid = tree._add_node_with_value(
                        splitters_[i].split_record.nid,
                        0, TREE_NOT_LEAF, FEAT_UNKNOWN, TREE_UNDEFINED,
                        next_splitters_[next_n_splitter + 1].split_record.impurity,
                        next_splitters_[next_n_splitter + 1].split_record.c_stats.n_samples,
                        next_splitters_[next_n_splitter + 1].split_record.c_stats.sum_weighted_samples,
                        next_splitters_[next_n_splitter + 1].split_record.c_stats.sum_y /
                        next_splitters_[next_n_splitter + 1].split_record.c_stats.sum_weighted_samples)
                    # add the id to the different split_record
                    splitter_set_nid(&next_splitters_[next_n_splitter], left_nid)
                    splitter_set_nid(&next_splitters_[next_n_splitter + 1], right_nid)

                    # only consider the new splitter if there is enough data
                    # or that the impurity is large enough
                    b_impurity = (
                        next_splitters_[next_n_splitter].split_record.impurity >
                        self.min_impurity_split)
                    b_samples_split = (
                        next_splitters_[next_n_splitter].split_record.c_stats.n_samples >=
                        self.min_samples_leaf)
                    b_samples_leaf = (
                        next_splitters_[next_n_splitter].split_record.c_stats.n_samples >=
                        self.min_weight_leaf)

                    if b_impurity and b_samples_leaf and b_samples_split:
                        b_grow = 1
                        nid_to_splitter_idx_[left_nid] = next_n_splitter
                        next_n_splitter += 1
                    else:
                        tree._update_node(
                            left_nid,
                            TREE_LEAF, TREE_LEAF,
                            TREE_UNDEFINED,
                            next_splitters_[next_n_splitter].original_split_record.impurity,
                            TREE_UNDEFINED,
                            next_splitters_[next_n_splitter].original_split_record.c_stats.n_samples,
                            next_splitters_[next_n_splitter].original_split_record.c_stats.sum_weighted_samples)
                        splitter_copy_to(&next_splitters_[next_n_splitter + 1],
                                         &next_splitters_[next_n_splitter])

                    b_impurity = (
                        next_splitters_[next_n_splitter].split_record.impurity >
                        self.min_impurity_split)
                    b_samples_split = (
                        next_splitters_[next_n_splitter].split_record.c_stats.n_samples >=
                        self.min_samples_leaf)
                    b_samples_leaf = (
                        next_splitters_[next_n_splitter].split_record.c_stats.n_samples >=
                        self.min_weight_leaf)

                    if b_impurity and b_samples_leaf and b_samples_split:
                        b_grow = 1
                        nid_to_splitter_idx_[right_nid] = next_n_splitter
                        next_n_splitter += 1
                    else:
                        tree._update_node(
                            right_nid,
                            TREE_LEAF, TREE_LEAF,
                            TREE_UNDEFINED,
                            next_splitters_[next_n_splitter].original_split_record.impurity,
                            TREE_UNDEFINED,
                            next_splitters_[next_n_splitter].original_split_record.c_stats.n_samples,
                            next_splitters_[next_n_splitter].original_split_record.c_stats.sum_weighted_samples)

                    # update the parent node
                    tree._update_node(
                        splitters_[i].split_record.nid,
                        left_nid, right_nid,
                        splitters_[i].best_split_record.threshold,
                        splitters_[i].best_split_record.impurity,
                        splitters_[i].best_split_record.feature,
                        splitters_[i].best_split_record.c_stats.n_samples,
                        splitters_[i].best_split_record.c_stats.sum_weighted_samples)

            # affect each samples to the right node
            if b_grow:
                # for i in range(n_samples):
                #     X_nid_tmp[i] = X_nid[i]
                for i in range(n_samples):
                    for j in range(n_features):
                        X_idx = X_idx_sorted[i, j]
                        if X_nid_visited[X_idx] == 0:
                            parent_nid = X_nid[X_idx]
                            if parent_nid != -1:
                                splitter_idx = nid_to_splitter_idx_[parent_nid]
                                if (splitters_[splitter_idx].best_split_record.feature == j):
                                    parent_n_left_samples = splitters_[splitter_idx].best_split_record.l_stats.n_samples
                                    if isnan(splitters_[splitter_idx].best_split_record.threshold):
                                        X_nid[X_idx] = -1
                                    else:
                                        count_X_nid_label[splitter_idx] += 1
                                        if count_X_nid_label[splitter_idx] <= parent_n_left_samples:
                                            if tree.nodes[parent_nid].left_child == TREE_LEAF:
                                                X_nid[X_idx] = -1
                                            else:
                                                if (tree.nodes[tree.nodes[parent_nid].left_child].left_child == TREE_LEAF and
                                                    tree.nodes[tree.nodes[parent_nid].left_child].right_child == TREE_LEAF):
                                                    X_nid[X_idx] = -1
                                                else:
                                                    X_nid[X_idx] = tree.nodes[parent_nid].left_child
                                        else:
                                            if tree.nodes[parent_nid].right_child == TREE_LEAF:
                                                X_nid[X_idx] = -1
                                            else:
                                                if (tree.nodes[tree.nodes[parent_nid].right_child].left_child == TREE_LEAF and
                                                    tree.nodes[tree.nodes[parent_nid].right_child].right_child == TREE_LEAF):
                                                    X_nid[X_idx] = -1
                                                else:
                                                    X_nid[X_idx] = tree.nodes[parent_nid].right_child
                                    X_nid_visited[X_idx] = 1
                            else:
                                X_nid_visited[X_idx] = 1

            for i in range(n_samples):
                X_nid_visited[i] = 0

            # free memory
            free(splitters_)
            free(count_X_nid_label)

            splitters_ = next_splitters_
            next_splitters_ = NULL

            max_n_splitter = next_max_n_splitter
            n_splitter = next_n_splitter

            count_X_nid_label = <int*> malloc(max_n_splitter * sizeof(int))
            for i in range(max_n_splitter):
                count_X_nid_label[i] = 0

            if b_grow:
                current_depth += 1
            else:
                break

        # Set all remaining nodes as leaf
        for i in range(n_splitter):
            tree._update_node(splitters_[i].split_record.nid,
                              TREE_LEAF, TREE_LEAF,
                              TREE_UNDEFINED,
                              splitters_[i].best_split_record.impurity,
                              TREE_UNDEFINED,
                              splitters_[i].best_split_record.c_stats.n_samples,
                              splitters_[i].best_split_record.c_stats.sum_weighted_samples)

        # Deallocate X_nid and splitters_
        free(splitters_)
        free(count_X_nid_label)
        free(X_nid)
        free(X_nid_visited)

        rc = tree._resize_c_py(tree.node_count)

        if rc >= 0:
            tree.max_depth = current_depth
