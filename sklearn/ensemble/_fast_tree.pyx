# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Authors: Raghav RV <rvraghav93@gmail.com>
#          Guillaume Lemaitre <guillaume.lemaitre@inria.fr>
#
# License: BSD 3 clause


from __future__ import division, print_function

from collections import defaultdict
from math import ceil
import numbers

import numpy as np
cimport numpy as np
from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE

from sklearn.base import RegressorMixin
from sklearn.externals import six
from sklearn.tree._tree import Tree
from sklearn.tree.tree import BaseDecisionTree
from sklearn.utils.validation import check_array, check_random_state
from sklearn.ensemble.criterion import _impurity_mse

from libcpp.unordered_map import unordered_map as umap

cdef int IS_FIRST = 1
cdef int IS_NOT_FIRST = 0
cdef int IS_LEFT = 1
cdef int IS_NOT_LEFT = 0

cdef SIZE_t TREE_LEAF = -1
cdef SIZE_t TREE_UNDEFINED = -2
cdef SIZE_t INITIAL_STACK_SIZE = 10
cdef FEAT_UNKNOWN = -3


"""
This module implement a regression tree specifically design for the
gradient-boosting regression trees.
"""


class RegressionTree(BaseDecisionTree, RegressorMixin):
    def __init__(self,
                 criterion="mse", splitter="best", max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_features=None, random_state=None, max_leaf_nodes=None,
                 min_impurity_split=1e-7):
        super(RegressionTree, self).__init__(
            criterion=criterion, splitter=splitter, max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features, max_leaf_nodes=max_leaf_nodes,
            random_state=random_state, min_impurity_split=min_impurity_split,
            presort=True)

    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted=None):
        random_state = check_random_state(self.random_state)
        if check_input:
            # FIXME do not accept sparse data for the moment
            X = check_array(X, dtype=DTYPE)
            y = check_array(y, ensure_2d=False, dtype=None)

        # Determine output settings
        n_samples, self.n_features_ = X.shape

        y = np.atleast_1d(y)
        expanded_class_weight = None

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        self.classes_ = [None] * self.n_outputs_
        self.n_classes_ = [1] * self.n_outputs_

        self.n_classes_ = np.array(self.n_classes_, dtype=np.intp)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        # Check parameters
        max_depth = ((2 ** 31) - 1 if self.max_depth is None
                     else self.max_depth)
        max_leaf_nodes = (-1 if self.max_leaf_nodes is None
                          else self.max_leaf_nodes)

        if isinstance(self.min_samples_leaf, (numbers.Integral, np.integer)):
            if not 1 <= self.min_samples_leaf:
                raise ValueError("min_samples_leaf must be at least 1 "
                                 "or in (0, 0.5], got %s"
                                 % self.min_samples_leaf)
            min_samples_leaf = self.min_samples_leaf
        else:  # float
            if not 0. < self.min_samples_leaf <= 0.5:
                raise ValueError("min_samples_leaf must be at least 1 "
                                 "or in (0, 0.5], got %s"
                                 % self.min_samples_leaf)
            min_samples_leaf = int(ceil(self.min_samples_leaf * n_samples))

        if isinstance(self.min_samples_split, (numbers.Integral, np.integer)):
            if not 2 <= self.min_samples_split:
                raise ValueError("min_samples_split must be an integer "
                                 "greater than 1 or a float in (0.0, 1.0]; "
                                 "got the integer %s"
                                 % self.min_samples_split)
            min_samples_split = self.min_samples_split
        else:  # float
            if not 0. < self.min_samples_split <= 1.:
                raise ValueError("min_samples_split must be an integer "
                                 "greater than 1 or a float in (0.0, 1.0]; "
                                 "got the float %s"
                                 % self.min_samples_split)
            min_samples_split = int(ceil(self.min_samples_split * n_samples))
            min_samples_split = max(2, min_samples_split)

        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

        if isinstance(self.max_features, six.string_types):
            if self.max_features == "auto":
                max_features = self.n_features_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError(
                    'Invalid value for max_features. Allowed string '
                    'values are "auto", "sqrt" or "log2".')
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1,
                                   int(self.max_features * self.n_features_))
            else:
                max_features = 0

        self.max_features_ = max_features

        if len(y) != n_samples:
            raise ValueError("Number of labels=%d does not match "
                             "number of samples=%d" % (len(y), n_samples))
        if not 0 <= self.min_weight_fraction_leaf <= 0.5:
            raise ValueError("min_weight_fraction_leaf must in [0, 0.5]")
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than zero. ")
        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")
        if not isinstance(max_leaf_nodes, (numbers.Integral, np.integer)):
            raise ValueError("max_leaf_nodes must be integral number but was "
                             "%r" % max_leaf_nodes)
        if -1 < max_leaf_nodes < 2:
            raise ValueError(("max_leaf_nodes {0} must be either smaller than "
                              "0 or larger than 1").format(max_leaf_nodes))

        if sample_weight is not None:
            if (getattr(sample_weight, "dtype", None) != DOUBLE or
                    not sample_weight.flags.contiguous):
                sample_weight = np.ascontiguousarray(
                    sample_weight, dtype=DOUBLE)
            if len(sample_weight.shape) > 1:
                raise ValueError("Sample weights array has more "
                                 "than one dimension: %d" %
                                 len(sample_weight.shape))
            if len(sample_weight) != n_samples:
                raise ValueError("Number of weights=%d does not match "
                                 "number of samples=%d" %
                                 (len(sample_weight), n_samples))

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Set min_weight_leaf from min_weight_fraction_leaf
        if sample_weight is None:
            sample_weight = np.ones(y.size)
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               n_samples)
        else:
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               np.sum(sample_weight))

        if self.min_impurity_split < 0.:
            raise ValueError("min_impurity_split must be greater than "
                             "or equal to 0")

        # TODO Refactor the min_impurity_decrease here

        # If multiple trees are built on the same dataset, we only want to
        # presort once. Splitters now can accept presorted indices if desired,
        # but do not handle any presorting themselves. Ensemble algorithms
        # which desire presorting must do presorting themselves and pass that
        # matrix into each tree.
        if X_idx_sorted is None and self.presort:
            X_idx_sorted = np.asfortranarray(np.argsort(X, axis=0),
                                             dtype=np.int32)

        if self.presort and X_idx_sorted.shape != X.shape:
            raise ValueError("The shape of X (X.shape = {}) doesn't match "
                             "the shape of X_idx_sorted (X_idx_sorted"
                             ".shape = {})".format(X.shape,
                                                   X_idx_sorted.shape))

        _build_tree(X, y, X_idx_sorted, sample_weight, self.weighted_n_samples,
                    max_features, max_depth, min_samples_split,
                    min_samples_leaf, self.min_weight_fraction_leaf,
                    self.min_impurity_split, self.n_jobs)
        return self


cpdef int _build_tree(np.ndarray X, np.ndarray y, np.ndarray X_idx_sorted,
                      np.ndarray sample_weight, double weighted_n_samples,
                      SIZE_t max_features, SIZE_t max_depth,
                      SIZE_t min_samples_split, SIZE_t min_samples_leaf,
                      double min_weight_fraction_leaf,
                      double min_impurity_split, SIZE_t n_jobs):
    cdef SIZE_t n_nodes

    cdef SIZE_t current_depth = 0
    cdef bint early_stop = 0

    cdef SIZE_t n_samples = X.shape[0]
    cdef SIZE_t n_features = X.shape[1]
    cdef SIZE_t n_outputs = 1  # TODO Support n_outputs > 1

    # The id-s of all expanding nodes (nodes that are not leaf at
    # the current height/level)
    cdef SIZE_t* expanding_nodes
    cdef SIZE_t n_expanding_nodes

    cdef SplitRecord *current_splits, *best_splits
    cdef SplitRecord current_split, best_split

    # This maps the node id to the corresponding index of current_splits,
    # best_splits and expanding_nodes
    cdef umap[SIZE_t, SIZE_t]* node_id_to_split_idx_map

    cdef SIZE_t node_id
    cdef SIZE_t i, j, p q, feat_i

    cdef double sum_y, sum_ysq, mean_y, impurity, impurity_improvement
    cdef double *value_array, *node_array
    cdef double *node_stats, *node

    # XXX For sparse this should be a map?
    # Which node each sample belongs to at current level
    # 0 denotes a sample belonging to a leaf node
    cdef SIZE_t* sample_idx_to_node_id_map
    safe_realloc(&sample_idx_to_node_id_map, n_samples)

    # All samples belong to the root node
    for i in range(n_samples):
        sample_idx_to_node_id_map[i] = 1

    # n_classes is set to 3, the number of statistic that you store
    # for each node.
    # This is a hack to avoid additional structure(s) to
    # store the sum_y, sum_y_sq of each node and reuse the
    # self.tree_.value, which will be a 3D array of shape
    # n_outputs x n_nodes x n_classes(=3)
    # We store `[mean_y, sum_y, sum_y_sq]` per node for each target/output
    # XXX Is this hack okay or should we do it differently?
    self.tree_ = Tree(n_features=n_features, n_classes=3, n_outputs=n_outputs)
    value_array = self.tree_.value
    node_array = self.tree_.node

    # Create the root node
    node_id = self.tree_._add_node(self,
                                   parent=TREE_UNDEFINED,
                                   is_left=false, is_leaf=false,
                                   feature=TREE_UNDEFINED,
                                   threshold=np.nan, impurity=np.inf,
                                   n_node_samples=n_samples,
                                   weighted_n_samples=np.sum(sample_weight))

    # Update the statistics for this root node
    for i in range(n_outputs):
        # Get the pointer for the current (root) node object
        node = self.tree_.node[node_id]
        # Get the pointer for the 1D value array of this node for i-th output
        # See sklearn.tree.tree_.Tree.value
        node_stats = value_array[node_id][i]

        # sum_y
        sum_y = node_stats[1] = np.sum(np.ravel(y[:, i]) * sample_weight)
        # mean_y, this will be the prediction value for this node
        mean_y = node_stats[0] = sum_y / node.weighted_n_node_samples
        # sum_ysq
        sum_ysq = node_stats[2] = np.sum(np.ravel(y[:, i] ** 2) * sample_weight)
        # mse of this root node
        node.impurity = _impurity_mse(node, node_stats)

    # Root is the only expanding node at level 0 and all the samples belong to
    # the root node
    n_nodes = n_expanding_nodes = 1
    safe_realloc(&expanding_nodes, n_expanding_nodes)
    expanding_nodes[0] = 1  # node_id of the root is 1

    for i in range(n_samples):
        sample_idx_to_node_id_map[i] = 1

    # Todo update early stop accordingly
    current_depth = 0
    max_depth = 1
    while (n_expanding_nodes <= 0 or early_stop or
               current_depth >= max_depth):
        current_depth += 1

        # (Re)Allocate memory in heap for this map for the expanding nodes at
        # current height
        del node_id_to_split_idx_map
        node_id_to_split_idx_map = new umap[SIZE_t, SIZE_t]()

        # Re allocate current_splits and best_splits based on n_expanding_nodes
        # n_explanding_nodes / expanding_nodes is updated at the end of each
        # splitting iteration
        safe_realloc(&current_splits, n_expanding_nodes)
        safe_realloc(&best_splits, n_expanding_nodes)

        # expanding_nodes contains the list of node_id-s of nodes that can be
        # expanded (split) into further partitions at the current height
        for i in range(n_expanding_nodes):
            node_id = expanding_nodes[i]
            # A reverse lookup map for easy access to the current_splits array
            node_id_to_split_idx_map[node_id] = i

        reset_split_records(current_splits, node_array, value_array,
                            expanding_nodes, n_expanding_nodes)
        # Now mem-copy this block of init-ed data to best_splits
        memcopy(current_splits, best_splits,
                n_expanding_nodes * sizeof(SplitRecord))

        # shuffled the features and select max_features number of features
        permuted_features = random_state.choice(np.arange(X.shape[1]),
                                                size=max_features,
                                                replace=False)
        # TODO: handle constant features
        for i in range(max_features):
            # TODO this is the function to pass via joblib
            # TODO if passing via joblib, we need one best_splits and
            # current_splits array per feature
            # And aggregate all the best splits to find the final best split
            # at the end

            # Find the best splits at this level,
            find_best_splits(
                X_idx_sorted,
                permuted_features[i],   # split in this feature space
                expanding_nodes,  # for these node-ids that are not leaf at
                                  # current level
                current_splits,  # Use this to store stats as we search through
                best_splits,     # And this to store the best split so far
                node_id_to_split_idx_map,   # To get the index for current_splits
                                            # and best_splits given the node_id
                sample_idx_to_node_id_map,  # To know which sample belongs to
                                            # which node-id, currently
                # TODO now we use mse by default. Pass it either as an int
                # option or as a function callback
                )

        # TODO Should we also parallelize this?
        # This seems to take up quite some time
        # If this needs paralelization, we have to split number of nodes to
        # n_jobs partition and then update the tree. But before, we have to
        # initialize it with appropriate size
        for node_id in expanding_nodes:
            j = node_id_to_split_idx_map[node_id]

            # For those nodes where a split was not possible in any feature,
            # the best_split's pos will remain unchanged from the init
            # value (which is 0)
            # We mark all samples from those nodes as leaf
            best_split = best_splits[j]
            best_feature = best_split.feature
            best_threshold = best_split.threshold

            if best_split.pos == 0:
                # This is not optimal. How can we speed this up?
                for i in range(n_samples):
                    if sample_idx_to_node_id_map[i] == node_id:
                        sample_idx_to_node_id_map[i] = 0
                mark_node_as_leaf(node_array[node_id])
            else:
                # For all other nodes we split it according to the best_split
                # by creating the left and right nodes accordingly
                left_nid = self.tree_._add_node_and_update_value(
                    parent=node_id, is_left=1, is_leaf=TREE_LEAF,
                    feature=FEAT_UNKNOWN, threshold=TREE_UNDEFINED,
                    impurity=best_split.impurity_left,
                    n_node_samples=best_split.n_left,
                    weighted_n_node_samples=best_split.weighted_n_left
                    node_value=(best_split.c_stats.sum_y_left /
                                best_split.weighted_n_left))

                right_nid = self.tree_._add_node_and_update_value(
                    parent=node_id, is_left=0, is_leaf=TREE_LEAF,
                    feature=FEAT_UNKNOWN, threshold=TREE_UNDEFINED,
                    impurity=best_split.impurity_right,
                    n_node_samples=best_split.n_right,
                    weighted_n_node_samples=best_split.weighted_n_right
                    node_value=(best_split.sum_y_right /
                                best_split.weighted_n_right))

                # Update the parent node with split information
                self.tree_._update_node(
                    node_id=node_id, left_child=left_nid,
                    right_child=right_nid, threshold=best_split.threshold,
                    impurity=best_split.impurity, feature=best_split.feature)

                # Update sample_idx_to_node_id_map
                for i in range(n_samples):
                    # If the sample belonged to this node
                    if sample_idx_to_node_id_map[i] == node_id:
                        # Check the data value and associate it with right or
                        # left based on the found split threshold
                        if X[i, best_feature] < best_threshold:
                            sample_idx_to_node_id_map[i] = left_nid
                        else:
                            sample_idx_to_node_id_map[i] = right_nid

    del node_id_to_split_idx_map
    del current_splits
    del best_splits
    return self


cdef inline int reset_split_records(SplitRecord* split_records,
                                    Node* node_array,
                                    double* value_array,
                                    SIZE_t* expanding_nodes,
                                    SIZE_t n_expanding_nodes) nogil except -1:
    cdef SIZE_t node_id
    for i in range(n_expanding_nodes):
        node_id = expanding_nodes[i]
        # One SplitRecord struct per expanding node to evaluate the splits
        # as search for the split position progresses from left to right
        # Start with all samples on the right partition
        init_SplitRecord_from_parent(split_records[i],
                                     # (current node is the parent of
                                     #  subsequent splits)
                                     node_array[node_id],
                                     value_array[node_id])

        # best_split entries will be memcopy-ied from the current_splits
        # so, no need to init them here
        return 0


cdef int mark_node_as_leaf(Node n):
    n.left_child = _TREE_LEAF
    n.right_child = _TREE_LEAF
    n.feature = _TREE_UNDEFINED
    n.threshold = _TREE_UNDEFINED


cpdef int find_best_splits(np.ndarray X_idx_sorted, np.ndarray y,
                           SIZE_t n_samples,
                           SIZE_t feature,
                           SIZE_t* expanding_nodes,
                           SIZE_t n_expanding_nodes,
                           SplitRecord* current_splits,
                           SplitRecord* best_splits,
                           umap[SIZE_t, SIZE_t] node_id_to_split_idx_map,
                           SIZE_t* sample_idx_to_node_id_map,
                           SIZE_t min_samples_split,
                           SIZE_t min_samples_leaf) nogil except -1:
    cdef SIZE_t i, j, p, q, node_id
    cdef DTYPE_t x_val
    for i in range(n_samples):
        node_id = sample_idx_to_node_id_map[i]
        if node_id == 0:  # If the sample is alreay in leaf skip
            continue
        # Else find the node that the sample belongs to and move it from
        # right to left partition
        x_val = X[X_idx_sorted[i, feature], feature]
        j = node_id_to_split_idx_map[node_id]
        current_splits[j].sum_y_right -= x_val
        current_splits[j].sum_y_left += x_val
    return 0