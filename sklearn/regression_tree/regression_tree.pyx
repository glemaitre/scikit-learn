# Decision_tree_learning

"""
This module implement a regression tree specifically design for the
gradient-boosting regression trees.
"""
from __future__ import division, print_function

import numbers

from collections import defaultdict
from math import ceil

from libcpp.unordered_map import unordered_map as umap

import numpy as np
cimport numpy as cnp

from ..tree.tree import BaseDecisionTree
from ..base import RegressorMixin
from ..utils.validation import check_array, check_random_state
from ..externals import six

from .splitter import Splitter
from .split_record import SplitRecord
from .stats_node import StatsNode
from .criterion import _impurity_mse_py

from ..tree._tree import Tree
from ..tree import _tree

TREE_UNDEFINED, TREE_LEAF, FEAT_UNKNOWN = -2, -1, -3
DTYPE = _tree.DTYPE
DOUBLE = _tree.DOUBLE


class RegressionTree(BaseDecisionTree, RegressorMixin):
    """A regression tree specifically designed for gradient-boosting.

    Parameters
    ----------
    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion, and "mae" for the mean
        absolute error.

        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

    splitter : string, optional (default="best")
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for percentages.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for percentages.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_leaf_nodes : int or None, optional (default=None)
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    min_impurity_split : float, optional (default=1e-7)
        Threshold for early stopping in tree growth. If the impurity
        of a node is below the threshold, the node is a leaf.

        .. versionadded:: 0.18

    Attributes
    ----------
    feature_importances_ : array of shape = [n_features]
        The feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the
        (normalized) total reduction of the criterion brought
        by that feature. It is also known as the Gini importance [4]_.

    max_features_ : int,
        The inferred value of max_features.

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : Tree object
        The underlying Tree object.

    See also
    --------
    DecisionTreeClassifier, DecisionTreeRegressor

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Decision_tree_learning

    .. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
           and Regression Trees", Wadsworth, Belmont, CA, 1984.

    .. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
           Learning", Springer, 2009.

    .. [4] L. Breiman, and A. Cutler, "Random Forests",
           http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
    """
    def __init__(self,
                 criterion="mse",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_split=1e-7):
        super(RegressionTree, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            min_impurity_split=min_impurity_split,
            presort=True)

    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted=None):
        """Build a decision tree regressor from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (real numbers). Use ``dtype=np.float64`` and
            ``order='C'`` for maximum efficiency.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        X_idx_sorted : array-like, shape = [n_samples, n_features], optional
            The indexes of the sorted training input samples. If many tree
            are grown on the same dataset, this allows the ordering to be
            cached between trees. If None, the data will be sorted here.
            Don't use this parameter unless you know what to do.

        Returns
        -------
        self : object
            Returns self.
        """
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

        self._build_tree(X, y, X_idx_sorted, sample_weight, weighted_n_samples,
                         max_features, max_depth, min_samples_split,
                         min_samples_leaf, min_weight_fraction_leaf,
                         min_impurity_split)
        return self

    cpdef int self._build_tree(np.ndarray[DTYPE_t, ndim=2] X,
                               np.ndarray[SIZE_t, ndim=2] X_idx_sorted,
                               np.ndarray[DOUBLE_t, ndim=1] y,
                               np.ndarray[DOUBLE_t, ndim=1] sample_weight,
                               DOUBLE_t weighted_n_samples,
                               SIZE_t max_features, SIZE_t max_depth,
                               SIZE_t min_samples_split, SIZE_t min_samples_leaf,
                               DOUBLE_t min_weight_fraction_leaf,
                               DOUBLE_t min_impurity_split,
                               random_state) except 0:
        cdef Node* nodes
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

        cdef SplitRecord* expanding_current_split_records
        cdef SplitRecord* expanding_best_split_records

        # XXX Is it cleaner to directly use
        # umap[SIZE_t, <SplitRecord, SplitRecord>]
        # or is this better as expanding_current_records will be contiguous
        cdef umap[SIZE_t, SIZE_t]* node_id_to_record_idx_map

        cdef SIZE_t node_id
        cdef SIZE_t i, j, p q

        cdef double sum_y, sum_y_sq, mean_y, impurity, impurity_improvement
        cdef double* value_array

        # XXX For sparse this should be a map?
        cdef SIZE_t* sample_idx_to_node_id_map
        safe_realloc(&sample_idx_to_node_id_map, n_samples)

        n_expanding_nodes = 1  # Starting with one node, the root node with all the data
        safe_realloc(&expanding_current_split_records, n_expanding_nodes)
        safe_realloc(&expanding_best_split_records, n_expanding_nodes)

        for i in range(n_samples):
            # All samples belong to the root node
            sample_idx_to_node_id_map[i] = 1

        # n_classes is set to 3, the number of statistic that you store
        # for each node.
        # This is a hack to avoid additional structure(s) to
        # store the sum_y, sum_y_sq of each node and reuse the
        # self.tree_.value which will be a 3D array with one 2D array having
        # data as [[mean_y, sum_y, sum_y_sq] per output] per node
        # XXX Is this hack okay or should we do it differently?
        self.tree_ = Tree(n_features=n_features, n_classes=3,
                          n_outputs=n_outputs)

        node_id = self.tree_._add_node(self,
                                       parent=TREE_UNDEFINED,
                                       is_left=false, is_leaf=false,
                                       feature=TREE_UNDEFINED,
                                       threshold=NAN,
                                       impurity=INFINITY,
                                       n_node_samples=n_samples,
                                       weighted_n_samples=np.sum(
                                           sample_weight))

        # Update the statistics for this root node
        for i in range(n_outputs):
          value_array = self.tree_.value[node_id][i]
          weighted_n_node_samples = (
            self.tree_.node_id[node_id].weighted_n_node_samples)

          # sum_y
          value_array[1] = np.sum(np.ravel(y) * sample_weight)
          # mean_y, this will be the prediction value for this node
          value_array[0] = value_array[1] / weighted_n_node_samples
          # sum_sq_y
          value_array[2] = np.sum(np.ravel(y ** 2) * sample_weight)

          self.tree_.node[node_id].impurity = _impurity_mse(
              sum_y, sum_y_sq,
              self.tree.nodes[node_id].weighted_n_node_samples)

        n_expanding_nodes = 1
        safe_realloc(&expanding_nodes, n_expanding_nodes)
        # Root is the only expanding node for now
        expanding_nodes[0] = node_id

        for i in range(n_samples):
            # All the samples belong to the root node
            sample_idx_to_node_id_map[i] = 1

        # Allocate memory in heap  XXX Remember to `del` it before
        # Mapping the tree ``node_id`` to the index of the split record in the
        # expanding_nodes array.
        node_id_to_record_idx_map = new umap[SIZE_t, SIZE_t]()
        node_id_to_record_idx_map[1] = 0

        while (n_expanding_nodes <= 0 or early_stop or
                   current_depth >= max_depth):

            current_depth += 1

            # expanding_nodes contains the list of node_id-s that can be
            # expanded (split) into further partitions
            for expanding_node_i in range(n_expanding_nodes):
                node_id = expanding_nodes[expanding_node_i]
                # Store a map entry to help access index via node id
                node_id_to_array_idx[node_id] = expanding_node_i

                # expanding_nodes is 0-indexed, thanks to node_id_to_array_idx
                safe_realloc(&current_splits, n_expanding_nodes)
                safe_realloc(&best_splits, n_expanding_nodes)

                # One working SplitRecord object per expanding node
                init_SplitRecord(&current_splits[expanding_node_i])
                # Update with statistics of root node
                current_splits[expanding_node_i].node_id = node_id
                # Initially all the data is assumed to be in right and later we
                # traverse by moving one sample at a time to the left partition
                # So, set 0 to left and copy the whole node statistics to the
                # right
                init_NodeStats(&current_splits[expanding_node_i].left_stats)
                copy_NodeStats(&nodes[node_id].stats,
                               &current_splits[expanding_node_i].right_stats)

        # One best-so-far SplitRecord object per expanding node
        parent_split_records[1].c_stats = &node_stats_array[1]
        parent_split_records[1].impurity = impurity

        weighted_n_samples = np.sum(sample_weight)

        # initialize the number of splitter

        # the array to map the samples to the correct splitter
        X_nid = np.zeros(y.size, dtype=int)

        # the list of splitter at each round
        splitter_list = []

        # the output split record
        split_record_map = defaultdict(lambda: None)


        impurity = _impurity_mse_py(root_stats)

        # We start with the root node
        safe_realloc(&expandable_node_ids, 1)
        safe_realloc(&node_split_records, 1)
        safe_realloc(&parent_node_ids, 1)
        safe_realloc(&node_stats_array, 1)

        # create the root node statistics for the first parent
        len_split_records = 1
        n_explandables = 1
        expandable_node_ids[0] = node_id_p
        init_SplitRecord(&parent_split_records[0])
        node_stats_array[0].sum_y=np.sum(np.ravel(y) * sample_weight),
        node_stats_array[0].sum_sq_y=np.sum(np.ravel(y ** 2) * sample_weight),
        node_stats_array[0].n_samples=n_samples,
        node_stats_array[0].sum_weighted_samples=weighted_n_samples)
        parent_split_records[0].c_stats = &node_stats_array[0]
        parent_split_records[0].impurity = impurity

        current_depth = 0
        while current_depth < max_depth:
            if n_explandables > n_split_records:
                # Extend the split_records array if needed
                safe_realoc(&parent_split_records, n_explandables)

                splitter_list += [Splitter(X, y, sample_weight,
                                           weighted_n_samples,
                                           FEAT_UNKNOWN, TREE_UNDEFINED,
                                           parent_split_map[nid],
                                           min_samples_leaf,
                                           min_weight_leaf)
                                  for nid in expandable_nids[
                                          curr_n_splitters:]]

            # drop splitters
            else:
                splitter_list = splitter_list[:n_splitters]

            # create a dictionary from the list of splitter
            splitter_map = {nid: splitter_list[i]
                            for i, nid in enumerate(expandable_nids)}

            # Create an array from where to select randomly the feature
            shuffled_feature_idx = random_state.choice(np.arange(X.shape[1]),
                                                       size=self.max_features_,
                                                       replace=False)

            joblibparallel(find_best_split)
            # get the feature
            for feat_i in shuffled_feature_idx:
                # Get the sorted index
                X_col = X_idx_sorted[:, feat_i]

                # reset the splitter

                for i, nid in enumerate(expandable_nids):
                    splitter_map[nid].reset(feat_i, X_col[0],
                                            parent_split_map[nid])


                # scans all samples and evaluate all possible splits for all
                # the different splitters
                for sample_idx_sorted in X_col:
                    # Samples which are not in a leaf
                    if X_nid[sample_idx_sorted] != -1:
                        # check that the sample value are different enough
                        splitter_map[X_nid[
                            sample_idx_sorted]].node_evaluate_split(
                                sample_idx_sorted)

                # copy the split_record if the improvement is better
                for nid in expandable_nids:
                    if ((split_record_map[nid] is None) or
                            (splitter_map[nid].best_split_record.impurity_improvement >
                             split_record_map[nid].impurity_improvement)):
                        best = splitter_map[nid].best_split_record
                        split_record_map[nid] = SplitRecord()
                        split_record_map[nid].copy_from(best)

            feature_update_X_nid = []
            for nid in expandable_nids:
                # store the feature to visit for the update of X_nid
                feature_update_X_nid.append(split_record_map[nid].feature)

                # expand the tree structure
                if not np.isnan(split_record_map[nid].threshold):
                    best_split = split_record_map[nid]

                    # create the left and right which have been found
                    # from the parent splits
                    print("1")
                    print(hasattr(best_split, 'expand_record'))
                    1/0
                    #left_sr, right_sr = best_split.expand_record()

                    # the statistics for the children are not computed yet
                    # add a node for left child
                    # find out if the next node will be a lead or not
                    left_nid = self.tree_._add_node_py(
                        parent=nid,
                        is_left=1,
                        is_leaf=TREE_LEAF,
                        feature=FEAT_UNKNOWN,
                        threshold=TREE_UNDEFINED,
                        impurity=left_sr.impurity,
                        n_node_samples=left_sr.c_stats.n_samples,
                        weighted_n_node_samples=left_sr.c_stats.sum_weighted_samples,
                        node_value=(left_sr.c_stats.sum_y /
                                    left_sr.c_stats.sum_weighted_samples))

                    # add a node for the right child
                    right_nid = self.tree_._add_node_py(
                        parent=nid,
                        is_left=0,
                        is_leaf=TREE_LEAF,
                        feature=FEAT_UNKNOWN,
                        threshold=TREE_UNDEFINED,
                        impurity=right_sr.impurity,
                        n_node_samples=right_sr.c_stats.n_samples,
                        weighted_n_node_samples=right_sr.c_stats.sum_weighted_samples,
                        node_value=(right_sr.c_stats.sum_y /
                                    right_sr.c_stats.sum_weighted_samples))

                    # Update the parent node with the found best split
                    self.tree_._update_node_py(
                        node_id=nid,
                        left_child=left_nid,
                        right_child=right_nid,
                        threshold=best_split.threshold,
                        impurity=best_split.impurity,
                        feature=best_split.feature,
                        n_node_samples=best_split.c_stats.n_samples,
                        weighted_n_node_samples=best_split.c_stats.sum_weighted_samples)

                    # update the dictionary with the new record
                    # add only the record if the impurity at the node is large
                    # enough
                    if left_sr.impurity > self.min_impurity_split:
                        parent_split_map.update({left_nid: left_sr})
                    if right_sr.impurity > self.min_impurity_split:
                        parent_split_map.update({right_nid: right_sr})

                    self.counter_X_nid_labels_ = np.zeros(
                        max(parent_split_map.keys()), dtype=int)

                # we can flush the data from the parent_split_map for the
                # current node
                del parent_split_map[nid]

            # update of the expandable nodes
            expandable_nids = list(parent_split_map.keys())

            # remove redundant index of feature to visit when updating X_nid
            feature_update_X_nid = np.unique(feature_update_X_nid)

            # check that some node need to be extended before to update
            # the node index
            # make a copy of X_nid
            X_nid_tmp = X_nid.copy()
            if not expandable_nids:
                # break if we cannot grow anymore
                break
            else:
                for sample_idx in range(X.shape[0]):
                    for feat_i in feature_update_X_nid:
                        # get the index of samples to update
                        X_idx = X_idx_sorted[sample_idx, feat_i]
                        parent_nid = X_nid[X_idx]
                        # only if the sample was not a leaf
                        if parent_nid != -1:
                            if split_record_map[parent_nid].feature == feat_i:
                                # if the feature correspond, we can update the
                                # feature
                                parent_n_leeft_samples = split_record_map[
                                    parent_nid].l_stats.n_samples

                                # no threshold found -> this is a leaf
                                if np.isnan(split_record_map[
                                        parent_nid].threshold):
                                    X_nid_tmp[X_idx] = -1
                                else:
                                    # counter to know how many samples we
                                    # checked per splitter
                                    self.counter_X_nid_labels_[parent_nid] += 1
                                    # track how many samples we send to the
                                    # left child handle the time that several
                                    # samples are equal
                                    if (self.counter_X_nid_labels_[
                                            parent_nid] <=
                                            parent_n_left_samples):
                                        # is it a leaf
                                        if (self.tree_.children_left[
                                                parent_nid] in
                                                expandable_nids):
                                            X_nid_tmp[X_idx] = self.tree_.children_left[
                                                parent_nid]
                                        else:
                                            X_nid_tmp[X_idx] = -1
                                    else:
                                        # is it a leaf
                                        if (self.tree_.children_right[
                                                parent_nid] in
                                                expandable_nids):
                                            X_nid_tmp[X_idx] = self.tree_.children_right[
                                                parent_nid]
                                        else:
                                            X_nid_tmp[X_idx] = -1
                X_nid = X_nid_tmp

            current_depth += 1

        return self

        # Create the parent split record
        # compute the impurity for the parent node
        # FIXME only MSE impurity for the moment
        node_id_p = self.tree_._add_node_py(
                parent=TREE_UNDEFINED,
                is_left=1, is_leaf=TREE_LEAF,
                feature=FEAT_UNKNOWN,
                threshold=TREE_UNDEFINED,
                impurity=impurity,
                n_node_samples=n_samples,
                weighted_n_node_samples=weighted_n_samples,
                node_value=(root_stats.sum_y /
                            root_stats.sum_weighted_samples))