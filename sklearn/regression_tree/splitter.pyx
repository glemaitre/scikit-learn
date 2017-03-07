# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


cdef class Splitter(object):
    """New type of splitter driven by data

    Parameters
    ----------
    X: ndarray, shape (n_samples, n_features)
        The dataset to be fitted.

    y: ndarray, shape (n_samples,)
        The residuals.

    sample_weight: ndarray, shape (n_samples,)
        The weight associated to each samples.

    sum_total_weighted_samples: float,
        The sum of the weights for all samples.

    split_record: SplitRecord,
        The split record to associate with this splitter.

    min_samples_leaf: int,
        The minimum number of samples to have a proper split record.

    min_weight_leaf: float,
        The minimum weight to have a proper split record.

    Attributes
    -------
    best_split_record: SplitRecord,
        The best split record found after iterating all the samples.
    """

    def __cinit__(self, np.ndarray[DOUBLE_t, ndim=2] X,
                  np.ndarray[DOUBLE_t, ndim=2] y,
                  np.ndarray[DOUBLE_t, ndim=1] sample_weight,
                  DOUBLE_t sum_total_weighted_samples, SIZE_t feature_idx,
                  SIZE_t start_idx, SplitRecord split_record,
                  SIZE_t min_samples_leaf, DOUBLE_t min_weight_leaf):
        # store the information related to the dataset
        self.X = X
        self.y = y
        self.sample_weight = sample_weight
        self.sum_total_weighted_samples = sum_total_weighted_samples

        # information about the feature and first sampled
        self.feature_idx = feature_idx
        self.start_idx = start_idx
        self.prev_idx = start_idx

        # split record to work with
        self.split_record = SplitRecord()
        self.split_record.copy_from(split_record)
        # Update the feature and position
        self.split_record.feature = self.feature_idx
        self.split_record.pos = self.start_idx

        self.best_split_record = SplitRecord()
        # split to store the best split record
        self.best_split_record.copy_from(split_record)

        # parameters for early stop of split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf

    cpdef reset(self, SIZE_t feature_idx, SIZE_t start_idx,
                    SplitRecord split_record):
        """Reset a splitter with a new samples set.
        It could correspond to a new feature to be scanned.
        """
        # information about the feature and first sampled
        self.feature_idx = feature_idx
        self.start_idx = start_idx
        self.prev_idx = start_idx

        # split record to work with
        self.split_record.copy_from(split_record)
        self.split_record.feature = self.feature_idx
        self.split_record.pos = self.start_idx
        # split to store the best split record
        self.best_split_record.copy_from(split_record)

    cpdef update_stats(self, SIZE_t sample_idx):
        # FIXME This needs a cleaner approach

        # make an update of the statistics
        # collect the statistics to add to the left node

        cdef np.ndarray[DOUBLE_t, ndims=2] y
        cdef np.ndarray[DOUBLE_t, ndims=1] sample_weight

        StatsNode l_stats = self.split_record.l_stats
        StatsNode r_stats = self.split_record.r_stats
        StatsNode c_stats = self.split_record.c_stats

        self.temp = StatsNode(
            sum_y=y[sample_idx] * sample_weight[sample_idx],
            sum_sq_y=(y[sample_idx, 0] ** 2.0 *
                      sample_weight[sample_idx]),
            n_samples=1,
            sum_weighted_samples=sample_weight[sample_idx]))

        # add these statistics to the left child
        self.split_record.l_stats.sum_y += y[sample_idx] * sample_weight[sample_idx]
        self.SplitRecord
        # update the statistics of the right child based on the new l_stats
        self.split_record.r_stats.copy_from(self.split_record.c_stats)
        self.split_record.r_stats.sub(self.split_record.l_stats)

    cpdef node_evaluate_split(self, SIZE_t sample_idx):
        """Update the impurity and check the corresponding split should be
        kept.
        """
        cdef DOUBLE_t c_impurity_improvement
        cdef SIZE_t feat_i = self.feature_idx

        cdef np.ndarray[DOUBLE_t, ndims=2] X

        # check that the two consecutive samples are not the same
        cdef bint b_samples_var =  (fabs(X[sample_idx, feat_i] -
                                         X[self.prev_idx, feat_i]) >
                                    FEATURE_THRESHOLD)

        # check that there is enough samples to make a split
        cdef bint b_n_samples = not (
            self.split_record.l_stats.n_samples <
            self.min_samples_leaf or
            (self.split_record.r_stats.n_samples <
             self.min_samples_leaf))

        # check that the weights corresponding to samples is great enough
        cdef bint b_weight_samples = not(
            self.split_record.l_stats.sum_weighted_samples <
            self.min_weight_leaf or
            self.split_record.r_stats.sum_weighted_samples <
            self.min_weight_leaf)

        # try to split if necessary
        if b_samples_var and b_n_samples and b_weight_samples:

            # compute the impurity improvement
            # FIXME we use the mse impurity for the moment
            c_impurity_improvement = impurity_improvement(
                self.split_record,
                self.sum_total_weighted_samples)

            # check the impurity improved
            if (c_impurity_improvement >
                    self.best_split_record.impurity_improvement):
                # update the best split
                self.best_split_record.reset(
                    feature=feat_i,
                    pos=self.prev_idx,
                    threshold=((X[sample_idx, feat_i] +
                                X[self.prev_idx, feat_i]) / 2.),
                    impurity=self.split_record.impurity,
                    impurity_improvement=c_impurity_improvement,
                    nid=self.split_record.nid,
                    c_stats=self.split_record.c_stats,
                    l_stats=self.split_record.l_stats,
                    r_stats=self.split_record.r_stats)

        self.update_stats(sample_idx)
        self.prev_idx = sample_idx