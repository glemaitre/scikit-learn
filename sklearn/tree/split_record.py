from __future__ import division, print_function

from numpy import nan
from numpy import inf

from .stats_node import StatsNode
from .criterion import _impurity_mse


class SplitRecord(object):
    """Container of the parent and children statistics at a node

    Attributes
    ----------
    feature: int,
        The index of the feature to apply the split.

    pos: int,
        The sample index where to apply the split.

    threshold: float,
        The threshold of X[pos, feature] to make the split.

    impurity: float,
        The impurity for that specific split.

    nid: int,
        The id of the current node.

    c_stats: StatsNode,
        Current node statistics.

    l_stats: StatsNode,
        Left node statistics.

    r_stats: StatsNode,
        Right node statistics.
    """

    def __init__(self):
        self.feature = 0
        self.pos = 0
        self.threshold = nan
        self.impurity = inf
        self.nid = 0

        # statistics related to current and children node
        self.c_stats = StatsNode(0., 0., 0, 0.)
        self.l_stats = StatsNode(0., 0., 0, 0.)
        self.r_stats = StatsNode(0., 0., 0, 0.)

    def reset(self, feature, pos, threshold, impurity,
              nid, c_stats, l_stats, r_stats):
        """Reset the split record"""
        self.feature = int(feature)
        self.pos = (pos)
        self.threshold = float(threshold)
        self.impurity = float(impurity)
        self.nid = int(nid)
        self.c_stats = c_stats
        self.l_stats = l_stats
        self.r_stats = r_stats

    def clear(self):
        """Clear the split record"""
        self.feature = 0
        self.pos = 0
        self.threshold = nan
        self.impurity = inf
        self.nid = 0
        self.c_stats.clear()
        self.l_stats.clear()
        self.r_stats.clear()

    def expand_record(self):
        """Create two new records from the left and right stats"""
        # create the left child split record
        left_sr = SplitRecord()
        left_sr.c_stats = self.l_stats
        # FIXME stuck with impurity mse for the moment
        left_sr.impurity = _impurity_mse(left_sr.c_stats)

        # create the right child split record
        right_sr = SplitRecord()
        right_sr.c_stats = self.l_stats
        # FIXME stuck with impurity mse for the moment
        right_sr.impurity = _impurity_mse(right_sr.c_stats)

        return left_sr, right_sr

    def __str__(self):
        info = ("feature: {}\n"
                "position: {}\n"
                "threshold: {}\n"
                "impurity: {}\n"
                "node id: {}\n"
                "current stats: {}\n"
                "left stats: {}\n"
                "right stats: {}\n".format(self.feature,
                                           self.pos,
                                           self.threshold,
                                           self.impurity,
                                           self.nid,
                                           self.c_stats,
                                           self.l_stats,
                                           self.r_stats))
        return info
