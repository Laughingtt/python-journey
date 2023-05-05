import pandas
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin


class CorrelationThreshold(SelectorMixin, BaseEstimator):
    """
        method : {'pearson', 'kendall', 'spearman'} or callable
            Method of correlation:

            * pearson : standard correlation coefficient
            * kendall : Kendall Tau correlation coefficient
            * spearman : Spearman rank correlation
            * callable: callable with input two 1d ndarrays
                and returning a float. Note that the returned matrix from corr
                will have 1 along the diagonals and will be symmetric
                regardless of the callable's behavior.

        >>> X = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]
        >>> selector = CorrelationThreshold()
        >>> selector.fit(X)
        >>> selector.transform(X)


        array([[2, 0],
               [1, 4],
               [1, 1]])
    """

    def __init__(self, threshold=0.8, method="pearson"):
        self.threshold = threshold
        self.method = method
        self._not_support_m = None
        self._support_m = None

    def fit(self, X, y=None):
        """Learn empirical variances from X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data from which to compute variances, where `n_samples` is
            the number of samples and `n_features` is the number of features.

        y : any, default=None
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if len(X) <= 0:
            raise ValueError
        if isinstance(X, pd.DataFrame):
            X = X
        else:
            X = pd.DataFrame(X)

        corr = X.corr(self.method)

        self._not_support_m = (corr.mask(np.eye(len(corr), dtype=bool)).abs() > self.threshold).any()
        self._support_m = (~self._not_support_m).to_numpy()

        return self

    def get_support(self, indices=False):
        return self._support_m

    def transform(self, X):
        if isinstance(X, np.ndarray):
            _X = X[:, self._support_m]
        elif isinstance(X, pandas.DataFrame):
            _X = X.to_numpy()[:, self._support_m]
        else:
            _X = np.array(X)[:, self._support_m]
        return _X

    def _more_tags(self):
        return {"allow_nan": True}

    def _get_support_mask(self):
        pass
