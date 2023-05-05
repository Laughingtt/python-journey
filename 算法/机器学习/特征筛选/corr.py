def corr_func():
    from sklearn.feature_selection import SelectKBest
    from scipy.stats import pearsonr
    import pandas as pd
    import numpy as np

    df = pd.read_csv("../data/my_data_guest.csv")
    X = df.drop(columns=["id", "bad"])
    y = df['bad']

    # 选择与目标变量最相关的k个特征
    # selector = SelectKBest(score_func=pearsonr, k=4)
    # selector.fit(X, y)
    # selected_features = selector.transform(X)

    corr = X.corr()
    m = ~(corr.mask(np.eye(len(corr), dtype=bool)).abs() > 0.8).any()
    raw = corr.loc[m, m]

    print(raw)


def corr_select():
    import correlation_select
    import pandas as pd

    transfrom = correlation_select.CorrelationThreshold(threshold=0.5)
    df = pd.read_csv("../data/my_data_guest.csv")
    X = df.drop(columns=["id", "bad"])
    print(X.shape)

    X = X.to_numpy()
    transfrom.fit(X)

    X_ = transfrom.transform(X)
    print(X_.shape)


if __name__ == '__main__':
    corr_select()
