import copy
import os
import platform
import sys
import warnings
from typing import List, Union, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from lightgbm import LGBMClassifier, LGBMRegressor
from matplotlib.pyplot import MultipleLocator
from pandas import DataFrame
from scipy.stats import ttest_1samp
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, AdaBoostClassifier, \
    AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.linear_model import LassoCV, LogisticRegression, ElasticNetCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, mean_squared_error
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.random_projection import SparseRandomProjection
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor



def init_CN():
    warnings.filterwarnings("ignore")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['figure.figsize'] = (10.0, 8.0)
    # plt.rcParams['font.size'] = 15
    plt.rcParams[
        'font.sans-serif'] = 'Microsoft YaHei' if 'windows' in platform.platform().lower() else 'Arial Unicode MS'
    plt.rcParams['axes.unicode_minus'] = False


def read_data(filename: str):
    assert os.path.exists(filename), f'{filename} 不存在！'
    if filename.endswith('.csv'):
        return pd.read_csv(filename, header=0)
    elif filename.endswith('.xlsx'):
        pd.read_excel(filename, header=0)
    else:
        raise ValueError(f'文件类型未定义！')


def compress_feature(features: np.array, dim: int) -> np.array:
    assert len(features.shape) == 2, '特征的维度必须为2维'
    if dim > features.shape[1]:
        logger.warning(f"降维的维度（{dim}）不能多于样本数({features.shape[0]}), 使用{features.shape[1]}作为降维维度！")
        dim = features.shape[1]
    pca = PCA(dim)
    features = pca.fit_transform(features)
    return features


def compress_df_feature(features: pd.DataFrame, dim: int, not_compress: Union[str, List[str]] = None,
                        prefix='') -> pd.DataFrame:
    """
    压缩深度学习特征
    Args:
        features: 特征DataFrame
        dim: 需要压缩到的维度，此值需要小于样本数
        not_compress: 不进行压缩的列。
        prefix: 所有特征的前缀。

    Returns:

    """
    if not_compress is not None:
        if isinstance(not_compress, str):
            not_compress = [not_compress]
        elif not isinstance(not_compress, Iterable):
            raise ValueError(f"not_compress设置出错！")
        not_compress_data = features[not_compress]
        features = features[[c for c in features.columns if c not in not_compress]]
        features = compress_feature(features, dim=dim)
        features = pd.DataFrame(features, columns=[f"{prefix}{i}" for i in range(features.shape[1])])
        return pd.concat([not_compress_data, features], axis=1)
    else:
        features = compress_feature(features, dim=dim)
        features = pd.DataFrame(features, columns=[f"{prefix}{i}" for i in range(features.shape[1])])
        return features


def split_dataset(X_data, y_data, test_size=0.2, random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, random_state=random_state, test_size=test_size)
    return X_train, X_test, y_train, y_test


def cluster_analysis(features, n_clusters=2):
    cluster = KMeans(n_clusters=n_clusters, random_state=0)
    cluster.fit(features)
    return cluster


def normalize_df(data: pd.DataFrame, not_norm: Union[str, List[str]] = None, method='gaussian', group: str = None) -> \
        pd.DataFrame:
    """
    Normalize data frame
    Args:
        data: DataFrame to be Normalized.
        not_norm: Columns not be normalized.
        method: method to be used, choice: gaussian, min_max, default gaussian.
        group: Normalize by group separately, default None treated as a whole.

    Returns: Normalized Dataframe

    """
    if not_norm is None:
        not_norm = []
    elif isinstance(not_norm, str):
        not_norm = [not_norm]
    if group is not None:
        not_norm = not_norm + [group]
        assert group in data.columns, f"group: {group}分组没有在{data.columns}!"
    columns = [c for c in data.columns if c not in not_norm]
    new_data = data.copy(deep=True)

    def _norm_data(data_):
        desc = data_.describe()
        for column in columns:
            try:
                if method == 'gaussian':
                    data_[column] = (data_[column] - desc[column]['mean']) / desc[column]['std']
                else:
                    data_[column] = (data_[column] - desc[column]['min']) / (desc[column]['max'] - desc[column]['min'])
            except Exception as e:
                logger.warning(f"特征：{column}存在问题：{e}，造成z-score失败！")
        return data_

    if group is None:
        _norm_data(new_data)
    else:
        ugroups = np.unique(data[group])
        ug_list = []
        for ug in ugroups:
            ug_data = new_data[new_data['group'] == ug]
            _norm_data(ug_data)
            ug_list.append(ug_data)
        new_data = pd.concat(ug_list, axis=0)
    return new_data


def convert2onehot(data, n_classes):
    data = np.reshape(data, -1)
    onehot_encoder = []
    for d in data:
        onehot = [0] * n_classes
        onehot[d] = 1
        onehot_encoder.append(onehot)
    return np.array(onehot_encoder)


def draw_roc_per_class(y_test, y_score, n_classes, title='ROC per Class', include_spec_class: bool = True,
                       mapping=None):
    """

    Args:
        mapping: label的映射
        y_test: 真实标签
        y_score: 预测标签
        n_classes: 类别数
        title: 标题
        include_spec_class: 是否包括每个细分标签的ROC曲线。

    Returns:

    """
    if mapping is None:
        mapping = {}
    y_test_binary = convert2onehot(y_test, n_classes=n_classes)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # Compute micro-average ROC curve and ROC area
    # print(y_test_binary.ravel().shape, y_score.ravel().shape)
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binary.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    lw = 2
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    try:
        for i in range(n_classes):
            try:
                fpr[i], tpr[i], _ = roc_curve(y_test_binary[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            except Exception as e:
                logger.error(f'解析{i}类别出错, {e}')
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label="macro-average ROC (area = {0:0.2f})".format(roc_auc["macro"]),
            color="navy",
            linestyle=":",
            linewidth=4,
        )
        if include_spec_class:
            for i in range(n_classes):
                plt.plot(
                    fpr[i],
                    tpr[i],
                    lw=lw,
                    label="class {0} ROC (area = {1:0.2f})".format(mapping[i] if i in mapping else i, roc_auc[i]),
                )
    except:
        logger.error(f'解析每个类别的ROC出错，大概率是应为数据没有指定类别的样本！')
    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title(title)
    plt.legend(loc="lower right")


def draw_roc(y_test, y_score, title='ROC', labels=None):
    """
    绘制ROC曲线
    Args:
        y_test: list或者array，为真实结果。
        y_score: list orray或者array，为模型预测结果。
        title: 图标题
        labels: 图例名称

    Returns:

    """
    if not isinstance(y_test, (list, tuple)):
        y_test = [y_test]
    if not isinstance(y_score, (list, tuple)):
        y_score = [y_score]
    if labels is None:
        labels = [''] * len(y_score)
    assert len(y_test) == len(y_score) == len(labels)
    colors = ["deeppink", "navy", "aqua", "darkorange", "cornflowerblue"]
    ls = ['-', ':', '--', ':']
    for idx, (y_test_, y_score_, label) in enumerate(zip(y_test, y_score, labels)):
        y_score_ = np.array(y_score_)
        # enc = OneHotEncoder(handle_unknown='ignore')
        # y_test_binary = enc.fit_transform(y_test_.reshape(-1, 1)).toarray()
        if len(y_score_.shape) == 1:
            y_score_1 = y_score_
        else:
            y_score_1 = y_score_[:, 1]
        fpr, tpr, _ = roc_curve(y_test_, y_score_1)
        # print(y_test_, y_score_1)
        from delong import calc_95_CI
        auc_, ci = calc_95_CI(np.squeeze(y_test_), y_score_1)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{label} AUC: {roc_auc:0.3f} (95%CI {ci[0]:.3f}-{ci[1]:.3f})",
                 color=colors[idx % len(colors)], linestyle=ls[idx % len(ls)], linewidth=4)

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title(title)
    plt.legend(loc="lower right")


def draw_calibration(y_test, pred_scores, model_names, **kwargs):
    """

    Args:
        y_test:
        pred_scores:
        model_names:
        **kwargs:

    Returns:

    """
    version_info = sys.version_info
    assert version_info.major >= 3 and version_info.minor > 6, "Python版本必须3.7及以上。"
    if not isinstance(y_test, (tuple, list)):
        y_test = [y_test] * len(model_names)
    assert len(pred_scores) == len(model_names)
    from sklearn.calibration import CalibrationDisplay
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 1)
    fig = plt.figure(figsize=(10, 10))
    ax_calibration_curve = fig.add_subplot(gs[0, 0])

    def moving_average(interval, windowsize):
        window = np.ones(int(windowsize)) / float(windowsize)
        return np.convolve(interval, window, 'full')

    for model_name, scores, gt in zip(model_names, pred_scores, y_test):
        if kwargs.get('smooth', False):
            pred = scipy.signal.savgol_filter(scores[:, 1], kwargs.get('window_length', 9), kwargs.get('k', 3))
            pred = np.clip(pred, 0, 1)
        else:
            pred = scores[:, 1]
        disp = CalibrationDisplay.from_predictions(gt, pred,
                                                   n_bins=kwargs.get('n_bins', 5),
                                                   ax=ax_calibration_curve, name=model_name)


def get_bst_split(X_data: pd.DataFrame, y_data: pd.DataFrame,
                  models: dict, test_size=0.2, metric_fn=accuracy_score, n_trails=10,
                  cv: bool = False, shuffle: bool = False, metric_cut_off: float = None, random_state=None,
                  use_smote: bool = False, **kwargs):
    """
    寻找数据集中最好的数据划分。
    Args:
        X_data: 训练数据
        y_data: 监督数据
        models: 模型名称，Dict类型、
        test_size: 测试集比例
        metric_fn: 评价模型好坏的函数，默认准确率，可选roc_auc_score。
        n_trails: 尝试多少次寻找最佳数据集划分。
        cv: 是否是交叉验证，默认是False，当为True时，n_trails为交叉验证的n_fold
        shuffle: 是否进行随机打乱
        metric_cut_off: 当metric_fn的值达到多少时进行截断。
        random_state: 随机种子
        use_smote: bool, 是否使用SMOTE技术，进行重采样。
        kwargs: 其他模型训练的参数。

    Returns: {'max_idx': max_idx, "max_model": max_model, "max_metric": max_metric, "results": results}

    """
    assert metric_fn in (roc_auc_score, accuracy_score, mean_squared_error)
    results = []
    max_model = None
    max_model_name = None
    max_idx = 0
    max_metric = None
    metrics = {}
    dataset = []
    if not isinstance(X_data, pd.DataFrame) or not isinstance(y_data, pd.DataFrame):
        X_data = pd.DataFrame(X_data)
        y_data = pd.DataFrame(y_data)
        logger.warning('你的数据不是DataFrame类型，可能遇到未知错误！')
    if cv:
        skf = StratifiedKFold(n_splits=n_trails, shuffle=shuffle or random_state is not None, random_state=random_state)
        for train_index, test_index in skf.split(X_data, y_data):
            X_train, X_test = X_data.loc[train_index], X_data.loc[test_index]
            y_train, y_test = y_data.loc[train_index], y_data.loc[test_index]
            dataset.append([X_train, X_test, y_train, y_test])
    for idx in range(n_trails):
        trail = []
        if cv:
            X_train, X_test, y_train, y_test = dataset[idx]
        else:
            rs = None if random_state is None else (idx + random_state)
            X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_size, random_state=rs)
        X_train_smote, y_train_smote = X_train, y_train
        if use_smote:
            X_train_smote, y_train_smote = smote_resample(X_train, y_train)
        copy_models = copy.deepcopy(models)
        for model_name, model in copy_models.items():
            # model.fit(X_train, y_train)
            # sample_weight = [1 if i == 0 else 0.5 for i in list(np.array(y_train))]
            if kwargs:
                try:
                    model.fit(X_train_smote, y_train_smote, **kwargs)
                    logger.info(f'正在训练{model_name}, 使用{kwargs}。')
                except Exception as e:
                    model.fit(X_train_smote, y_train_smote)
                    logger.warning(f'因为：{e}，训练{model_name}使用{kwargs}失败。')
            else:
                model.fit(X_train_smote, y_train_smote)
            y_pred = model.predict(X_test)
            if metric_fn == roc_auc_score:
                y_proba = model.predict_proba(X_test)[:, 1]
                metric = metric_fn(y_test, y_proba)
            else:
                metric = metric_fn(y_test, y_pred)
            if model_name not in metrics:
                metrics[model_name] = []
            metrics[model_name].append((idx, metric))
            if max_metric is None or metric > max_metric:
                max_metric = metric
                max_idx = idx
                max_model = model
                max_model_name = model_name
            trail.append(metric)
        results.append((trail, (X_train, X_test, y_train, y_test)))
        # 当满足用户需求的时候，可以停止。
        if metric_cut_off is not None and max_metric is not None and max_metric > metric_cut_off:
            logger.info(f'Get best split cut off on {idx + 1} trails!')
            break
    return {'max_idx': max_idx, "max_model": max_model, "max_metric": max_metric, 'max_model_name': max_model_name,
            "results": results, 'metrics': metrics}


def get_cv_metric_binary_task(models, data_splits, labels, model_names=None):
    """
    生成CV结果，只对二分类模型有效。
    Args:
        models: 需要使用的模型
        data_splits: 数据划分
        labels: 数据对应任务的label
        model_names: 模型名称

    Returns: 所有交叉验证的结果。
    """
    metric = []
    for cv_index, (X_train_sel, X_test_sel, y_train_sel, y_test_sel) in enumerate(data_splits):
        predictions = [[(model.predict(X_train_sel), model.predict(X_test_sel))
                        for model in target] for label, target in zip(labels, models)]
        pred_scores = [[(model.predict_proba(X_train_sel), model.predict_proba(X_test_sel))
                        for model in target] for label, target in zip(labels, models)]

        pred_sel_idx = []
        for model, label, prediction, scores in zip(models, labels, predictions, pred_scores):
            pred_sel_idx_label = []
            if model_names is None:
                model_names = [str(m.__class__) for m in model]
            assert len(prediction) == len(model_names), "模型名称必须与模型长度相同"
            for mname, (train_pred, test_pred), (train_score, test_score) in zip(model_names, prediction, scores):
                # 计算训练集指数
                acc, auc, ci, tpr, tnr, ppv, npv, precision, recall, f1, thres = analysis_pred_binary(
                    y_train_sel[label],
                    train_score[:, 1])
                ci = f"{ci[0]:.4f} - {ci[1]:.4f}"
                metric.append((mname, acc, auc, ci, tpr, tnr, ppv, npv, precision, recall, f1, thres,
                               f"CV-{cv_index}-{label}-train"))

                # 计算验证集指标
                acc, auc, ci, tpr, tnr, ppv, npv, precision, recall, f1, thres = analysis_pred_binary(y_test_sel[label],
                                                                                                      test_score[:, 1])
                ci = f"{ci[0]:.4f} - {ci[1]:.4f}"
                metric.append((mname, acc, auc, ci, tpr, tnr, ppv, npv, precision, recall, f1, thres,
                               f"CV-{cv_index}-{label}-test"))
                # 计算thres对应的sel idx
                pred_sel_idx_label.append(np.logical_or(test_score[:, 0] >= thres, test_score[:, 1] >= thres))

            pred_sel_idx.append(pred_sel_idx_label)
    metric = pd.DataFrame(metric, index=None, columns=['model_name', 'Accuracy', 'AUC', '95% CI',
                                                       'Sensitivity', 'Specificity',
                                                       'PPV', 'NPV', 'Precision', 'Recall', 'F1',
                                                       'Threshold', 'Task'])
    return metric


def create_clf_model(model_names):
    models = {}
    # 判断是纯字符串，使用默认参数进行配置
    if isinstance(model_names, (list, tuple)):
        if 'lr' in model_names or 'LR' in model_names:
            models['LR'] = LogisticRegression(random_state=0)
        # NB
        if 'nb' in model_names or 'NaiveBayes' in model_names:
            models['NaiveBayes'] = GaussianNB()
        # SVM
        if 'svm' in model_names or 'SVM' in model_names:
            models['SVM'] = SVC(probability=True, random_state=0)
        # KNN
        if 'knn' in model_names or 'KNN' in model_names:
            models['KNN'] = KNeighborsClassifier(algorithm='kd_tree')
        # DecisionTree
        if 'dt' in model_names or 'DecisionTree' in model_names:
            models['DecisionTree'] = DecisionTreeClassifier(max_depth=None,
                                                            min_samples_split=2, random_state=0)
        # RandomForest
        if 'rf' in model_names or 'RandomForest' in model_names:
            models['RandomForest'] = RandomForestClassifier(n_estimators=10, max_depth=None,
                                                            min_samples_split=2, random_state=0)
        # ExtraTree
        if 'et' in model_names or 'ExtraTrees' in model_names:
            models['ExtraTrees'] = ExtraTreesClassifier(n_estimators=10, max_depth=None,
                                                        min_samples_split=2, random_state=0)
        # XGBoost
        if 'xgb' in model_names or 'XGBoost' in model_names:
            models['XGBoost'] = XGBClassifier(n_estimators=10, objective='binary:logistic',
                                              use_label_encoder=False, eval_metric='error')
        # LightGBM
        if 'lgb' in model_names or 'LightGBM' in model_names:
            models['LightGBM'] = LGBMClassifier(n_estimators=10, max_depth=-1, objective='binary')

        # GBM
        if 'gbm' in model_names or 'GradientBoosting' in model_names:
            models['GradientBoosting'] = GradientBoostingClassifier(n_estimators=10, random_state=0)
        # AdaBoost
        if 'adaboost' in model_names or 'AdaBoost' in model_names:
            models['AdaBoost'] = AdaBoostClassifier(n_estimators=10, random_state=0)

        # Multi layer perception
        if 'mlp' in model_names or 'MLP' in model_names:
            models['MLP'] = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300, solver='sgd', random_state=0)
    elif isinstance(model_names, dict):
        for model_name, params in model_names.items():
            if 'svm' == model_name or 'SVM' == model_name:
                models['SVM'] = SVC(**params)
            # KNN
            if 'knn' == model_name or 'KNN' == model_name:
                models['KNN'] = KNeighborsClassifier(**params)
            # NB
            if 'nb' == model_name or 'NaiveBayes' == model_name:
                models['NaiveBayes'] = GaussianNB(**params)
            # DecisionTree
            if 'dt' == model_name or 'DecisionTree' == model_name:
                models['DecisionTree'] = DecisionTreeClassifier(**params)
            # RandomForest
            if 'rf' == model_name or 'RandomForest' == model_name:
                models['RandomForest'] = RandomForestClassifier(**params)
            # ExtraTree
            if 'et' == model_name or 'ExtraTrees' == model_name:
                models['ExtraTrees'] = ExtraTreesClassifier(**params)
            # XGBoost
            if 'xgb' == model_name or 'XGBoost' == model_name:
                models['XGBoost'] = XGBClassifier(**params)
            # LightGBM
            if 'lgb' == model_name or 'LightGBM' == model_name:
                models['LightGBM'] = LGBMClassifier(**params)
            # GBM
            if 'gbm' == model_name or 'GradientBoosting' == model_name:
                models['GradientBoosting'] = GradientBoostingClassifier(**params)

            # AdaBoost
            if 'adaboost' == model_name or 'AdaBoost' == model_name:
                models['AdaBoost'] = AdaBoostClassifier(**params)

            # Multi layer perception
            if 'mlp' == model_name or 'MLP' == model_name:
                models['MLP'] = MLPClassifier(**params)
    return models


def create_reg_model(model_names):
    models = {}
    # 判断是纯字符串，使用默认参数进行配置
    if isinstance(model_names, (list, tuple)):
        if 'svm' in model_names or 'SVM' in model_names:
            models['SVM'] = SVR()
        # KNN
        if 'knn' in model_names or 'KNN' in model_names:
            models['KNN'] = KNeighborsRegressor(algorithm='kd_tree')
        # DecisionTree
        if 'dt' in model_names or 'DecisionTree' in model_names:
            models['DecisionTree'] = DecisionTreeRegressor(max_depth=None,
                                                           min_samples_split=2, random_state=0)
        # RandomForest
        if 'rf' in model_names or 'RandomForest' in model_names:
            models['RandomForest'] = RandomForestRegressor(n_estimators=10, max_depth=None,
                                                           min_samples_split=2, random_state=0)
        # ExtraTree
        if 'et' in model_names or 'ExtraTrees' in model_names:
            models['ExtraTrees'] = ExtraTreesRegressor(n_estimators=10, max_depth=None,
                                                       min_samples_split=2, random_state=0)
        # XGBoost
        if 'xgb' in model_names or 'XGBoost' in model_names:
            models['XGBoost'] = XGBRegressor(n_estimators=10, max_depth=5, objective='reg:squarederror',
                                             eval_metric='rmse')
        # LightGBM
        if 'lgb' in model_names or 'LightGBM' in model_names:
            models['LightGBM'] = LGBMRegressor(n_estimators=10, max_depth=4, objective='regression')

        # GBM
        if 'gbm' in model_names or 'GradientBoosting' in model_names:
            models['GradientBoosting'] = GradientBoostingRegressor(n_estimators=10, random_state=0)

        # AdaBoost
        if 'adaboost' in model_names or 'AdaBoost' in model_names:
            models['AdaBoost'] = AdaBoostRegressor(n_estimators=10, random_state=0)

        # Multi layer perception
        if 'mlp' in model_names or 'MLP' in model_names:
            models['MLP'] = MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=300, solver='sgd')
    elif isinstance(model_names, dict):
        for model_name, params in model_names.items():
            if 'svm' == model_name or 'SVM' == model_name:
                models['SVM'] = SVR(**params)
            # KNN
            if 'knn' == model_name or 'KNN' == model_name:
                models['KNN'] = KNeighborsRegressor(**params)
            # DecisionTree
            if 'dt' == model_name or 'DecisionTree' == model_name:
                models['DecisionTree'] = DecisionTreeRegressor(**params)
            # RandomForest
            if 'rf' == model_name or 'RandomForest' == model_name:
                models['RandomForest'] = RandomForestRegressor(**params)
            # ExtraTree
            if 'et' == model_name or 'ExtraTrees' == model_name:
                models['ExtraTrees'] = ExtraTreesRegressor(**params)
            # XGBoost
            if 'xgb' == model_name or 'XGBoost' == model_name:
                models['XGBoost'] = XGBRegressor(**params)
            # LightGBM
            if 'lgb' == model_name or 'LightGBM' == model_name:
                models['LightGBM'] = LGBMRegressor(**params)
            # GBM
            if 'gbm' == model_name or 'GradientBoosting' == model_name:
                models['GradientBoosting'] = GradientBoostingRegressor(**params)
            # AdaBoost
            if 'adaboost' == model_name or 'AdaBoost' in model_name:
                models['AdaBoost'] = AdaBoostRegressor(**params)
            # Multi layer perception
            if 'mlp' == model_name or 'MLP' == model_name:
                models['MLP'] = MLPRegressor(**params)
    return models


def calc_confusion_matrix(prediction: List[int], gt: List[int], sel_idx: Union[List[bool], np.ndarray] = None,
                          class_mapping: Union[str, dict, list, tuple] = None, num_classes: int = None):
    """

    Args:
        prediction: Prediction of each results.
        gt: Ground truth of each results.
        sel_idx: Use which index of data to calculate cm. default None for all.
        class_mapping: mapping class index to readable classes.
        num_classes: Number of classes.

    Returns:

    """
    num_classes = num_classes or len(set(gt))
    if num_classes != len(set(gt)):
        logger.warning(f'num_classes({num_classes}) is not equal to labels in gt({len(set(gt))}).')
    cm = np.zeros((num_classes, num_classes))
    if sel_idx is not None:
        logger.info(f"使用筛选阈值的数据绘制混淆矩阵！样本量从{len(gt)}变到{sum(sel_idx)}.")
        prediction = prediction[sel_idx]
        gt = gt[sel_idx]
    for pred, y in zip(prediction, gt):
        # cm[int(pred), int(y)] += 1
        cm[int(y), int(pred)] += 1

    mapping = {}
    if isinstance(class_mapping, dict):
        mapping = class_mapping
    elif isinstance(class_mapping, (list, tuple)):
        mapping = dict(enumerate(class_mapping))
    elif class_mapping and os.path.exists(class_mapping):
        label_names = [l_.strip() for l_ in open(class_mapping).readlines()]
        mapping = {i: l for i, l in enumerate(label_names)}
    labels = [mapping[i] if i in mapping else f"label_{i}" for i in range(num_classes)]
    return pd.DataFrame(cm, index=labels, columns=labels)


def draw_matrix(data: pd.DataFrame, norm: bool = False, **kwargs):
    if norm:
        data = data.div(np.sum(data, axis=1), axis=0)
    if 'fmt' not in kwargs:
        kwargs['fmt'] = ".2g" if norm else ".3f"
    sns.heatmap(data, **kwargs)


def draw_predict_score(pred_score, y_test):
    d = pd.concat([pd.DataFrame(pred_score[:, 1]), y_test.reset_index()], axis=1)
    # plt.axis('off')
    d.columns = ['predict_score', 'index', 'label']
    d['predict_score'] = (d['predict_score'] - 0.5) * 2
    d = d.sort_values('predict_score')
    d['index'] = range(d.shape[0])
    plt.bar(d[d['label'] == 0]['index'], d[d['label'] == 0]["predict_score"])
    plt.bar(d[d['label'] == 1]['index'], d[d['label'] == 1]["predict_score"])


def draw_cv_box(data, x, y, hue=None, **kwargs):
    if hue:
        sns.boxplot(data=data, x=x, y=y, hue=hue, **kwargs)
    else:
        sns.boxplot(data=data, x=x, y=y, **kwargs)


def analysis_features(rad_features, labels, not_use: Union[List[str], str] = ['ID', 'label'],
                      methods: Union[List[str], str] = 't-SNE', n_neighbors=30, save_dir=None):
    """

    Args:
        rad_features: 待分析的特征
        labels: 每个样本对饮的标签
        not_use: 那些列不使用
        methods: 选择可视化的方法, Random projection, Truncated SVD, Isomap, Standard LLE, Modified LLE,
                MDS, Random Trees, Spectral, t-SNE, NCA
        n_neighbors:
        save_dir: 保存目录

    Returns:

    """
    if not isinstance(not_use, (list, tuple)):
        not_use = [not_use]
    if methods is not None and not isinstance(methods, (list, tuple)):
        methods = [methods]
    data = np.array(rad_features[[c for c in rad_features.columns if c not in not_use]])
    embeddings = {
        "Random projection": SparseRandomProjection(n_components=2, random_state=42),
        "Truncated SVD": TruncatedSVD(n_components=2),
        "Isomap": manifold.Isomap(n_neighbors=n_neighbors, n_components=2),
        "Standard LLE": manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2, method="standard"),
        "Modified LLE": manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2, method="modified"),
        # "Hessian LLE": manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2, method="hessian"),
        # "LTSA LLE": manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2, method="ltsa"),
        "MDS": manifold.MDS(n_components=2, n_init=1, max_iter=120, n_jobs=2),
        "Random Trees": make_pipeline(RandomTreesEmbedding(n_estimators=200, max_depth=5, random_state=0),
                                      TruncatedSVD(n_components=2), ),
        "Spectral": manifold.SpectralEmbedding(n_components=2, random_state=0, eigen_solver="arpack"),
        "t-SNE": manifold.TSNE(n_components=2, init="pca", learning_rate="auto", n_iter=500,
                               n_iter_without_progress=150, n_jobs=2, random_state=0, ),
        "NCA": NeighborhoodComponentsAnalysis(n_components=2, init="pca", random_state=0),
    }
    projections = {}
    for name, transformer in embeddings.items():
        if methods is None or name in methods:
            try:
                projections[name] = transformer.fit_transform(data, labels)
            except Exception as e:
                logger.error(f"使用{name}进行降维可视化过程中出错。{e}")
    for name in projections:
        X = MinMaxScaler().fit_transform(projections[name])
        color_list = sns.color_palette("hls", len(np.unique(labels)))
        for idx, label in enumerate(np.unique(labels)):
            plt.scatter(*X[labels == label].T, s=60, alpha=0.425, zorder=2, color=color_list[idx])
        plt.legend(labels=np.unique(labels), loc="lower right")
        plt.title(f'Method: {name}')
        if save_dir is not None:
            try:
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, f'feature_{name}_viz.svg'), bbox_inches='tight')
            except Exception as e:
                logger.error(f'保存{name}可视化功能失效，因为：「{e}」')
        plt.show()


def select_feature(corr, threshold: float = 0.9, keep: int = 1, verbose: bool = False, topn: int = 1):
    feature_names = corr.columns
    drop_feature_names = [x for x, y in np.array(corr.isna().all().reset_index()) if y]
    has_corr_features = True
    while has_corr_features:
        has_corr_features = False
        corr_fname = {}
        feature2drop = [fname for fname in feature_names if fname not in drop_feature_names]
        for i, fi in enumerate(feature2drop):
            corr_num = 0
            for j in range(i + 1, len(feature2drop)):
                if abs(corr[fi][feature2drop[j]]) > threshold:
                    corr_num += 1
            corr_fname[fi] = corr_num
        corr_fname = sorted(corr_fname.items(), key=lambda x: x[1], reverse=True)
        for fname, corr_num in corr_fname[:topn]:
            if corr_num >= keep:
                has_corr_features = True
                drop_feature_names.append(fname)
                if verbose:
                    logger.info(f'len {len(feature2drop)}, {fname} has {corr_num} features')

    return [fname for fname in feature_names if fname not in drop_feature_names]


def select_feature_mrmr(data: pd.DataFrame, method='MID', num_features: Union[int, float] = 0.1,
                        label_column: str = 'label'):
    """

    Args:
        data: 数据，DataFrame
        method: mrmr的方法，支持MID、MIQ
        num_features: 筛选的特征数量，如果是小数则为百分比，如果是整数则为个数
        label_column: 目标列名，默认为label

    Returns:

    """
    import pymrmr
    if isinstance(num_features, float):
        num_features = max(1, int(data.shape[1] * num_features))
    ordered_column_name = [label_column] + [c for c in data.columns if c != label_column and data[c].dtype != object]
    return pymrmr.mRMR(data[ordered_column_name], method, num_features + 1)


def select_feature_ttest(data: pd.DataFrame, popmean: Union[float, np.ndarray], threshold: float = 0.05):
    feature_names = data.columns
    if not isinstance(popmean, (list, tuple)):
        popmean = [popmean] * len(feature_names)
    elif len(popmean) != len(feature_names):
        raise ValueError('mean is not equal to feature length!')
    res = ttest_1samp(data, popmean)
    return [fname for fname, flag in zip(feature_names, res.pvalue < threshold) if flag]


def lasso_cv_coefs(X_data, y_data, alpha_logmin=-3, points=50, column_names: List[str] = None, cv: int = 10,
                   ensure_lastn: int = None, model_name: str = 'lasso', force_alpha=None, **kwargs):
    """

    Args:
        X_data: 训练数据
        y_data: 监督数据
        alpha_logmin: alpha的log最小值
        points: 打印多少个点。默认50
        column_names: 列名，默认为None，当选择的数据很多的时候，建议不要添加此参数
        cv: 交叉验证次数。
        ensure_lastn: bool, 确保是最后一个MSE不降的alpha
        model_name: 可选使用Lasso还是ElasticNet。
        force_alpha: deprecated!
        **kwargs: 其他用于打印控制的参数。

    """
    # 每个特征值随lambda的变化
    alphas = np.logspace(alpha_logmin, 0, points)
    if points != 50 or cv != 10:
        print(f"Points: {points}, CV: {cv}")
    if model_name.lower() == 'lasso':
        lasso_cv = LassoCV(alphas=alphas, cv=cv, n_jobs=-1, random_state=0).fit(X_data, y_data)
    else:
        lasso_cv = ElasticNetCV(alphas=alphas, cv=cv, n_jobs=-1, random_state=0).fit(X_data, y_data)
    _, coefs, _ = lasso_cv.path(X_data, y_data, alphas=alphas)
    coefs = np.squeeze(coefs).T

    MSEs = lasso_cv.mse_path_
    MSEs_mean = np.mean(MSEs, axis=1)
    # plt.rcParams['font.sans-serif'] = 'stixgeneral'
    plt.semilogx(lasso_cv.alphas_, coefs, '-', **kwargs)
    if column_names is not None:
        # print(column_names, coefs.shape[1])
        assert len(column_names) == coefs.shape[1]
        plt.legend(labels=column_names, loc='best')
    if ensure_lastn is not None and lasso_cv.alpha_ == 1.0:
        lasso_cv.alpha_ = alphas[-ensure_lastn]
        logger.warning(f'你正在使用ensure_lastn={ensure_lastn}...')
    lambda_info = ''
    if force_alpha is not None:
        lasso_cv.alpha_ = force_alpha
    if lasso_cv.alpha_ != 1.0:
        plt.axvline(lasso_cv.alpha_, color='black', ls="--", **kwargs)
        lambda_info = f"(λ={lasso_cv.alpha_:.6f})"

    plt.xlabel(f'Lambda{lambda_info}')
    plt.ylabel('Coefficients')
    # plt.show()
    return lasso_cv.alpha_


def lasso_cv_efficiency(X_data, y_data, alpha_logmin=-3, points=50, cv: int = 10, ensure_lastn: int = None,
                        model_name: str = 'lasso', force_alpha=None, **kwargs):
    """
    Args:
        Xdata: 训练数据
        ydata: 测试数据
        alpha_logmin: alpha的log最小值
        points: 打印的数据密度
        cv: 交叉验证次数
        ensure_lastn: bool, 确保是最后一个MSE不降的alpha
        model_name: 可选使用Lasso还是ElasticNet。
        force_alpha: deprecated!
        **kwargs: 其他的图像样式
            # 数据点标记, fmt="o"
            # 数据点大小, ms=3
            # 数据点颜色, mfc="r"
            # 数据点边缘颜色, mec="r"
            # 误差棒颜色, ecolor="b"
            # 误差棒线宽, elinewidth=2
            # 误差棒边界线长度, capsize=2
            # 误差棒边界厚度, capthick=1
    Returns:
    """
    alphas = np.logspace(alpha_logmin, 0, points)
    if model_name.lower() == 'lasso':
        lasso_cv = LassoCV(alphas=alphas, cv=cv, n_jobs=-1, random_state=0).fit(X_data, y_data)
    else:
        lasso_cv = ElasticNetCV(alphas=alphas, cv=cv, n_jobs=-1, random_state=0).fit(X_data, y_data)
    MSEs = lasso_cv.mse_path_
    MSEs_mean = np.mean(MSEs, axis=1)
    MSEs_std = np.std(MSEs, axis=1)

    # plt.rcParams['figure.figsize'] = (10.0, 8.0)
    default_params = {'fmt': "o", 'ms': 3, 'mfc': 'r', 'mec': 'r', 'ecolor': 'b', 'elinewidth': 2, 'capsize': 2,
                      'capthick': 1}
    default_params.update(kwargs)
    plt.errorbar(lasso_cv.alphas_, MSEs_mean, yerr=MSEs_std, **default_params)
    plt.semilogx()
    if ensure_lastn is not None and lasso_cv.alpha_ == 1.0:
        lasso_cv.alpha_ = alphas[-ensure_lastn]
        logger.warning(f'你正在使用ensure_lastn={ensure_lastn}...')

    lambda_info = ''
    if force_alpha is not None:
        lasso_cv.alpha_ = force_alpha
    if lasso_cv.alpha_ != 1.0:
        plt.axvline(lasso_cv.alpha_, color='black', ls="--", **kwargs)
        lambda_info = f"(λ={lasso_cv.alpha_:.6f})"

    plt.xlabel(f'Lambda{lambda_info}')
    plt.ylabel('MSE')
    ax = plt.gca()
    y_major_locator = MultipleLocator(0.1)
    ax.yaxis.set_major_locator(y_major_locator)
    # plt.show()


def calc_icc(Y, icc_type="icc(3,1)"):
    """
    Args:
        Y: 待计算的数据
        icc_type: 共支持 icc(2,1), icc(2,k), icc(3,1), icc(3,k)四种
    """

    [n, k] = Y.shape

    # Degrees of Freedom
    dfc = k - 1
    dfe = (n - 1) * (k - 1)
    dfr = n - 1

    # Sum Square Total
    mean_Y = np.mean(Y)
    SST = ((Y - mean_Y) ** 2).sum()

    # create the design matrix for the different levels
    x = np.kron(np.eye(k), np.ones((n, 1)))  # sessions
    x0 = np.tile(np.eye(n), (k, 1))  # subjects
    X = np.hstack([x, x0])

    # Sum Square Error
    predicted_Y = np.dot(
        np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X))), X.T), Y.flatten("F")
    )
    residuals = Y.flatten("F") - predicted_Y
    SSE = (residuals ** 2).sum()

    MSE = SSE / dfe

    # Sum square column effect - between colums
    SSC = ((np.mean(Y, 0) - mean_Y) ** 2).sum() * n
    # MSC = SSC / dfc / n
    MSC = SSC / dfc

    # Sum Square subject effect - between rows/subjects
    SSR = SST - SSC - SSE
    MSR = SSR / dfr

    if icc_type == "icc(2,1)" or icc_type == 'icc(2,k)':
        if icc_type == 'icc(2,k)':
            k = 1
        ICC = (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n + 1e-6)
    elif icc_type == "icc(3,1)" or icc_type == 'icc(3,k)':
        if icc_type == 'icc(3,k)':
            k = 1
        # print(SSR, dfr, SSE, dfe, MSR, MSE, MSR - MSE)
        ICC = (MSR - MSE) / (MSR + (k - 1) * MSE + 1e-6)
    else:
        raise ValueError(f"icc_type{icc_type} 没有找到！")
    return max(0, ICC)


def icc_filter_features(dfs: List[DataFrame], threshold: float = 0.9, with_icc: bool = False,
                        columns_start: int = 0, not_icc: List[str] = None) -> Union[list, dict]:
    """
    ICC校验，取大于threshold阈值的特征
    Args:
        dfs: DataFrame，一般对应于不同人标注的结果。
        threshold: ICC阈值，大于此阈值返回。默认0.9，此值为None，返回每个特征的ICC dict.
        with_icc: Boolean, 是否返回每个特征具体的ICC值。
        columns_start: int，筛选的特征起始列。
        not_icc: 不进行ICC的列名。
    Returns:

    """
    assert isinstance(dfs, (list, tuple)) and len(dfs) > 1, "做ICC校验的数据至少需要2组数据。"
    assert all(isinstance(df, DataFrame) for df in dfs), "所有的数据必须是DataFrame。"
    assert all(dfs[0].shape == df.shape and len(df.shape) == 2 for df in dfs), '所有的数据维度必须相同'
    columns = dfs[0].columns
    data = np.array(dfs)
    selected_features = {}
    # if isinstance(not_icc, (list, tuple)):
    #     not_icc = [not_icc]
    for idx, column in enumerate(columns):
        if not_icc is not None and column in not_icc:
            logger.info(f'{column}不进行ICC。')
            continue
        if idx >= columns_start:
            try:
                icc = calc_icc(data[:, :, idx].T)
                selected_features[column] = icc
            except Exception as e:
                logger.error(f"{column}计算失败，因为：{e}")
    if threshold is not None:
        sel_features = [k for k, v in selected_features.items() if v > threshold]
    else:
        sel_features = selected_features
    if with_icc:
        return sel_features, selected_features
    else:
        return sel_features


def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model


def calculate_net_benefit_all(thresh_group, y_label):
    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
    total = tp + tn
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all


def plot_DCA(y_pred_score, y_label, title='Model DCA', labels=None, y_min=None):
    """
    plot DCA曲线
    Args:
        y_pred_score: list或者array，为模型预测结果。
        y_label: list或者array，为真实结果。
        title: 图标题
        labels: 图例名称
        y_min: 最小值
    Returns:

    """
    thresh_group = np.arange(0, 1, 0.01)
    fig, ax = plt.subplots()
    net_benefit_all = calculate_net_benefit_all(thresh_group, y_label)
    net_benefit_model = None
    if not isinstance(y_pred_score, (tuple, list)):
        y_pred_score = [y_pred_score]
    if not isinstance(labels, (tuple, list)):
        labels = [labels]
    assert len(y_pred_score) == len(labels)
    for y_pred, label in zip(y_pred_score, labels):
        net_benefit_model = calculate_net_benefit_model(thresh_group, y_pred, y_label)
        ax.plot(thresh_group, net_benefit_model, label=label if label is not None else 'Model')
    ax.plot(thresh_group, net_benefit_all, color='black', label='Treat all')
    ax.plot((0, 1), (0, 0), color='black', linestyle=':', label='Treat none')

    if len(y_pred_score) == 1:
        # Fill，显示出模型较于treat all和treat none好的部分
        y2 = np.maximum(net_benefit_all, 0)
        y1 = np.maximum(net_benefit_model, y2)
        ax.fill_between(thresh_group, y1, y2, color='crimson', alpha=0.2)

    # Figure Configuration， 美化一下细节
    ax.set_xlim(0, 1)
    if y_min is not None:
        y_min = max(y_min, net_benefit_model.min() - 0.15)
    else:
        y_min = net_benefit_model.min() - 0.15
    ax.set_ylim(y_min, net_benefit_model.max() + 0.15)  # justify the y axis limitation
    ax.set_xlabel(
        xlabel='Threshold Probability',
        fontdict={'family': 'Times New Roman', 'fontsize': 15}
    )
    ax.set_ylabel(
        ylabel='Net Benefit',
        fontdict={'family': 'Times New Roman', 'fontsize': 15}
    )
    ax.grid('major')
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))
    ax.legend(loc='upper right')
    plt.title(title)


def smote_resample(X, y, method='smote'):
    from imblearn.combine import SMOTEENN, SMOTETomek
    from imblearn.over_sampling import SMOTE, KMeansSMOTE
    if method == 'smote':
        model = SMOTE(random_state=0)
    elif method == 'smote_enn':
        model = SMOTEENN(random_state=0)
    elif method == 'smote_tomek':
        model = SMOTETomek(random_state=0)
    elif method == 'kmeans_smote':
        model = KMeansSMOTE(random_state=0)
    else:
        raise ValueError('Method not found! 目前仅支持: smote, smote_enn, smote_tomek, kmeans_smote')
    return model.fit_resample(X, y)


def fillna(x: pd.DataFrame, inplace: bool = False, fill_mod: str = 'best'):
    """
    填充样本空值
    Args:
        x: 待填充的数据DataFrame
        inplace: 是否直接改变原始x的结果
        fill_mod: 填充模式，目前支持制定数值填充，以及
            * best: 自动寻找最优方式，连续值填充均值，离散值填充中位数。
            * 50%: 中位数填充。
            * mean: 均值填充。

    Returns:

    """
    assert isinstance(fill_mod, (int, float)) or fill_mod in ['best', '50%', 'mean']
    if inplace:
        data = x
    else:
        data = copy.deepcopy(x)
    desc = x.describe()
    for c in desc.columns:
        if fill_mod == 'best':
            if data[c].dtype == float:
                fill_value = desc[c]['mean']
            else:
                fill_value = desc[c]['50%']
        elif isinstance(fill_mod, (float, int)):
            fill_value = fill_mod
        else:
            fill_value = desc[c][fill_mod]
        data[c] = data[c].fillna(fill_value)
    return data


def plot_feature_importance(model, feature_names=None, save_dir=None, prefix='', trace: bool = False, topk=None):
    """

    Args:
        model: 模型类
        feature_names: 特征可读名称。
        save_dir: 保存图像目录
        prefix: prefix, 默认为空。
        topk: int， 最重要的k个特征
        trace: 是否跟踪错误信息。

    Returns:

    """
    try:
        imp = model.feature_importances_
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(len(imp))]
        imp_ = sorted(list(np.array([feature_names, imp]).T), key=lambda x: x[1], reverse=True)[:topk]
        importance = pd.DataFrame(imp_, columns=['feature_name', 'importance'])
        importance['importance'] = importance['importance'].astype(float)
        sns.barplot(y="feature_name", x="importance", data=importance)
        plt.title(model.__class__.__name__)
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'{prefix}model_{model.__class__.__name__}_feature_importance.svg'),
                        bbox_inches='tight')
            importance.to_csv(os.path.join(save_dir, f'{prefix}model_{model.__class__.__name__}_importance_weight.csv'),
                              index=False)
        plt.show()
        return imp_
    except:
        if trace:
            import traceback
            traceback.print_exc()
        pass


init_CN()
