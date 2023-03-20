import autosklearn.classification
import sklearn.model_selection
from sklearn import datasets
import sklearn.metrics

if __name__ == "__main__":
    X, y = datasets.load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=120,
        per_run_time_limit=30,
        tmp_folder="/tmp/autosklearn_classification_example_tmp",
        include={
            'classifier': ["random_forest"],
            'feature_preprocessor': ["no_preprocessing"]
        }
    )
    automl.fit(X_train, y_train)
    y_hat = automl.predict(X_test)
    automl.get_models_with_weights()
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))
    print(automl.leaderboard())
    models_with_weights = automl.get_models_with_weights()
    with open('../../preprocess/models_report.txt', 'w') as f:
        for model in models_with_weights:
            f.write(str(model) + '\n')

"""
['adaboost', 'bernoulli_nb', 'decision_tree', 'extra_trees', 'gaussian_nb', 'gradient_boosting', 'k_nearest_neighbors', 'lda', 'liblinear_svc', 'libsvm_svc', 'mlp', 'multinomial_nb', 'passive_aggressive', 'qda', 'random_forest', 'sgd']
"""
