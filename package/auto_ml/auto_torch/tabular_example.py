from autoPyTorch.api.tabular_classification import TabularClassificationTask


# data and metric imports
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

X, y = sklearn.datasets.load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(X, y, random_state=1)

# initialise Auto-PyTorch api
api = TabularClassificationTask()

# Search for an ensemble of machine learning algorithms
api.search(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    optimize_metric='accuracy',
    total_walltime_limit=300,
    func_eval_time_limit_secs=50,
    memory_limit=None,
)

# Calculate test accuracy
y_pred = api.predict(X_test)
score = api.score(y_pred, y_test)
print("Accuracy score", score)
print(api.get_models_with_weights())
print(api.show_models())
print(api.sprint_statistics())
