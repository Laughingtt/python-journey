from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
)

cs = ConfigurationSpace()
hidden_layer_depth = UniformIntegerHyperparameter(
    name="hidden_layer_depth", lower=1, upper=3, default_value=1
)
num_nodes_per_layer = UniformIntegerHyperparameter(
    name="num_nodes_per_layer", lower=16, upper=216, default_value=32
)
activation = CategoricalHyperparameter(
    name="activation",
    choices=["identity", "logistic", "tanh", "relu"],
    default_value="relu",
)
alpha = UniformFloatHyperparameter(
    name="alpha", lower=0.0001, upper=1.0, default_value=0.0001
)
solver = CategoricalHyperparameter(
    name="solver", choices=["lbfgs", "sgd", "adam"], default_value="adam"
)
cs.add_hyperparameters(
    [
        hidden_layer_depth,
        num_nodes_per_layer,
        activation,
        alpha,
        solver,
    ]
)


print(cs)