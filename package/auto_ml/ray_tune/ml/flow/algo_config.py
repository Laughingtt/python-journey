class AlgoConfig:
    def __init__(self, model_type, include, optional_algorithm):
        self.model_type = model_type
        self.include = include
        self.optional_algorithm = optional_algorithm

    def check_algorithm(self):
        optional_algorithm = [i.lower() for i in self.optional_algorithm]
        for algo in self.include:
            if algo.lower() not in optional_algorithm:
                raise ValueError("algorithm {} "
                                 "not in {}".format(algo, self.optional_algorithm))
