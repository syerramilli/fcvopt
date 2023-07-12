from fcvopt.configspace import ConfigurationSpace
from ConfigSpace import Float,Integer,Categorical

DATA_IDS = {
    'gina':1038,
    'hiva':1039,
    'madelon':1485,
    'bioresponse':4134,
}

def get_config() -> ConfigurationSpace:
    config = ConfigurationSpace(seed=1234)
    config.add_hyperparameters([
        Integer('max_depth',bounds=(1,12),log=True),
        Float('min_impurity_decrease',bounds=(1e-8,10),log=True),
        Float('max_features',bounds=(0.005,0.5),log=True),
        Integer('min_samples_split',bounds=(5,250),log=True),
    ])

    config.generate_indices()
    return config