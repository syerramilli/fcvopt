import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from collections import OrderedDict
from typing import List, Optional, Any

class ConfigurationSpace(CS.ConfigurationSpace):
    """Extended ConfigurationSpace for FCVOpt optimizers.

    This is a wrapper around :class:`ConfigSpace.ConfigurationSpace` that provides
    additional utilities for hyperparameter optimization in FCVOpt:

      - Reconstructing a :class:`ConfigSpace.Configuration` from a numeric array via
        :meth:`get_conf_from_array`.
      - Generating Latin Hypercube samples of configurations

    Example:
        .. code-block:: python
            
            from fcvopt.configspace import ConfigurationSpace
            from ConfigSpace import Float, Integer
            
            config = ConfigurationSpace(seed=1234)
            config.add([ # add two hyperparameters
                Float('x1', lower=0.0, upper=1.0),
                Integer('x2', lower=1, upper=10),
            ])

            # generate a random latin hypercube sample of size 5
            samples_list = config.latinhypercube_sample(size=5)

            # convert the list of configurations to a numpy array with scaled values
            samples_array = np.array([conf.get_array() for conf in samples_list])

            # convert a numeric array back to a Configuration object
            conf = config.get_conf_from_array(samples_array[0])
    """
    def generate_indices(self) -> None:
        """
        deprecated since 0.4.0
        """
        return None

    def get_conf_from_array(self, x: np.ndarray) -> CS.Configuration:
        """Convert a numeric array into a Configuration object.

        Args:
            x (np.ndarray): 1D array of length ``ndim`` containing numeric values.
                For categorical hyperparameters, the array value is rounded to the
                nearest integer index of the category.

        Returns:
            CS.Configuration: A configuration mapping hyperparameter names to values.
        """
        conf_dict = {}
        for idx, hyp in enumerate(self.values()):
            if isinstance(hyp, CSH.CategoricalHyperparameter):
                # interpret x[idx] as integer index into hyp.choices
                level = int(np.round(x[idx]))
                conf_dict[hyp.name] = hyp.choices[level]
            else:
                conf_dict[hyp.name] = hyp.to_value(x[idx])

        return CS.Configuration(self, conf_dict)

    def latinhypercube_sample(self, size: int) -> List[CS.Configuration]:
        """Generate a Latin hypercube sample over the numerical inputs.

        Note:
            Binary values categorical inputs are supported while general categorical
            inputs are not.

        Args:
            size (int): Number of configurations to sample.

        Returns:
            List[CS.Configuration]: List of sampled configurations of length ``size``.

        Raises:
            ValueError: If the configuration space contains categorical parameters.
        """
        for hyp in self.values():
            if isinstance(hyp, CSH.CategoricalHyperparameter) and len(hyp.choices) > 2:
                raise ValueError(f"Only binary valued categorical inputs are supported.")

        n_dim = len(self)

        # Create size-by-n_dim uniform grid and sample within each cell
        grid_bounds = np.linspace(0.0, 1.0, size + 1)
        lower = np.repeat(grid_bounds[:-1][:, None], n_dim, axis=1)
        upper = np.repeat(grid_bounds[1:][:, None], n_dim, axis=1)
        # random offset within each cell
        uniform = self.random.rand(size, n_dim)
        samples = lower + (upper - lower) * uniform

        # shuffle each dimension to ensure stratification
        for d in range(n_dim):
            self.random.shuffle(samples[:, d])

        # Convert to Configuration objects
        return [self.get_conf_from_array(row) for row in samples]

    @classmethod
    def from_serialized_dict(cls, serialized_dict: dict) -> "ConfigurationSpace":
        """Reconstruct a ConfigurationSpace from a serialized dictionary.

        Args:
            serialized_dict: Dictionary containing serialized configuration space.

        Returns:
            ConfigurationSpace: Reconstructed configuration space.
        """
        # Create ConfigurationSpace from base serialization using parent class method
        config_space = CS.ConfigurationSpace.from_serialized_dict(serialized_dict)

        # Convert to our extended class
        # Don't try to extract seed from the original space, use None
        extended_space = cls(seed=None)
        for hp in config_space.values():
            extended_space.add(hp)

        return extended_space
