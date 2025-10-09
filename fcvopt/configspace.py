import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from collections import OrderedDict
from typing import List

class ConfigurationSpace(CS.ConfigurationSpace):
    """Extended ConfigurationSpace for FCVOpt optimizers.

    This is a wrapper around :class:`ConfigSpace.ConfigurationSpace` that provides
    additional utilities for hyperparameter optimization in FCVOpt:

      - Indexing of continuous (quantitative) and categorical parameters via
        :meth:`generate_indices`.
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
            config.generate_indices()

            # generate a random latin hypercube sample of size 5
            samples_list = config.latinhypercube_sample(size=5)

            # convert the list of configurations to a numpy array with scaled values
            samples_array = np.array([conf.get_array() for conf in samples_list])

            # convert a numeric array back to a Configuration object
            conf = config.get_conf_from_array(samples_array[0])
    """

    def generate_indices(self) -> None:
        """Compute indices and names of quantitative and categorical hyperparameters.

        Populates the following attributes on the instance:

        - ``quant_index`` (List[int]): Indices of float and integer hyperparameters.
        - ``quant_names`` (List[str]): Names of quantitative hyperparameters.
        - ``qual_index`` (List[int]): Indices of categorical hyperparameters.
        - ``qual_names`` (List[str]): Names of categorical hyperparameters.
        - ``num_levels`` (OrderedDict[int, int]): Number of categories for each categorical index.
        - ``ndim`` (int): Total number of hyperparameters.
        """
        self.quant_index: List[int] = []
        self.quant_names: List[str] = []
        self.qual_index: List[int] = []
        self.qual_names: List[str] = []
        self.num_levels: OrderedDict = OrderedDict()
        self.ndim = 0

        for idx, hyp in enumerate(self.values()):
            self.ndim += 1
            if isinstance(hyp, (CSH.UniformFloatHyperparameter,
                                CSH.UniformIntegerHyperparameter)):
                self.quant_index.append(idx)
                self.quant_names.append(hyp.name)
            elif isinstance(hyp, CSH.CategoricalHyperparameter):
                self.qual_index.append(idx)
                self.qual_names.append(hyp.name)
                self.num_levels[idx] = len(hyp.choices)

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
        """Generate a Latin hypercube sample over the numerical hyperparameters.

        Only supports spaces with no categorical parameters.

        Args:
            size (int): Number of configurations to sample.

        Returns:
            List[CS.Configuration]: List of sampled configurations of length ``size``.

        Raises:
            ValueError: If the configuration space contains categorical parameters.
        """
        if getattr(self, 'qual_index', None) and len(self.qual_index) > 0:
            raise ValueError(
                "Latin hypercube sampling requires only numerical hyperparameters."
            )

        n_dim = getattr(self, 'ndim', None) or len(self.values())

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

    def to_serialized_dict(self) -> dict:
        """Serialize the configuration space to a dictionary for MLflow logging.

        Returns:
            dict: Serialized configuration space that can be saved as JSON.
        """
        # Use ConfigSpace's built-in serialization
        serialized = super().to_serialized_dict()

        # Add our custom attributes if they exist
        custom_attrs = {}
        if hasattr(self, 'quant_index'):
            custom_attrs['quant_index'] = self.quant_index
        if hasattr(self, 'quant_names'):
            custom_attrs['quant_names'] = self.quant_names
        if hasattr(self, 'qual_index'):
            custom_attrs['qual_index'] = self.qual_index
        if hasattr(self, 'qual_names'):
            custom_attrs['qual_names'] = self.qual_names
        if hasattr(self, 'num_levels'):
            custom_attrs['num_levels'] = dict(self.num_levels)
        if hasattr(self, 'ndim'):
            custom_attrs['ndim'] = self.ndim

        serialized['_fcvopt_attrs'] = custom_attrs
        return serialized

    @classmethod
    def from_dict(cls, serialized_dict: dict) -> "ConfigurationSpace":
        """Reconstruct a ConfigurationSpace from a serialized dictionary.

        Args:
            serialized_dict: Dictionary containing serialized configuration space.

        Returns:
            ConfigurationSpace: Reconstructed configuration space.
        """
        # Make a copy to avoid modifying the original
        serialized_copy = serialized_dict.copy()

        # Extract custom attributes
        custom_attrs = serialized_copy.pop('_fcvopt_attrs', {})

        # Create ConfigurationSpace from base serialization using parent class method
        config_space = CS.ConfigurationSpace.from_serialized_dict(serialized_copy)

        # Convert to our extended class
        # Don't try to extract seed from the original space, use None
        extended_space = cls(seed=None)
        for hp in config_space.values():
            extended_space.add(hp)

        # Restore custom attributes
        for attr_name, attr_value in custom_attrs.items():
            if attr_name == 'num_levels':
                # Convert back to OrderedDict with int keys
                setattr(extended_space, attr_name, OrderedDict((int(k), v) for k, v in attr_value.items()))
            else:
                setattr(extended_space, attr_name, attr_value)

        return extended_space
