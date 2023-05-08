import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from collections import OrderedDict

class ConfigurationSpace(CS.ConfigurationSpace):

    def generate_indices(self) -> None:
        self.quant_index = []
        self.quant_names = []
        self.qual_index = []
        self.qual_names = []
        self.num_levels = OrderedDict()
        self.ndim = 0

        conditionals = self.get_all_conditional_hyperparameters()

        for i,hyp in enumerate(self.get_hyperparameters()):
            self.ndim +=1
            if isinstance(hyp,CSH.UniformFloatHyperparameter) or isinstance(hyp,CSH.UniformIntegerHyperparameter):
                self.quant_index.append(i)
                self.quant_names.append(hyp.name)
                
            elif isinstance(hyp,CSH.CategoricalHyperparameter):
                self.qual_index.append(i)
                self.qual_names.append(hyp.name)
                self.num_levels[i] = len(hyp.choices)
    
    def get_conf_from_array(self,x:np.ndarray) -> CS.Configuration:
        conf_dict = {}
        for idx,hyp in enumerate(self.get_hyperparameters()):
            conf_dict[hyp.name] = hyp._transform(
                np.round(x[idx]).astype(int) if isinstance(hyp,CSH.CategoricalHyperparameter) else x[idx]
            )
        
        return CS.Configuration(self,conf_dict)
    
    def latinhypercube_sample(self,size:int):
        # valid only for configurations with all numerical parameters 
        n_dim = self.ndim
        
        # generate row and column grids
        grid_bounds = np.stack([np.linspace(0., 1., size+1) \
                                for i in np.arange(n_dim)],axis=1)
        grid_lower = grid_bounds[:-1,:]
        grid_upper = grid_bounds[1:,:]
        
        # generate 
        grid = grid_lower + (grid_upper-grid_lower)*self.random.rand(size,n_dim)
        
        # shuffle and return
        for i in range(n_dim):
            self.random.shuffle(grid[:,i])
        
        return [self.get_conf_from_array(x) for x in grid]
    