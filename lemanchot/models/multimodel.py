
""" 
    @title A Deep Semi-supervised Segmentation Approach for Thermographic Analysis of Industrial Components
    @organization Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
"""

from dotmap import DotMap

from lemanchot.models.core import BaseModule, load_model, load_model_inline__, model_register

@model_register('multimodel')
class MultiModule(BaseModule):
    def __init__(self, name: str, config: DotMap) -> None:
        super().__init__('multimodel', {})
        self.__initialize(config)
        
    def __initialize(self, config):
        # Initialize the embedded models
        self.model_lookup = {}
        for k in config.keys():
            cfg = config[k]
            tmp = load_model_inline__(cfg)
            tmp.to(self.device)
            self.model_lookup[k] = tmp
    
    def get_model(self, name):
        return self.model_lookup[name]
    
    def lookup_result(self, res):
        record = {}
        keys = self.model_lookup.keys()
        for index in range(len(res)):
            k = keys[index]
            record[k] = res[index]
        return record
            
    def forward(self, x):
        return tuple(map(lambda model: model(x), self.model_lookup.values()))