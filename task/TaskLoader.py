from collections.abc import Mapping
import copy


class Opt(object):
    def __init__(self, init = None):
        super().__init__()
        
        if init is not None:
            self.merge(init)
        
    def merge(self, opts, ele_s = None):
        '''
        Only merge the key-value not in the current Opt.\n
        Using ele_s to select the element in opts to be merged.
        '''
        if isinstance(opts, Mapping):
            new = opts
        else:
            assert isinstance(opts, object)
            new = vars(opts)
        
        if ele_s is None:
            for key in new:
                if not key in self.dict:
                    self.dict[key] = copy.copy(new[key]) 
        else:
            for key in ele_s:
                if not key in self.dict:
                    self.dict[key] = copy.copy(new[key])
                
    def update(self, opts, ignore_unk = False):
        if isinstance(opts, Mapping):
            new = opts
        else:
            assert isinstance(opts, object)
            new = vars(opts)
        for key in new:
            if not key in self.dict and ignore_unk is False:
                raise ValueError(
                "Unknown config key '{}'".format(key))
            self.dict[key] = copy.copy(new[key])
            
    @property
    def dict(self):
        '''Gives dict-like access to Params instance by params.dict['learning_rate']'''
        return self.__dict__