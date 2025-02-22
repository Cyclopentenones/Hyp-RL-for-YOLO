import torch 
import torch.nn as nn 
ACTION_SPACE = [1e-2, 1e-3, 1e-4, 1e-5] 
class TuningRLEnv(nn.Module): 
    def __init__(self, hyperparameter): 
        super(TuningRLEnv, self).__init__()
        self.action_space = ACTION_SPACE 
        self.surrmodel = Surrmodel(hyperparameter) 

    def reset(self): 
        s
