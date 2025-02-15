import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader 

class LoadData: 
    def __init__(self, dataset): 
        self.data =dataset 
    

    def __len__(self): 
        return len(self.data) 
    