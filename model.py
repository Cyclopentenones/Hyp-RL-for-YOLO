import torch.nn as nn 
from backbone import Darknet53 

class YOLOV3: 
    def __init__(self, C, S, B): 
        self.S = S 
        self.B = B 
        self.C = C 
        self.backbone = Darknet53() 
    
    def predict(self, X): 
        prediction = self.backbone(X) 
        prediction = prediction.view(-1, self.S, self.S, self.B*5 + self.C)
        
        