import torch.nn as nn 
from backbone import Darknet53, CNNBlock
class ScalePrediction(nn.Module): 
    def __init__(self, in_channels, num_box, num_classes): 
        super(ScalePrediction, self).__init__() 
        self.CNNBlock1 = CNNBlock(in_channels, in_channels*2, kernel_size=3, stride=1, padding=1) 
        self.CNNBlock2= CNNBlock(in_channels*2, (num_classes+5)*num_box, kernel_size=1, stride=1, padding=1, bn_act=False) 
        self.num_classes = num_classes 
    def forward(self, x): 
        x = self.CNNBlock2(self.CNNBlock1(x)) 
        x = x.view(x.size(0), 3, (self.num_classes+5), x.size(2), x.size(3)) 
        x = x.permute(0, 1, 3, 4, 2) 
        return x 

class YOLOV3: 
    def __init__(self, in_channels, num_classes): 
        self.backbone = Darknet53()
        self.in_channels = in_channels 
        self.num_classes = num_classes

    def forward(self, X): 
        predictions = self.backbone.forward(X) #return (Batch, (num_classes+5)*num_box, S, S) 

        for i in range(predictions): 
            predictions[i] = ScalePrediction(
                in_channels=predictions[i].size(1),
                num_box=3, 
                num_classes=self.num_classes
            ).forward(predictions[i]) #return (Batch, num_box, S, S, num_classes+5) 
        return predictions
    
        
        