import torch 
import torch.nn as nn

class Criterion(nn.Module): 
    def __init__(self): 
        super(Criterion, self).__init__()
        self.MSE = nn.MSELoss() 
        self.CrossEntropy = nn.CrossEntropyLoss() 
        self.BCE = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
    
    def Loss(self, target, prediction):
        #target = (Batch, 1, S, S, 5 + class_idx) 
        #prediction = (Batch, num_box, S, S, 5 + num_classes)
        """
           We have image and bbox (x, y, w, h, obj, class)
           Then convert image to n*(S*S) cells, each cell has 1 box (x, y, w, h, conf, class_idx)
           Then we have to calculate loss for each cell
        """
        mask_obj = target[..., 4] == 1
        mask_no_obj = target[..., 4] == 0 

        #Calculate IOU Loss 
        

