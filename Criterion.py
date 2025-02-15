import torch 
import torch.nn as nn
import torch.nn.functional as f
from utils import intersect_over_union 
from Kmeans_selection import anchor 

class Criterion(nn.Module): 
    def __init__(self): 
        super(Criterion, self).__init__()
        self.MSE = nn.MSELoss() 
        self.CE = nn.CrossEntropyLoss() 
        self.BCE = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.lambda_coord = 5
        self.lambda_obj = 1
        self.lambda_no_obj = 0.5
        self.lambda_class = 1 
    
    def forward(self, target, prediction):
        #target = (Batch, 1, S, S, 5 + class_idx) 
        #prediction = (Batch, num_box, S, S, 5 + num_classes)
        """
           We have image and bbox (x, y, w, h, obj, class)
           Then convert image to n*(S*S) cells, each cell has 1 box (x, y, w, h, conf, class_idx)
           Then we have to calculate loss for each cell
        """
        mask_obj = target[..., 4] == 1
        mask_no_obj = target[..., 4] == 0 


        #Loss coordinate 
        box1 = prediction[..., 0:4]*mask_obj # Batch, tx, ty, tw, th
        box2 = target[..., 0:4]*mask_obj #Batch, bx, by, bw, bh 
        iou = intersect_over_union(box1, box2) 
        """ 
            We calculate the loss of x, y by Bx, By values 
            We calculate the loss of w, h by tw, th values 
        """
        box1[..., 0:2] = self.sigmoid(box1[..., 0:2]) 
        box2[..., 2:4] = torch.log(box2[..., 2:4]/anchor + 1e-6)

        loss_coord = self.lambda_coord*self.MSE(box1[..., 0:2], box2[..., 0:2]) + self.MSE(nn.square(box1[..., 2:4]), nn.square(box2[..., 2:4])) 

        #objectness loss 
        loss_obj =self.lambda_obj*self.MSE(prediction[..., 4]*mask_obj, target[..., 4]*mask_obj) 
        loss_no_obj = self.lambda_no_obj*self.BCE(prediction[..., 4]*mask_no_obj, target[..., 4]*mask_no_obj) #try focal loss 

        #class loss 
        one_hot = f.one_hot(target[..., 5], num_classes = prediction[..., 5:].shape[-1])
        loss_class = self.lambda_class*self.CE(prediction[..., 5], one_hot) 