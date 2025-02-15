import torch 
import torch.nn as nn 

def convert_to_real_values(box, image_size, anchor, grid_size): 
    """Converts the box coordinates from relative values to real values. 
    Args: 
        box: A tensor of shape (4,) representing the box coordinates in relative values. 
        image_size: A tuple of the form (width, height) representing the image size. 
    Returns: 
        A tensor of shape (4,) representing the box coordinates in real values. 
    """
    W, H = image_size 
    tx, ty ,tw, th = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    Bx = nn.Sigmoid(tx) + grid_size
    By = nn.Sigmoid(ty) + grid_size 
    Bw = anchor[0]*torch.exp(tw)
    Bh = anchor[1]*torch.exp(th) 
    return Bx, By, Bw, Bh

def intersect_over_union(box1, box2): 
    """Calculates the intersection over union of two boxes."""
    x1, y1, w1, h1 = convert_to_real_values(box1) 
    x1_box1 = x1 - w1/2 
    y1_box1 = y1 + h1/2 
    x2_box1 = x1 + w1/2 
    y2_box1 = y1 - h1/2


    x2, y2, w2, h2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]
    x1_box2 = x2 - w2/2
    y1_box2 = y2 + h2/2
    x2_box2 = x2 + w2/2
    y2_box2 = y2 - h2/2


    x_inter1 = torch.max(x1_box1, x1_box2) 
    y_inter1 = torch.min(y1_box1, y1_box2)
    
    x_inter2 = torch.min(x2_box1, x2_box2)
    y_inter2 = torch.max(y2_box1, y2_box2)

    intersection = (x_inter2 - x_inter1).clamp(0) * (y_inter1 - y_inter2).clamp(0) 
    area_box1 = w1*h1
    area_box2 = w2*h2
    union = area_box1 + area_box2 - intersection
    return intersection/(union+1e-6)


    