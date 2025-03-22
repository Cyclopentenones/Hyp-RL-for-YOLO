import torch 
import torch.nn as nn 
from yolov5.models.yolo import ClassificationModel
import os
import yaml
current_dir = os.path.dirname(os.path.abspath(__file__))
yolov5_dir = os.path.join(current_dir, "yolov5") 
yaml_path = os.path.join(yolov5_dir, "models", "yolov5n.yaml")
with open(yaml_path, "r") as f:
    config = yaml.safe_load(f)
class SurModel: 
    def __init__(self, mode, num_classes, device='cuda'):
        self.device = device
        self.model = ClassificationModel(cfg=mode, nc=num_classes).to(device)
        self.model.load_state_dict(torch.load(f"{mode}.pt", map_location=device)["model"])
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def forward(self, x, target=None):
        assert target is not None
        x = x.to(self.device)
        logits = self.model(x)
        target = target.to(self.device) 

        loss = self.criterion(logits, target) 

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
    
    def update(self,action):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] += action
        
class TuningEnv(nn.Module): 
    def __init__(self, data, target, num_classes, random=False): 
        super(TuningEnv, self).__init__()
        self.env = data 
        self.target = target 
        self.num_classes = num_classes
        self.state = SurModel(mode=config, num_classes=num_classes)
        self.random = random
        self.old_score = [1e-5] 

    def reset(self): 
        self.i = 0  
        self.state = SurModel(mode=config, num_classes=self.num_classes)
        self.old_score = [1e-5]  

    def action(self, TuningAgent): 
        ######################### Chọn hành động ##########################
        action, log_prob, v = TuningAgent.act()  
        self.state.update(action)  

        if self.random: 
            self.i = torch.randint(0, len(self.env), (1,)).item()

        prev_score = self.old_score[-1]  # Lưu loss trước đó

        ################## Chỗ này train YOLO #####################
        _, new_score = self.state.forward(self.env[self.i], target=self.env[self.i])  # Trả về loss
        #########################################################

        self.i += 1

        # Reward: Nếu loss giảm thì reward dương
        r = (prev_score - new_score).item() / (prev_score.item() + 1e-8)

        self.old_score.append(new_score.item())

        return action, log_prob, r, v, new_score

    def done(self):
        if len(self.old_score) < 2: 
            return False  
        return abs(self.old_score[-1] - self.old_score[-2]) < 1e-3
