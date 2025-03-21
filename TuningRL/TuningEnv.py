import torch 
import torch.nn as nn 
from YOLOv5.models.yolo import ClassificationModel
class SurModel: 
    def __init__(self, mode, num_classes, device='gpu'):
        self.device = device
        self.model = ClassificationModel(f"{mode}.yaml", nc=num_classes).to(device)
        self.model.load_state_dict(torch.load(f"{mode}.pt", map_location=device)["model"])
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def forward(self, x, target=None, update_weights=True):
            x = x.to(self.device)
            logits = self.model(x)

            if target is not None: 
                target = target.to(self.device)
                loss = self.criterion(logits, target)
                if update_weights:  
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                return logits, loss 

            return logits  
    def update(self,action):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] += action
        
        
        
    
class TuningEnv(nn.Module): 
    def __init__(self, data, random=False): 
        super(TuningEnv, self).__init__()
        self.env = data  
        self.state = SurModel(mode="YOLOv5n")
        self.random = random
        self.old_score = [1e-5] 

    def reset(self): 
        self.i = 0  
        self.state = SurModel(mode="YOLOv5n")
        self.old_score = [1e-5]  

    def action(self, TuningAgent): 
        ######################### chọn hành động ##########################
        action, log_prob, v = TuningAgent.act()  
        self.state.update(action)  

        if self.random: 
           self.i = torch.randint(0, len(self.env), (1,)).item()

        score = TuningAgent.memory.scores[-1]


        ##################Chỗ này train YOLO#####################
        new_score = self.state.forward(self.env[self.i]) 
        #########################################################

        self.i += 1

        r = torch.log(new_score / score) + 1

        self.old_score.append(new_score)

        return  action, log_prob, r,  v, new_score
    
    def done(self):
        if len(self.old_score) < 2: 
            return False  
        return abs(self.old_score[-1] - self.old_score[-2]) < 1e-3






#### Choose action -> Action, log_p, Value
#### Apply actioin -> Action, log_p, Score, Value, Score 
### =>>> Action trong main 

### Update trọng số mạng: 
