import torch
import torch.nn as nn
from Encode.Encode import Duck 
#Env
class TuningEnv(nn.Module):
    def __init__(self, data, targets, num_classes, random=False, device='cuda'):
        super(TuningEnv, self).__init__()
        self.device = device
        self.env = data.to(device)
        self.target = targets.to(device)
        self.num_classes = num_classes

        self.state = Duck(16, 0.1, 'large', 'ViT-Hybrid').to(self.device)
        self.FC= nn.Sequential(
                nn.Linear(1024, self.num_classes).to(self.device),
                nn.Softmax(dim=-1)
                ).to(device)  
        self.optimizer = torch.optim.Adam(
            list(self.state.parameters()) + list(self.FC.parameters()), lr=1e-4
        )
        self.criterion = nn.CrossEntropyLoss()
        self.random = random
        self.old_score = [1e-5]
        self.i = 0

    def reset(self):
        self.i = 0
        self.state = Duck(16, 0.1, 'large', 'ViT').to(self.device)
        self.FC= nn.Sequential(
                nn.Linear(1024, self.num_classes).to(self.device),
                nn.Softmax(dim=-1)
                )
        self.old_score = [1e4]
    

    def action(self, tuning_agent):
        #apply action 
        action, log_prob, value = tuning_agent.act()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] += action
            param_group['lr'] = max(1e-6, param_group['lr'])

        #choose mode of dataset
        if self.random:
            self.i = torch.randint(0, len(self.env), (1,))
        else:
            self.i = min(self.i, len(self.env) - 1)

        # get new state(score) and update weight
        prev_score = self.old_score[-1]
        y_pred = self.state(self.env)
        y_pred = y_pred[:,0,:]
        y_pred = self.FC(y_pred)
        new_score = self.criterion(y_pred, self.target)
        
        self.optimizer.zero_grad()
        new_score.backward()
        self.optimizer.step()

        # update index of dataset
        self.i += 1

        #reward
        r = (prev_score - new_score) / (prev_score + 1e-8)
        self.old_score.append(new_score)

        return action, log_prob, r, value, new_score

    def done(self):
        if len(self.old_score) < 2:
            return False
        return abs(self.old_score[-2] - self.old_score[-1]) < 1e-6 or self.i >= len(self.env)