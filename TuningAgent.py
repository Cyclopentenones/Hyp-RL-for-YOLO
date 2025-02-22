import torch
from model import Net
from torch.distributions import Beta
class Buffer:
    def __init__(): 
        pass
class TuningAgent(nn.Module):
    def __init_(self, device): 
        self.net = Net() 
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = 1e-3) 
        self.buffer = Buffer() 
        self.device = device

    def select_action(self, state): #state = [Batch, [mAp, IOU, F1_score, FPS, Robutness]]
        state = torch.tensor(state, dtype = torch.float).to(self.device) 
        
        with torch.no_grad(): 
            alpha, beta, v = self.net(state) 
        dist = Beta(alpha, beta) 
        action = dist.sample() 
        a_log = dist.log_prob(action).sum(dim=1)  

        return action, a_log, v 
    
    def update(self, batch_size, gamma): 
        



