import torch
import torch.nn as nn
from model import Net
from torch.distributions import Beta

class Buffer:
    def __init__(self, Maximum_Batch_size=10):
        self.states = [] 
        self.actions = []
        self.logs = [] 
        self.values = []
        self.rewards = []
        self.max = Maximum_Batch_size
    def push(self, state, action, log, value, reward, done): 
        if len(self.states) > self.max:
            self.states.pop(0)
            self.actions.pop(0)
            self.logs.pop(0)
            self.values.pop(0)
            self.rewards.pop(0)
        self.states.append(state)
        self.actions.append(action)
        self.logs.append(log)   
        self.values.append(value)
        self.rewards.append(reward)


class TuningAgent(nn.Module):
    def __init__(self, device): 
        self.net = Net() 
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = 1e-3) 
        self.buffer = Buffer() 
        self.device = device
        self.clip = 0.1 
        self.k = 10 
        self.batch_size = 32 
        self.gamma = 0.99 
        self.eps = 0.2
        self.lambda_gae = 0.95

    def select_action(self, state): #state = [Batch, [mAp, IOU, F1_score, FPS, Robutness]]
        state = torch.tensor(state, dtype = torch.float).to(self.device) 
        
        with torch.no_grad(): 
            alpha, beta, v = self.net(state) 
        dist = Beta(alpha, beta) 
        action = dist.sample() 
        a_log = dist.log_prob(action).sum(dim=1)  

        return action, a_log, v 
    
    def GAE(self, rewards, values, dones): 
        """
        GAE (Generalized Advantage Estimation)  = delta + gamma+lambda*(1-dones)*gae
        with delta = reward + gamma*values[t+1] - values[t]
        """
        gae = 0 
        advantages = []
        values = values + [0]
        for t in reversed(range(len(rewards))): 
            delta = rewards[t] + self.gamma*values[t+1]*(1-dones[t]) - values[t]
            gae = delta + self.gamma*self.lambda_gae*(1-dones[t])*gae
            advantages.insert(0, gae) 
        return advantages
    
    def update(self):
        """Rules of update PPO:
        1. Collect batch_size samples
        2. Calculate GAE
        3. Update network with K_epochs
        """
        states = torch.tensor(self.buffer.states, dtype=torch.float).to(self.device)
        actions = torch.tensor(self.buffer.actions, dtype=torch.float).to(self.device)
        logs_old = torch.tensor(self.buffer.logs, dtype=torch.float).to(self.device)
        values = torch.tensor(self.buffer.values, dtype=torch.float).to(self.device)
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float).to(self.device)

        # Calculate GAE 
        advantages = self.GAE(rewards, values)
        advantages = torch.tensor(advantages, dtype=torch.float).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        #Loops of K_epochs 
        for _ in range(self.k):
            alpha, beta, new_values = self.net(states)
            dist = Beta(alpha, beta)
            logs_new = dist.log_prob(actions).sum(dim=1)
            ratio = torch.exp(logs_new - logs_old)

            # PPO Clip Objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            target_values = rewards + self.gamma * new_values.detach()
            critic_loss = nn.MSELoss()(new_values, target_values)

            total_loss = actor_loss + 0.5 * critic_loss 

            # Backpropagation
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)  # Gradient clipping
            self.optimizer.step()


            


    


