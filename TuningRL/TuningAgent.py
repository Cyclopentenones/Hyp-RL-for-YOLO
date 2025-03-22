import torch
import torch.nn as nn
from model import Net 

class Memory:
    def __init__(self):
        self.scores = [1e-5] * 8
        self.actions = [1e-5] * 8
        self.log_probs = [1e-5] * 8
        self.rewards = [1e-5] * 8
        self.values = [1e-5] * 8

    def store(self, score, action, log_prob, reward, value):
        for memory_list, val in zip(
            [self.scores, self.actions, self.log_probs, self.rewards, self.values],
            [score, action, log_prob, reward, value]
        ):
            memory_list.append(val)
            if len(memory_list) > 8:
                memory_list.pop(0)

        for f in [self.scores, self.actions, self.log_probs, self.rewards, self.values]:
            assert len(f) <= 8

    def clear(self):
        self.scores = [1e-5] * 8
        self.actions = [1e-5] * 8
        self.log_probs = [1e-5] * 8
        self.rewards = [1e-5] * 8
        self.values = [1e-5] * 8


class TuningAgent(nn.Module):
    def __init__(self, input_size, gamma=0.99, lr=1e-3, k_ep=5, device="cuda"):
        super(TuningAgent, self).__init__()
        self.net = Net(input_size=input_size).to(device)  # Net dùng phân phối Gaussian
        self.gamma = gamma
        self.k_ep = k_ep
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.memory = Memory()
        self.device = device

    def act(self):
        scores = self.memory.scores[-8:] 
        scores = torch.tensor(scores, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            mean, log_std, v = self.net(scores)
            std = torch.clamp(log_std.exp(), 1e-4, 1.0)  # Giới hạn độ lệch chuẩn
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            action = torch.tanh(action)*0.01 
            log_prob = dist.log_prob(action)
        return action.item(), log_prob, v

    def evaluate(self, scores, actions):
        mean, log_std, v = self.net(scores)
        std = torch.clamp(log_std.exp(), 1e-4, 1.0)  # Giữ giá trị std hợp lý
        dist = torch.distributions.Normal(mean, std)
        raw_actions = torch.atanh(actions/0.01)
        print(raw_actions)
        log_prob = dist.log_prob(raw_actions)
        entropy = dist.entropy()
        return log_prob, entropy, v.squeeze(-1)

    def compute_gae(self, rewards, values, gamma=0.99, lambda_=0.95):
        a = torch.zeros_like(rewards).to(self.device)
        last_a = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] - values[t]
            a[t] = last_a = delta + gamma * lambda_ * last_a
        return a

    def update(self):
        scores = torch.tensor(self.memory.scores[-8:], dtype=torch.float32).to(self.device)
        actions = torch.tensor(self.memory.actions[-8:], dtype=torch.float32).to(self.device)
        rewards = torch.tensor(self.memory.rewards[-8:], dtype=torch.float32).to(self.device)
        log_probs_old = torch.tensor(self.memory.log_probs[-8:], dtype=torch.float32).to(self.device)
        values_old = torch.tensor(self.memory.values[-8:], dtype=torch.float32).to(self.device)

        # Update
        for _ in range(self.k_ep):
            # Evaluate
            log_probs, entropy, values = self.evaluate(scores, actions)

            with torch.no_grad():
                all_values = torch.cat([values, values_old.detach()])

            # GAE
            td_target = rewards + self.gamma * all_values[1:] 
            a = self.compute_gae(rewards, all_values, gamma=self.gamma, lambda_=0.95)
            a = (a - a.mean()) / (a.std() + 1e-8)
            ratio = torch.exp(log_probs - log_probs_old)
            clipped_ratio = torch.clamp(ratio, 0.8, 1.2)

            policy_loss = -torch.min(ratio * a, clipped_ratio * a).mean()
            value_loss = (values - td_target).pow(2).mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory.clear()

if __name__ == "__main__":
    agent = TuningAgent(input_size=1)
    for i in range(10):  
        print(f"ep:{i}")
        action, log_prob, value = agent.act()
        reward = 1.0 
        score = 1.0
        agent.memory.store(score, action, log_prob, reward, value)
        agent.update()
