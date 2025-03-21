import torch
from TuningEnv import TuningEnv
from TuningAgent import TuningAgent
from Encode.Encode import Duck
from Data import  dataloader 

def train_agent(agent, env, episodes=100):
    rewards = []
    for episode in range(episodes):
        env.reset()
        total_reward = 0
        done = False
        
        while not done: #### Tới khi nào trò chơi kết thúc#####
            action, log_prob, reward, value, score = env.action(agent)  
            agent.memory.store(score, action, log_prob, value, reward)
            total_reward += reward
            done = env.done()
        
        agent.update() 
        rewards.append(total_reward)
        
        print(f"Episode {episode + 1}/{episodes}: Total Reward = {total_reward:.4f}")
    
    return rewards

def main():
    # Load data
    data = Duck(16, 0.3, 'large', 'ViT-Hybrid')(dataloader("Link"))
    
    agent = TuningAgent(input_size=8)
    env = TuningEnv(data)
    
    train_agent(agent, env, episodes=100)
    
if __name__ == "__main__":
    main()
