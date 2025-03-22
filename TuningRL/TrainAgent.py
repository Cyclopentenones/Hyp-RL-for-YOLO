import sys 
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from TuningEnv import TuningEnv
from TuningAgent import TuningAgent
from Encode.Encode import Duck
from torch.utils.data import DataLoader
from Dataset import TinyImageNetDataset
import torch.nn.functional as F    
def train_agent(agent, env, episodes=100):
    rewards = []
    for episode in range(episodes):
        print(f"Episode {episode + 1}/{episodes}")
        env.reset()
        total_reward = 0
        done = False
        
        while not done: #### Tới khi nào trò chơi kết thúc #####
            action, log_prob, reward, value, score = env.action(agent)  
            agent.memory.store(score, action, log_prob, value, reward)
            total_reward += reward
            done = env.done()
            agent.update() 
        rewards.append(total_reward)
        
        print(f"Episode {episode + 1}/{episodes}: Total Reward = {total_reward:.4f}")
    
    return rewards

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    img_dir = "tiny-imagenet-200/train"
    img_files = []
    labels = []
    num_classes = 0
    for class_folder in os.listdir(img_dir):  
        num_classes += 1
        class_path = os.path.join(img_dir, class_folder, "images")
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_files.append(os.path.join(class_folder, "images", img_name))
                labels.append(class_folder)

    dataset = TinyImageNetDataset(img_files, img_dir, labels)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    all_imgs = []
    targets = []
    print("hello")
    for i, (img, tar, _) in enumerate(dataloader): 
        if(i==10): break
        all_imgs.append(img.to(device))  # Chuyển ảnh lên GPU
        targets.append(tar)   # Chuyển target lên GPU
    targets = torch.cat(targets, dim=0).to(device)  # Convert list thành tensor
    targets = F.one_hot(targets, num_classes=num_classes).to(device)  # Chuyển target thành one-hot
    print("quát đờ phắc")

    encoder = Duck(16, 0.3, 'large', 'ViT-Hybrid').to(device)  # Chuyển encoder lên GPU
    print("kim chi đỏ ao")

    all_imgs = torch.cat(all_imgs, dim=0)  # Convert list thành tensor
    data_processing = encoder(all_imgs)  # Đưa vào encoder (đã ở GPU)

    agent = TuningAgent(input_size=8).to(device)  # Chuyển agent lên GPU
    print("Súc vật")
    env = TuningEnv(data_processing, targets, num_classes=num_classes).to(device)  # Chuyển env lên GPU nếu cần

    print("thơm phức dé hà")
    train_agent(agent, env, episodes=5)

if __name__ == "__main__":
    main()
