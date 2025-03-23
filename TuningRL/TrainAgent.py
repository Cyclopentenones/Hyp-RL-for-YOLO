import sys 
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm

from TuningEnv import TuningEnv
from TuningAgent import TuningAgent
from Dataset import TinyImageNetDataset

def train_worker(rank, data, targets, num_classes, episodes=5):
    """Hàm huấn luyện trong mỗi tiến trình."""
    device = f"cuda:{rank % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu"
    print(f"Process {rank} using {device}")
    agent = TuningAgent(input_size=1).to(device)
    env = TuningEnv(data, targets, num_classes).to(device)

    rewards = []
    for episode in range(episodes):
        env.reset()
        total_reward = 0
        done = False

        # Thanh tiến trình cho mỗi episode
        with tqdm(total=500, desc=f"[Process {rank}] Episode {episode + 1}/{episodes}", leave=True) as progress_bar:
            while not done:
                action, log_prob, reward, value, score = env.action(agent)
                agent.memory.store(score, action, log_prob, value, reward)
                total_reward += reward
                done = env.done()
                agent.update()

                progress_bar.update(1)
                progress_bar.set_postfix({"Reward": f"{total_reward:.4f}"})

        rewards.append(total_reward)

    print(f"Process {rank} finished training.")
    return rewards

import torch
import gc

torch.cuda.empty_cache()  # Giải phóng cache GPU
gc.collect()              # Dọn dẹp bộ nhớ Python
def main():

    mp.set_start_method("spawn", force=True)  # Khởi tạo multiprocessing

    # Xác định thiết bị
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True

    # Load dữ liệu
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

    # Load dataset
    dataset = TinyImageNetDataset(img_files, img_dir, labels)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    # Lấy dữ liệu mẫu
    all_imgs, targets = [], []
    for i, (img, tar, _) in enumerate(dataloader):
        all_imgs.append(img.to(device))
        targets.append(tar)
    
    targets = torch.cat(targets, dim=0).to(device)
    all_imgs = torch.cat(all_imgs, dim=0)

    # Số môi trường song song
    num_processes = min(4, torch.cuda.device_count())  # Sử dụng tối đa 4 GPU

    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train_worker, args=(rank, all_imgs, targets, num_classes))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()  # Đợi tất cả tiến trình hoàn thành

if __name__ == "__main__":
    main()
