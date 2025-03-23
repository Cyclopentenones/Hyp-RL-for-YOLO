import os
import torch
import cv2
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class_mapping = {}  
wnid_to_name = {} 

with open(r"tiny-imagenet-200\words.txt", "r") as f:
    for line in f.readlines():
        wnid, name = line.strip().split("\t", 1)
        wnid_to_name[wnid] = name

with open(r"tiny-imagenet-200/wnids.txt", "r") as f:
    labels = []
    for idx, line in enumerate(f.readlines()):
        wnid = line.strip()
        class_mapping[wnid] = idx

class TinyImageNetDataset(Dataset):
    def __init__(self, img_files,  img_dir, labels):
        self.img_files = img_files
        self.labels = labels
        self.class_names = [wnid_to_name[label] for label in labels]
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_files[index])
        label_index = class_mapping[self.labels[index]]
        class_name = self.class_names[index]
        assert os.path.exists(img_path), f"Image file not found: {img_path}"

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)

        return image, torch.tensor(label_index, dtype=torch.long), class_name
    

def main():
    img_dir = "tiny-imagenet-200/train"
    img_files = []
    labels = []
    for class_folder in os.listdir(img_dir):  
        class_path = os.path.join(img_dir, class_folder, "images")
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_files.append(os.path.join(class_folder, "images", img_name))
                labels.append(class_folder)

    dataset = TinyImageNetDataset(img_files, img_dir, labels)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True) 

    # In thử 5 ảnh đầu tiên
    for i, (image, label, class_name) in enumerate(dataloader):
        if i >= 5:
            break

        label_int = label
        class_wnid = list(class_mapping.keys())[label_int]

        print(f"Image {i + 1}: {dataset.img_files[i]}")
        print(f"Label ID: {label_int}")
        print(f"Class Wnid: {class_wnid}")
        print(f"Class Name: {class_name[0]}")

        # Hiển thị ảnh
        plt.imshow(image.squeeze().permute(1, 2, 0))
        plt.title(f"Label: {class_wnid} ({label_int})")
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    main()
