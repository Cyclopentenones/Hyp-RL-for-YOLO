#pipeline: 
#Hoàn thành với ViT Thông thường
#Hoàn thành với ViT-Hybrid 
"""
input: KITTI dataset -> encode: vector 
model encode có thể thử nghiệm: 
*Resnet-18 
Resnet-50
MobileNetV3
*EfficentNetB0
ViT-Tny 
*ViT-Hybrid 




encode đưa qua CNN mỏng để tạo state cho RL 
episodes, dự kiến 5000-10000
"""


"""
RL Agent: DQN, PPO, Bayes + RL. Dãy hành động là vector [+-1e-2, +-1e-3, ..., +-1e-6] chọn hành động thay đổi các siêu tham số 
RL Agent net: Theo paper thì sẽ thiết kế theo dạng LSTM (Có thể thử với Linear, RNN) 
State: Có thể là sự thay đổi của gradient hoặc một cái gì đó,.... 
Tuning hyp: So sánh giữa việc sử dụng  RL + Adam, RL thuần, Adam thuần 

Mục tiêu dự kiến: Giảm được số epoch mà một model Yolo cần để hội tụ
"""

"""
Pipeline:  Vector encode -> CNN trích xuất đặc trưng (env) -> State -> RL Agent -> Action -> Env 
Train YOLO: Dataset (Có  thể sử dụng encode hoặc không) -> train YOLO (dùng  RL Để tunng hyper mỗi epoch)
"""

"""
Encode: Vector thể hiện đặc trưng của cả dataset. 

"""

# Huớng khác là sử dụng pipeline train song song giữa RL với YOLO nhưng chưa nghĩ ra dc pipeline phù hợp 
import torch 
import torch.nn as nn 
from torchvision import transforms
from  PIL import Image 
from Backbone import Darknet53


class Patch_Embedding(nn.Module): 
    def __init__(self, patch_size, embed_dim, dropout, mode="ViT"):
        super(Patch_Embedding, self).__init__() 
        self.embed_dim = embed_dim
        # Trích xuất đặc trưng tốt hơn (I think so:v) + Flatten 
        if mode == "ViT": 
            self.projection = nn.Sequential(
                nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size),
                nn.Flatten(2) 
            ) # -> B, Embed_dim, numP, numP -> B, Embed_dim, numP*numP
        else: 
            model = Darknet53() # B, 1024, H, W 
            self.projection = nn.Sequential(
                model, 
                nn.Conv2d(1024, embed_dim, kernel_size=patch_size, stride=patch_size), # B, Embed_din, numP, numP
                nn.Flatten(2) # B, Embed_dim, numP*numP
            )
        self.dropout = nn.Dropout(dropout) 
        self.clstoken = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True) 
        self.position_encoding = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True)   
        
    def forward(self, X): 
        token = self.clstoken.expand(X.shape[0], -1, -1)
        x = self.projection(X).permute(0, 2, 1) # Transformer required shape B, numP*numP, C
        x = torch.cat((token, x), dim=1)  
        x += self.position_encoding
        x = self.dropout(x)
        return x 


config = {
            "tiny": {"embed_dim": 192, "num_heads": 3, "num_encoders": 4, "expansion": 2},
            "base": {"embed_dim": 768, "num_heads": 12, "num_encoders": 12, "expansion": 4},
            "large": {"embed_dim": 1024, "num_heads": 16, "num_encoders": 24, "expansion": 4},
}
class Duck(nn.Module):
    def __init__(self, patch_size, dropout, mode):
        super(Duck, self).__init__()
        param = config[mode]
        embed_dim= param["embed_dim"]
        num_heads = param["num_heads"]
        num_encoders = param["num_encoders"]
        expansion = param["expansion"]

        self.patch= Patch_Embedding(patch_size, embed_dim, dropout, mode="ViT-Hybrid")
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, dim_feedforward=int(embed_dim*expansion), activation="gelu", batch_first=True, norm_first=True)
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)

    def forward(self, x):
        x = self.patch(x)
        x = self.encoder_blocks(x)
        return x 
    
def main():
    model = Duck(16, 0.1, "tiny")
    model.eval() 
    image_path = "/Users/nguyenanhminh/Documents/Chovy.jpg"
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
    
    print(output)

"""Fix mệt chết mẹ, ảnh dưới 512 thì thầy chịu"""
if __name__ == "__main__":
    main()
