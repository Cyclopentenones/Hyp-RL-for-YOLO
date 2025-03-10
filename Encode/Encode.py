#pipeline: 
#Hoàn thành với ViT Thông thường
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
from Backbone import Darknet53
class Patch_Embedding(nn.Module): 
    def __init__(self, patches_size, embed_dim, dropout, mode="ViT"):
        super(Patch_Embedding, self).__init__() 
        # Trích xuất đặc trưng tốt hơn (I think so:v) + Flatten 
        if mode == "ViT": 
            self.projection = nn.Sequential(
                nn.Conv2d(3, embed_dim, kernel_size=patches_size, stride=patches_size),
                nn.Flatten(2) 
            ) # -> B, C, numP, numP -> B, C, numP*numP
        elif mode == "ViT_Hybrid": 
            model = Darknet53()
            self.projection =

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



class Duck(nn.Module):
    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels, num_heads, num_encoders, expansion, num_classes):
        super().__init__()
        self.patch= Patch_Embedding(embed_dim, patch_size, num_patches, dropout, in_channels)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, dim_feedforward=int(embed_dim*expansion), activation="gelu", batch_first=True, norm_first=True)
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)

        self.mlp = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes)
        ) # nhớ thử với Linear function 

    def forward(self, x):
        x = self.patch(x)
        x = self.encoder_blocks(x)
        x = self.mlp(x[:, 0, :]) 
        return x 