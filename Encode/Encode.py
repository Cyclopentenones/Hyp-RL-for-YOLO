import torch 
import torch.nn as nn 
from torchvision import transforms
from PIL import Image 
from .Backbone import Darknet53

config = {
    "tiny": {"embed_dim": 192, "num_heads": 3, "num_encoders": 4, "expansion": 2},
    "base": {"embed_dim": 768, "num_heads": 12, "num_encoders": 12, "expansion": 4},
    "large": {"embed_dim": 1024, "num_heads": 16, "num_encoders": 24, "expansion": 4},
}

class Patch_Embedding(nn.Module): 
    def __init__(self, patch_size, embed_dim, dropout, mode="ViT", device='cuda'):
        super(Patch_Embedding, self).__init__() 
        self.embed_dim = embed_dim
        if mode == "ViT": 
            self.projection = nn.Sequential(
                nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size),
                nn.Flatten(2) 
            ).to(device) 
        else: 
            model = Darknet53().to(device)  
            self.projection = nn.Sequential(
                model, 
                nn.Conv2d(1024, embed_dim, kernel_size=patch_size, stride=patch_size),
                nn.Flatten(2)  
            ).to(device)
        self.dropout = nn.Dropout(dropout) 
        self.clstoken = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True) 
        self.position_encoding = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True)   
        
    def forward(self, X): 
        token = self.clstoken.expand(X.shape[0], -1, -1)
        x = self.projection(X).permute(0, 2, 1)  
        x = torch.cat((token, x), dim=1)  
        x += self.position_encoding
        x = self.dropout(x)
        return x 
    
class Duck(nn.Module):
    def __init__(self, patch_size, dropout, size="tiny", mode="ViT"):
        super(Duck, self).__init__()
        param = config[size]
        embed_dim= param["embed_dim"]
        num_heads = param["num_heads"]
        num_encoders = param["num_encoders"]
        expansion = param["expansion"]

        self.patch= Patch_Embedding(patch_size, embed_dim, dropout, mode=mode)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, 
            dim_feedforward=int(embed_dim * expansion), activation="gelu", 
            batch_first=True, norm_first=True
        )
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)

    def forward(self, x):
        x = self.patch(x)
        x = self.encoder_blocks(x)
        return x 
    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Chuyển sang GPU nếu có
    print(f"Using device: {device}")  # Debug xem đang chạy trên gì

    model = Duck(16, 0.1, "tiny", "ViT").to(device)  # Chuyển model lên GPU
    model.eval()  

    image_path = "/Users/nguyenanhminh/Documents/Chovy.jpg"
    image = Image.open(image_path).convert("RGB")
    
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = preprocess(image).unsqueeze(0).to(device)  # Chuyển ảnh lên GPU

    with torch.no_grad():
        output = model(image)
    
    print(output)

if __name__ == "__main__":
    main()
