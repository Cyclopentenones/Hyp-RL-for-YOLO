import torch.nn as nn

#CNNBlock
class CNNBlock(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bn_act=True): 
        super(CNNBlock, self).__init__()
        self.Conv2d = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride, 
            padding=padding
        )
        self.BatchNorm2d = nn.BatchNorm2d(num_features=out_channels)
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.1)
        self.bn_act = bn_act 
    
    def forward(self, x): 
        if self.bn_act: 
            return self.LeakyReLU(self.BatchNorm2d(self.Conv2d(x)))
        else: 
            return self.LeakyReLU(self.Conv2d(x)) 
        
#ResidualBlock
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, num_repeats=1): 
        super(ResidualBlock, self).__init__()
        self.CNNBlock1 = CNNBlock(in_channels, in_channels//2, kernel_size=1, stride=1, padding=0) 
        self.CNNBlock2 = CNNBlock(in_channels//2, in_channels, kernel_size=3, stride=1, padding=1)
        self.num_repeats= num_repeats 

    def forward(self, x): 
        for _ in range(self.num_repeats): 
            x = x + self.CNNBlock2.forward(self.CNNBlock1.forward(x))
        return x

#Darknet53 
class Darknet53(nn.Module): 
    def __init__(self): 
        super(Darknet53, self).__init__()
        self.backbone = nn.Sequential(
            CNNBlock(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1), 
            CNNBlock(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), 
            ResidualBlock(in_channels=64), 
            CNNBlock(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1), 
            ResidualBlock(in_channels=128, num_repeats=2),
            CNNBlock(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1), 
            ResidualBlock(in_channels=256, num_repeats=8),
            CNNBlock(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            ResidualBlock(in_channels=512, num_repeats=8),
            CNNBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1), 
            ResidualBlock(in_channels=1024, num_repeats=4)
        )

    def forward(self, X): 
        return self.backbone(X) 
