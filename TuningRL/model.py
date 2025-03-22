import torch
import torch.nn as nn

class Net(nn.Module): 
    def __init__(self, input_size): 
        super(Net, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=64, num_layers=5)

        self.fc = nn.Sequential(
            nn.Linear(64, 32), 
            nn.ReLU()
        )  

        self.critic = nn.Sequential(
            nn.Linear(32, 16), 
            nn.ReLU(), 
            nn.Linear(16, 1)  
        )
        self.mean = nn.Linear(32, 1) 
        self.log_std = nn.Linear(32, 1)  

    def forward(self, x):
        x = x.clone().detach().requires_grad_(True).float().unsqueeze(0).unsqueeze(-1)
        lstm_out, _ = self.lstm(x)  
        x = lstm_out[:, -1, :]  
        
        x = self.fc(x)  

        mean = self.mean(x) 
        log_std = self.log_std(x)  # Log standard deviation
        v = self.critic(x)  

        return mean, log_std, v  
