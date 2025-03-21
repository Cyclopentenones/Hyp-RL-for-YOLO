import torch
import torch.nn as nn
class Net(nn.Module): 
    def __init__(self, input_size): 
        super(Net, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=64, num_layers=2)

        self.fc = nn.Sequential(
            nn.Linear(64, 32), 
            nn.ReLU()
        )  

        self.critic = nn.Sequential(
            nn.Linear(32, 16), 
            nn.ReLU(), 
            nn.Linear(16, 1)  
        )


        #using Beta distribution 
        self.alpha = nn.Sequential(
            nn.Linear(32, 1),  
            nn.Softplus()  
        )

        self.beta = nn.Sequential(
            nn.Linear(32, 1), 
            nn.Softplus()
        )

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        lstm_out, _ = self.lstm(x)  
        x = lstm_out[:, -1, :]  
        
        x = self.fc(x)  

        alpha = self.alpha(x) 
        beta = self.beta(x)  
        v = self.critic(x)  

        return alpha, beta, v
