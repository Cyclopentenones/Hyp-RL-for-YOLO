import torch
import torch.nn as nn

class Net(nn.Module): 
    def __init__(self, input_size): 
        super(Net, self).__init__()

        # LSTM block: Ghi nhớ khoảng 8 epoch trước
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=64,  # 64 hoặc 128 
                            num_layers=2
                            )

        # Fully Connected Layer sau LSTM
        self.fc = nn.Sequential(
            nn.Linear(64, 32), 
            nn.ReLU()
        )  

        # Critic Network (Value Function)
        self.critic = nn.Sequential(
            nn.Linear(32, 16), 
            nn.ReLU(), 
            nn.Linear(16, 1)  # Giá trị V (scalar)
        )

        # Actor Network (Beta Distribution: Alpha & Beta)
        self.alpha = nn.Sequential(
            nn.Linear(32, 3),  # 3 actions (ví dụ: Learning Rate, Momentum, Weight Decay)
            nn.Softplus()  # Softplus để đảm bảo alpha > 0
        )

        self.beta = nn.Sequential(
            nn.Linear(32, 3), 
            nn.Softplus()
        )

    def forward(self, x): 
        lstm_out, _ = self.lstm(x)  # Output có shape (batch, seq_len, hidden_size)
        x = lstm_out[:, -1, :]  # Lấy output của bước thời gian cuối cùng (seq_len cuối)
        
        x = self.fc(x)  # Fully connected
        v = self.critic(x)  # Giá trị V (Critic)
        alpha = self.alpha(x) + 1  # Alpha > 1
        beta = self.beta(x) + 1  # Beta > 1
        
        return alpha, beta, v
