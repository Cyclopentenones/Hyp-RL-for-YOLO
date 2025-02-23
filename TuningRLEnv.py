import torch
import torch.nn as nn
import numpy as np

ACTION_SPACE = [1e-2, 1e-3, 1e-4, 1e-5]

class TuningRLEnv(nn.Module): 
    def __init__(self, hyperparameter): 
        super(TuningRLEnv, self).__init__()
        self.action_space = ACTION_SPACE
        self.hyperparameter = hyperparameter
        self.surrmodel = SurrModel(hyperparameter)  # Mô hình dự đoán metrics
        self.state = None  # Trạng thái hiện tại của môi trường
    
    def reset(self):
        self.state = self._get_initial_state()
        return self.state
    
    def step(self, action_idx):
        action = self.action_space[action_idx] 
        self._apply_action(action) 

        next_state = self.surrmodel.predict()  
        reward = self.get_reward(next_state)
        done = self._check_done(next_state) 

        self.state = next_state
        return next_state, reward, done

    def get_reward(self, metrics):
        mAP, IOU, F1, FPS, Robustness = metrics
        reward = mAP + 0.5 * IOU + 0.5 * F1 + 0.2 * FPS + 0.3 * Robustness
        return reward
    
    def _apply_action(self, action):
        self.hyperparameter['lr'] *= action  # Giả sử action là tỷ lệ thay đổi learning rate
    
    def _get_initial_state(self):
        return np.array([0.5, 0.5, 0.5, 30, 0.5])  # Giá trị giả định cho metrics

    def _check_done(self, state):
        return False 
