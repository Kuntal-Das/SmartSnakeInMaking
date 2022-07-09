from mimetypes import init
from pathlib import Path
from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_Qnet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_path = os.path.join(model_folder_path, file_name)

        torch.save(self.state_dict(), file_path)


class QTrainer:
    def __init__(self, model, lr, gama):
        self.model = model
        self.lr = lr
        self.gama = gama
        self.optimiser = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        t_state = torch.tensor(state, dtype=torch.float)
        t_next_state = torch.tensor(next_state, dtype=torch.float)
        t_action = torch.tensor(action, dtype=torch.float)
        t_reward = torch.tensor(reward, dtype=torch.float)
        # (n , x)
        t_done = done  # torch.tensor(done, dtype=torch.float)

        if len(t_state.shape) == 1:
            # (1, x)
            t_state = torch.unsqueeze(state, 0)
            t_next_state = torch.unsqueeze(t_next_state, 0)
            t_action = torch.unsqueeze(t_action, 0)
            t_reward = torch.unsqueeze(t_reward, 0)
            t_done = (done, )

        # 1: Predicted Q values with current state
        pred = self.model(t_state)

        target = pred.clone()
        for i in range(len(done)):
            Q_new = reward[i]
            if not done[i]:
                Q_new += self.gama * torch.max(self.model(next_state[i]))

            target[i][torch.argmax(action).item] = Q_new

        # 2: Q_new =  r + gama * max(next_predicted Q value) - only do this is not done
        # pred.clone()
        # pred[argmax(action)] = Q_new

        self.optimiser.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimiser.step()
