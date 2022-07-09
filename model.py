from mimetypes import init
from pathlib import Path
from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# torch.Size([4, 1, 3, 3])
# torch.Size([4])
# torch.Size([8, 4, 3, 3])
# torch.Size([8])
# torch.Size([3, 384])
# torch.Size([3])


class Linear_Qnet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        # self.model = nn.Sequential(
        # 1 x 1 x 32 x 24
        self.l1 = nn.Conv2d(1, 8, kernel_size=3)
        # 1 x 8 x 30 x 22
        self.l2 = nn.ReLU()
        self.l3 = nn.MaxPool2d(2, 2)
        # 1 x 8 x 15 x 11
        self.l4 = nn.Conv2d(8, 16, kernel_size=(4, 3))
        # 1 x 16 x 12 x 9
        self.l5 = nn.ReLU()
        self.l6 = nn.MaxPool2d(2, 2)
        # 1 x 16 x 6 x 4
        self.l7 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # 1 x 16 x 1 x 1
        self.l8 = nn.Flatten()
        # 16
        self.l9 = nn.Linear(16, 3)
        # self.l9 = nn.Softmax(dim=1)
        # )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
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

        indx = 1 if len(t_state.shape) == 3 else 0
        t_state = t_state.unsqueeze(indx)
        t_next_state = t_next_state.unsqueeze(indx)
        
        # (n , x)
        t_done = done  # torch.tensor(done, dtype=torch.float)

        if len(t_state.shape) == 3:
            # (1, x)
            t_state = t_state.unsqueeze(0)
            t_next_state = t_next_state.unsqueeze(0)
            t_action = t_action.unsqueeze(0)
            t_reward = t_reward.unsqueeze(0)
            t_done = (done, )

        # 1: Predicted Q values with current state
        pred = self.model(t_state)

        target = pred.clone()
        for i in range(len(t_done)):
            Q_new = t_reward[i]
            if not t_done[i]:
                Q_new += self.gama * torch.max(self.model(t_next_state[i].unsqueeze(0)))

            target[i][torch.argmax(t_action).item()] = Q_new

        # 2: Q_new =  r + gama * max(next_predicted Q value) - only do this is not done
        # pred.clone()
        # pred[argmax(action)] = Q_new

        self.optimiser.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimiser.step()
