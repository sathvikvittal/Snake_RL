import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class QNet(nn.Module):
    def __init__(self, input_size,hidden_size,output_size):
        super().__init__()
        self.lin1 = nn.Linear(input_size,hidden_size)
        self.lin2 = nn.Linear(hidden_size,output_size)
    
    def forward(self,X):
        X = F.relu(self.lin1(X))
        X = self.lin2(X)
        return X
    
    def save_model(self,file_name='model.pth'):
        folder = './model'
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        file_name = os.path.join(folder,file_name)
        torch.save(self.state_dict(),file_name)


class QTrainer:
    def __init__(self,model,lr,gam):
        self.model = model
        self.lr = lr
        self.gam = gam
        self.optim = optim.Adam(model.parameters(),lr=self.lr)
        self.loss = nn.MSELoss()
    
    def train_step(self, state, action, reward, next_state, end):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state,0)
            action = torch.unsqueeze(action,0)
            reward = torch.unsqueeze(reward,0)
            next_state = torch.unsqueeze(next_state,0)
            end = (end,)

        pred = self.model(state)
        tar = pred.clone()

        for ind in range(len(state)):
            Q_new = reward[ind]

            if not end:
                Q_new = reward[ind] + self.gam * torch.max(self.model(next_state[ind]))
            
            tar[ind][torch.argmax(action).item()] = Q_new
        
        self.optim.zero_grad()
        loss = self.loss(pred,tar)
        loss.backward()

        self.optim.step()