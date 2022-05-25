import numpy as np
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, obs_size):
        super().__init__()

        self.fc1 = nn.Linear(obs_size, obs_size)
        self.fc2 = nn.Linear(obs_size, obs_size)
        self.fc3 = nn.Linear(obs_size, obs_size)

        self.relu = nn.ReLU()

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        self.epochs_trained = 0
        self.best_loss = 100

    def forward(self, obs):
        x = self.relu(self.fc1(obs))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def learn(self, obs, target, info=False):
        self.optimizer.zero_grad()
        obs.requires_grad = True
        prediction = self.forward(obs)
        prediction[target == 0] = 0
        loss = torch.sqrt(self.loss_fn(prediction, target))
        loss.backward()
        self.optimizer.step()

        if info:
            n_target = target.count_nonzero()
            # 0. to 1. = 0 stars to 5 stars
            # .5 star = .1
            # 1 star = .2

            nearest_val = ((prediction[target != 0]*10.).round()*.1 == target[target != 0]).sum()    # prediction rounded to the nearest .5 star
            one_star_range = ((target[target != 0] - prediction[target != 0]).abs() < .2).sum()    # prediction within 1 star of real value
            half_star_range = ((target[target != 0] - prediction[target != 0]).abs() < .1).sum()    # prediction within .5 star of real value

            return loss.cpu().detach().numpy(), ((nearest_val/n_target).cpu().detach().numpy(),
                                                (one_star_range/n_target).cpu().detach().numpy(),
                                                (half_star_range/n_target).cpu().detach().numpy())

    def evaluate(self, prediction, target):
        prediction[target == 0] = 0
        loss = torch.sqrt(self.loss_fn(prediction, target))

        n_target = target.count_nonzero()

        nearest_val = ((prediction[target != 0]*10.).round()*.1 == target[target != 0]).sum()    # prediction rounded to the nearest .5 star
        one_star_range = ((target[target != 0] - prediction[target != 0]).abs() < .2).sum()    # prediction within 1 star of real value
        half_star_range = ((target[target != 0] - prediction[target != 0]).abs() < .1).sum()    # prediction within .5 star of real value

        return loss.cpu().detach().numpy(), ((nearest_val/n_target).cpu().detach().numpy(),
                                            (one_star_range/n_target).cpu().detach().numpy(),
                                            (half_star_range/n_target).cpu().detach().numpy())

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epochs_trained': self.epochs_trained
            }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epochs_trained = checkpoint['epochs_trained']

    def load_exec(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.epochs_trained = checkpoint['epochs_trained']
