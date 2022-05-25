import os
import signal
import sys
import gc

import numpy as np
import pandas as pd
import optuna
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sparsedataset import SparseDataset
from model import Model



dataset = '10m'
datasets = {'100k': 9742, '1m': 3883, '10m': 10680, '20m': 27278}


def objective(trial):
    model = Model(datasets[dataset])
    model.cuda(0)

    lr = trial.suggest_loguniform("lr", 1e-6, 1e-2)
    model.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    nearest_val = 0.

    for epoch in range(100):
        model.train()
        train_ds = SparseDataset('data/'+dataset+'/train_ratings.npz', datasets[dataset])
        dataloader = DataLoader(train_ds, batch_size=32, num_workers=0, shuffle=True)

        for obs, target in dataloader:
            obs = obs.cuda(0)
            target = target.cuda(0)

            model.learn(obs, target)

        del train_ds
        gc.collect()


        model.eval()
        valid_ds = SparseDataset('data/'+dataset+'/valid_ratings.npz', datasets[dataset])
        dataloader = DataLoader(valid_ds, batch_size=1, num_workers=0)

        acc_list = []
        with torch.no_grad():
            for obs, target in dataloader:
                obs = obs.cuda(0)
                target = target.cuda(0)

                _, acc = model.evaluate(model.forward(obs), target)
                acc_list.append(acc)

        del dataloader
        del valid_ds
        gc.collect()

        nearest_val = 100. * np.mean(acc_list[0])
        # one_star_range = 100. * np.mean(acc_list[1])
        # half_star_range = 100. * np.mean(acc_list[2])

        trial.report(nearest_val, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return nearest_val

study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=100)

pruned_trials = study.get_trials(deepcopy=False, states=[optuna.structs.TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[optuna.structs.TrialState.COMPLETE])


print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
