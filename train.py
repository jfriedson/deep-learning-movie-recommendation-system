import os
import signal
import sys
import gc

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sparsedataset import SparseDataset
from model import Model


# default dataset
dataset = '10m'
datasets = {'100k': 9742, '1m': 3883, '10m': 10681, '20m': 27278}

# command line arguments
# py train.py [dataset] [model name]
if len(sys.argv) == 3:
    if str(sys.argv[1]) not in datasets:
        sys.exit('specified dataset does not exist')
    dataset = str(sys.argv[1])
    model_name = str(sys.argv[2])
elif len(sys.argv) == 2:
    if str(sys.argv[1]) not in datasets:
        sys.exit('specified dataset does not exist')
    dataset = str(sys.argv[1])
    model_name = 'recsys_'+dataset
else:
    model_name = 'recsys_'+dataset
model_dir = './nets/' + model_name


# tensorboard initialization
train_tb = SummaryWriter('logs/'+model_name+'/training')
valid_tb = SummaryWriter('logs/'+model_name+'/validation')

def ctrlc_handler(sig, frame):
    train_tb.flush()
    train_tb.close()
    valid_tb.flush()
    valid_tb.close()
    sys.exit(0)
signal.signal(signal.SIGINT, ctrlc_handler)


# create/load model
model = Model(datasets[dataset])
model.cuda(0)

if os.path.isdir(model_dir):
    models = [x for x in os.listdir(model_dir)]

    if 'last.zip' in models:
        model.load(model_dir + '/last.zip')
        model.cuda(0)
        print('loaded previously trained model \'' + model_name + '\' at epoch ' + str(model.epochs_trained))
    elif 'best.zip' in models:
        model.load(model_dir + '/best.zip')
        model.cuda(0)
        print('loaded previously trained model \'' + model_name + '\' at epoch ' + str(model.epochs_trained))

else:
    os.makedirs(model_dir)
    print('created model \'' + model_name + '\' at directory ' + model_dir)


def train(epochs):
    model.train()

    train_ds = SparseDataset('data/'+dataset+'/train_ratings.npz', datasets[dataset])
    dataloader = DataLoader(train_ds, batch_size=32, num_workers=0, shuffle=True)

    # train for 5 epochs
    for _ in range(epochs):
        loss_list = []
        acc_list = [[],[],[]]
        for obs, target in tqdm(dataloader):
            obs = obs.cuda(0)
            target = target.cuda(0)

            loss, acc = model.learn(obs, target, True)
            loss_list.append(loss)
            acc_list[0].append(acc[0])
            acc_list[1].append(acc[1])
            acc_list[2].append(acc[2])

        model.epochs_trained += 1

        loss = np.mean(loss_list)
        print('training loss=', np.mean(loss_list))

        nearest_val = 100. * np.mean(acc_list[0])
        one_star_range = 100. * np.mean(acc_list[1])
        half_star_range = 100. * np.mean(acc_list[2])
        print('training accuracy')
        print('nearest value=', nearest_val)
        print('one star range=', one_star_range)
        print('half star range=', half_star_range)

        train_tb.add_scalar('loss', loss, model.epochs_trained)
        train_tb.add_scalar('accuracy/nearest value', nearest_val, model.epochs_trained)
        train_tb.add_scalar('accuracy/1 star range accuracy', one_star_range, model.epochs_trained)
        train_tb.add_scalar('accuracy/.5 star range accuracy', half_star_range, model.epochs_trained)
        train_tb.flush()

        if model.epochs_trained % 100 == 0:
            model.save(model_dir + '/' + str(model.epochs_trained) + '.zip')
            print('saved model \'' + model_name + '\' at epoch ' + str(model.epochs_trained))
        if model.epochs_trained % 10 == 0:
            model.save(model_dir + '/last.zip')

    del train_ds
    gc.collect()


def validate():
    model.eval()

    valid_ds = SparseDataset('data/'+dataset+'/valid_ratings.npz', datasets[dataset])
    dataloader = DataLoader(valid_ds, batch_size=1, num_workers=0)

    loss_list = []
    acc_list = [[],[],[]]
    with torch.no_grad():
        for obs, target in tqdm(dataloader):
            obs = obs.cuda(0)
            target = target.cuda(0)

            loss, acc = model.evaluate(model.forward(obs), target)
            loss_list.append(loss)
            acc_list[0].append(acc[0])
            acc_list[1].append(acc[1])
            acc_list[2].append(acc[2])

    loss = np.mean(loss_list)
    print('validation loss=', loss)

    nearest_val = 100. * np.mean(acc_list[0])
    one_star_range = 100. * np.mean(acc_list[1])
    half_star_range = 100. * np.mean(acc_list[2])
    print('validation accuracy')
    print('nearest value=', nearest_val)
    print('one star range=', one_star_range)
    print('half star range=', half_star_range)

    valid_tb.add_scalar('loss', loss, model.epochs_trained)
    valid_tb.add_scalar('accuracy/nearest value', nearest_val, model.epochs_trained)
    valid_tb.add_scalar('accuracy/1 star range accuracy', one_star_range, model.epochs_trained)
    valid_tb.add_scalar('accuracy/.5 star range accuracy', half_star_range, model.epochs_trained)
    valid_tb.flush()

    if loss < model.best_loss:
        model.best_loss = loss
        model.save(model_dir + '/best.zip')
        print('saved best model')

    del dataloader
    del valid_ds
    gc.collect()

# train and evaluate 100 times
for _ in range(100):
    train(1 if model.epochs_trained < 5 else 5)
    validate()

train_tb.flush()
train_tb.close()
valid_tb.flush()
valid_tb.close()
