# Deep Learning Movie Rating Prediction
Collaborative movie recommendation system.  Provided movie ratings by a user, determine the rating that the user would give to every movie in the dataset.


## Preprocessing the Data
Data preprocessing removes a lot of overhead during training, validation, and testing:

Split data into train, validation, and evaluation files.

Normalize ratings from 0-5 stars to be between 0 and 1.

Data is saved in sparse matricies.  This saves lots of memory on disk by only storing ratings and reduces the loading time of data.


## Installation
Install PyTorch for your favorite environment. For conda with nvidia gpus:


conda install -n torch pytorch cudatoolkit=11.1 -c pytorch -c conda-forge

conda install -n torch --file requirements.txt

## For Tuning Hyperparameters
conda install -n torch -c conda-forge optuna


## How To Use
### Preprocessing the data
Only needs to be done once!: python preprocess_data.py

### Training
python train.py [model name]

### Evaluating
python exec.py [model name]

tensorboard --logdir logs
