# Deep Learning Movie Rating Prediction
Predict the ratings a user will give to movies they have not rated based solely on the ratings they have provided for other movies.  Written in python with the pytorch library.  Works on the newest MovieLens dataset.


## Architecture
This neural network architechture solves issues I found in other machine learning methods:

Clustering algorithms, such as k-means, limit the scope of recommendations because users are grouped.

Deep learning with embedding layers must be retrained every time a new user is added or a user adds a new rating.

My architechture doesn't group users or their interests and it does not need to be retrained for new users or new ratings.


## Nice-to-Haves
Preprocessing removes a lot of overhead during training, validation, and testing:

Split data into train, validation, and evaluation files.

Normalize ratings from 0-5 stars to be between 0 and 1.

Save data as sparse matricies.  This saves lots of memory on disk and reduces the loading time of data.


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
