import sys
import gc

import numpy as np
import pandas as pd


dataset = '10m'
datasets = {'100k': 9742, '1m': 3883, '10m': 10681, '20m': 27278}

# create/load model
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


# load data
movies = pd.read_csv('data/'+dataset+'/movies.csv', encoding='latin-1').dropna().drop(columns=['movieId'])


# most popular movies of all time
ratings_grouped = ratings.groupby('movieId')['rating']
movie_popularity = pd.DataFrame(ratings_grouped.count() * ratings_grouped.mean())
most_popular_movies = movie_popularity.sort_values('rating', ascending=False)

most_popular_idxs = pd.DataFrame({'movieId': most_popular_movies.index[:]})
most_popular_movies = most_popular_idxs.merge(movies, on='movieId')

print('Most popular movies of all time')
print(most_popular_movies[['title', 'genres']].head())


################################################################################################################
# This code is not part of the AI. It helps analyze biases within the data.

# popular recently released movies (year of 2018)
print('Popular recently released movies')
year_released = movies.copy()
year_released['year'] = year_released['title'].str[-5:-1]
# year_released['year'] = [int(x) if x.isnumeric() else 0 for x in year_released['year']]
# year_released = year_released.sort_values('year', ascending=False)

year_released = year_released[(year_released['year'] == '2017') | (year_released['year'] == '2018')]
popular_recent_movies = most_popular_idxs.merge(year_released, on='movieId')

print(popular_recent_movies['title'].head())

# popular movies by genre
genre_dummies = movies['genres'].str.get_dummies()

genres = list(genre_dummies.columns)
print(genres)

while(True):
    genre = input('choose genre: ')
    if genre in genres:
        break

genre_dummies = movies.join(genre_dummies)

genre_similar = genre_dummies[genre_dummies[genre] == 1]
genre_similar_popular = most_popular_idxs.merge(genre_similar, on='movieId')

print('Most popular ' + genre + ' movies')
print(genre_similar_popular[['title', 'genres']].head())

# popular recently released movies by genre
genre_similar_popular_recent = popular_recent_movies[['movieId', 'year']].merge(genre_similar, on=['movieId'])

print('Most popular recently released ' + genre + ' movies')
print(genre_similar_popular_recent[['title', 'year', 'genres']].head())

################################################################################################################


# predict movie ratings for user using neural network
import os
import signal

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from sparsedataset import SparseDataset
from model import Model


model = Model(datasets[dataset])
model.cuda(0)

if os.path.isdir(model_dir):
    models = [x for x in os.listdir(model_dir)]

    if 'best.zip' in models:
        model.load_exec(model_dir + '/best.zip')
        model.cuda(0)
        print('testing best model \'' + model_name + '\' at epoch ' + str(model.epochs_trained))
    elif 'last.zip' in models:
        model.load_exec(model_dir + '/last.zip')
        model.cuda(0)
        print('testing last model \'' + model_name + '\' at epoch ' + str(model.epochs_trained))
    else:
        print('no previously trained model is named \'' + model_name + '\'')
        sys.exit(0)

else:
    print('directory \'' + model_dir + '\' does not exist. exiting')
    sys.exit(0)


model.eval()


from scipy.sparse import csr_matrix, load_npz

sparse_mat = load_npz('data/'+dataset+'/test_ratings.npz')

user_id = 27
user_ratings = sparse_mat.getrow(user_id).A
user_ratings.resize(datasets[dataset])

idxs = user_ratings.nonzero()[0]
print('number of ratings by user', user_id, len(idxs))
rated_movies_names = movies.iloc[idxs]

rated_df = pd.DataFrame(user_ratings*5., columns = ['rating'])
rated_movies = rated_movies_names.join(rated_df, how='left')
print(rated_movies)

user_ratings = torch.from_numpy(user_ratings).cuda(0)
prediction = model.forward(user_ratings).cpu().detach().numpy()

pred_df = pd.DataFrame(prediction*5., columns = ['rating'])
# rated_movies = rated_movies_names.join(pred_df, how='left')
# print(rated_movies)

print('top rated predictions')
rated_movies = movies.join(pred_df, how='left').sort_values('rating', ascending=False).iloc[0:10]
rated_movies['rated_by_user'] = rated_movies.index.isin(idxs)
print(rated_movies)


# average rating of genres
genre_dummies = movies['genres'].str.get_dummies()

genres = list(genre_dummies.columns)

rated_genres = rated_df.join(genre_dummies.iloc[idxs])
# pred_genres = pred_df.join(genre_dummies)

import math
import operator

print('\n average rating by genre - user ratings')
genre_list = {}
for genre in genres:
    genre_list[genre] = rated_genres[rated_genres[genre] == 1]['rating'].mean()
for genre, avg_rating in sorted(genre_list.items(), key=lambda x: x[1], reverse=True):
    if not math.isnan(avg_rating):
        print(genre, avg_rating)



################################################################################################################
# This code 
# for genre in genre_list.sort_values:
# print(genre, rated_genres[rated_genres[genre] == 1]['rating'].mean())
#
# print('\n average rating by genre - predicted ratings')
# for genre in genres.sort_values:
#     print(genre, pred_genres[pred_genres[genre] == 1]['rating'].mean())

# del prediction
# del user_ratings
# del sparse_mat
# gc.collect()


# test_ds = SparseDataset('data/'+dataset+'/test_ratings.npz', datasets[dataset])
# dataloader = DataLoader(test_ds, batch_size=1, num_workers=0)
#
# loss_list = []
# acc_list = [[],[],[]]
# with torch.no_grad():
#     for obs, target in tqdm(dataloader):
#         obs = obs.cuda(0)
#         target = target.cuda(0)
#
#         loss, acc = model.evaluate(model.forward(obs), target)
#         loss_list.append(loss)
#         acc_list[0].append(acc[0])
#         acc_list[1].append(acc[1])
#         acc_list[2].append(acc[2])
#
# loss = np.mean(loss_list)
# print('validation loss=', loss)
#
# nearest_val = 100. * np.mean(acc_list[0])
# one_star_range = 100. * np.mean(acc_list[1])
# half_star_range = 100. * np.mean(acc_list[2])
# print('validation accuracy')
# print('nearest value=', nearest_val)
# print('one star range=', one_star_range)
# print('half star range=', half_star_range)
