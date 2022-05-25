import copy
import gc

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz


for dataset in ['100k', '1m', '10m', '20m']:
	movies = pd.read_csv('data/'+dataset+'/movies.csv', usecols=['movieId'], encoding='latin-1').dropna()
	movies['movie_idx'] = movies.index

	movies['rating_count'] = 0.

	for mode in ['train', 'valid', 'test']:
		ratings = pd.read_csv('data/'+dataset+'/'+mode+'_ratings.csv',
								usecols=['userId','movieId','rating'],
								dtype={'userId': np.int32, 'movieId': np.int32, 'rating': np.float32}).dropna()

		for userId_adj, userId in enumerate(ratings['userId'].unique()):
			ratings['userId'] = np.where(ratings['userId'] == userId, userId_adj, ratings['userId'])
		ratings['rating'] /= 5.0

		ratings = ratings.merge(movies, on=['movieId'], how='left').drop(columns='movieId')

		rating_count = ratings['movie_idx'].value_counts(dropna=False, sort=False)
		movies['rating_count'] = movies['rating_count'].add(rating_count, fill_value=0.)

		sparse_mat = csr_matrix((ratings['rating'].values, (ratings.userId, ratings.movie_idx)), dtype=np.float32)
		save_npz('data/'+dataset+'/'+mode+'_ratings.npz', sparse_mat)

		del ratings
		del sparse_mat
		gc.collect()

	movies = movies.drop('movieId', axis=1)
	highest_rating_count = movies['rating_count'].max()
	movies['rating_count'] = movies['rating_count'].divide(highest_rating_count)
	movies.to_csv('data/'+dataset+'/movies_data.csv', index=False)

	del movies
	gc.collect()
