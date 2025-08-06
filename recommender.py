import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    ratings = pd.read_csv("ml-100k/u.data", sep='\t', names=['user', 'item', 'rating', 'timestamp'])
    movies = pd.read_csv("ml-100k/u.item", sep='|', encoding='latin-1', header=None, usecols=[0, 1], names=['item', 'title'])
    ratings = ratings.drop(columns='timestamp').merge(movies, on='item')
    return ratings

def create_matrix(ratings):
    return ratings.pivot_table(index='user', columns='title', values='rating').fillna(0)

def train_similarity(user_item_matrix):
    item_sim = cosine_similarity(user_item_matrix.T)
    user_sim = cosine_similarity(user_item_matrix)
    return pd.DataFrame(item_sim, index=user_item_matrix.columns, columns=user_item_matrix.columns), \
           pd.DataFrame(user_sim, index=user_item_matrix.index, columns=user_item_matrix.index)

def item_based_recommend(user_id, user_item_matrix, item_sim_df, n=5):
    user_ratings = user_item_matrix.loc[user_id]
    scores = {}

    for item in user_item_matrix.columns:
        if user_ratings[item] == 0:
            sim_scores = item_sim_df[item]
            weighted_sum = np.dot(sim_scores, user_ratings)
            sum_sims = sim_scores[user_ratings > 0].sum()
            scores[item] = weighted_sum / (sum_sims + 1e-9)

    recommended = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
    return recommended

def user_based_recommend(user_id, user_item_matrix, user_sim_df, n=5):
    similar_users = user_sim_df[user_id].drop(user_id)
    weighted_ratings = pd.Series(dtype='float64')

    for other_user, similarity in similar_users.items():
        other_ratings = user_item_matrix.loc[other_user]
        weighted_ratings = weighted_ratings.add(other_ratings * similarity, fill_value=0)

    user_ratings = user_item_matrix.loc[user_id]
    scores = weighted_ratings / (similar_users.sum() + 1e-9)
    scores = scores[user_ratings == 0]

    return list(scores.sort_values(ascending=False).head(n).items())
