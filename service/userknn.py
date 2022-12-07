import pickle
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


# function to recommend to get user neighbours
def generate_implicit_recs_mapper(model, N, users_mapping, users_inv_mapping
                                  ) -> Callable:
    def _recs_mapper(user):
        user_id = users_mapping[user]
        recs = model.similar_items(user_id, N=N)
        return ([users_inv_mapping[user] for user, _ in recs],
                [sim for _, sim in recs])
    return _recs_mapper


class OfflineUserKnnRecommender:
    ''' Offline recommender for user-based model'''
    def __init__(self, model_name: str,
                 models_dir: Path = Path('models'),
                 popular_name: str = 'popular.pkl'):
        with open(models_dir / model_name, 'rb') as f:
            self.model = pickle.load(f)
        with open(models_dir / popular_name, 'rb') as f:
            self.popular = pickle.load(f)

    def recommend(self, user: str, k: int = 10):
        return self.model.get(user, self.popular)[:k]


class OnlineUserKnnRecommender:
    ''' Online recommender for user-based model'''
    def __init__(self, model_name: str,
                 models_dir: Path = Path('models'),
                 popular_name: str = 'popular.pkl',
                 train_name: str = 'train.csv'):
        with open(models_dir / model_name, 'rb') as f:
            self.model = pickle.load(f)
        with open(models_dir / popular_name, 'rb') as f:
            self.popular = pickle.load(f)
        self.train = pd.read_csv(models_dir / train_name)
        self._prepare()

    def _prepare(self) -> None:
        ''' Prepare model for recommendations '''

        # Pre-calculate IDF for items
        cnt: Counter = Counter(self.train['item_id'].values)
        self.idf = pd.DataFrame.from_dict(
            cnt,
            orient='index',
            columns=['doc_freq']
            ).reset_index()
        n = self.train.shape[0]
        self.idf['idf'] = self.idf['doc_freq'].apply(
            lambda x: np.log((1 + n) / (1 + x) + 1))

        # Create mapping for users
        users_inv_mapping = dict(enumerate(self.train['user_id'].unique()))
        self.users_mapping = {v: k for k, v in users_inv_mapping.items()}

        self.mapper = generate_implicit_recs_mapper(
            self.model,
            N=self.model.K,
            users_mapping=self.users_mapping,
            users_inv_mapping=users_inv_mapping
        )

        # Prepare watched items for each user
        watched_items_df = self.train.groupby(
            'user_id'
            ).agg({'item_id': list}).reset_index()
        self.watched_items = {}
        for _, row in watched_items_df.iterrows():
            # print(row)
            self.watched_items[row['user_id']] = row['item_id']

    def recommend(self, user: str, k: int = 10):
        if user not in self.users_mapping:
            return self.popular[:k]

        # Find similar users
        recs = pd.DataFrame({
            'user_id': [user]
        })
        recs['similar_user_id'], recs['similarity'] = zip(
            *recs['user_id'].map(self.mapper))
        recs = recs.set_index('user_id').apply(pd.Series.explode).reset_index()
        # delete recommendations of itself
        recs = recs[~(recs['user_id'] == recs['similar_user_id'])]

        # Get items watched by similar users
        recs['item_id'] = recs['user_id'].apply(
            lambda x: self.watched_items.get(x, [])
            )
        recs = recs.explode('item_id')
        recs = recs.sort_values(['user_id', 'similarity'], ascending=False)
        recs = recs.drop_duplicates(['user_id', 'item_id'], keep='first')

        # Add IDF for items and calculate final ranks
        recs_with_idf = recs.merge(
            self.idf[['index', 'idf']],
            left_on='item_id',
            right_on='index',
            how='left'
        ).drop(['index'], axis=1)
        recs_idf = recs_with_idf['idf']
        recs_with_idf['rank_idf'] = recs_with_idf['similarity'] * recs_idf
        recs_with_idf = recs_with_idf.sort_values(['user_id', 'rank_idf'],
                                                  ascending=False)
        # Calculate final ranks
        recs_with_idf['rank'] = recs_with_idf.groupby('user_id').cumcount() + 1

        # We need only 10 recs for each user
        recs_with_idf = recs_with_idf[recs_with_idf['rank'] < 11]
        user_recs = list(recs_with_idf['item_id'].values)

        # Add popular items if needed
        i = 0
        while len(user_recs) < 10:
            if self.popular[i] not in user_recs:
                user_recs.append(self.popular[i])
            i += 1
        return user_recs[:k]
