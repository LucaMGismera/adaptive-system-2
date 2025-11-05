from collections import defaultdict
from pprint import pprint

import pandas as pd
import numpy as np
from surprise import Reader, SVD
from surprise import Dataset
from surprise.accuracy import mae
from surprise.model_selection import train_test_split


# Load the movielens-100k dataset (download it if needed).
# data = Dataset.load_builtin('ml-100k')

def precision_recall_at_n(predictions, n=10, threshold=3.5):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of relevant and recommended items in top n
        n_rel_and_rec = sum(
            (true_r >= threshold)
            for (_, true_r) in user_ratings[:n]
        )

        # Precision@n: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec / n

        # Recall@n: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec / n_rel if n_rel != 0 else 0

    return precisions, recalls


def load_csv():
    # # Dataset Load

    # In[2]:

    # import csv file in python
    csv_file = pd.read_csv('data.csv', delimiter=';')

    # conversion of the table in a numpy matrix with only ratings
    # method .to_numpy() does not copy the header row
    # np.delete deletes the first column

    temp = np.delete(csv_file.to_numpy(), np.s_[0], axis=1)

    # Matrix is converted in User-Movies from Movies-User
    # Later, the matrix is flatten to a vector of values.
    # For each user, all the movie ratings are reported
    # user1item1, user1item2, user1item3,...

    ratings = temp.T.flatten()

    # Vectors users and movies are the corresponding columns in the dataframne.
    # As the ratings are ordered user1[allratings], user2[allratings], ...
    # the user and movies vectors follow the same logic

    i = 0
    j = 0
    users = []
    movies = []
    users.clear()
    movies.clear()
    while i < 50:
        while j < 20:
            movies.append(j)
            users.append(i)
            j += 1
        j = 0
        i += 1

    movies = np.array(movies)
    users = np.array(users)

    # The user, movies, and rating numpy vectors are converted in a rating dictionary
    # and later, in a Pandas dataframe

    ratings_dict = {'userID': users,
                    'itemID': movies,
                    'rating': ratings}

    df = pd.DataFrame(ratings_dict)

    # The dataframes are converted into a dataset suitable for Surprise

    reader = Reader(rating_scale=(1, 5))
    return Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)


# load ratings from csv

data = load_csv()

# Dataset splitting in trainset and testset for 25% sparsity

trainset25, testset25 = train_test_split(data, test_size=.25,
                                         random_state=22)

algo = SVD(random_state=3)
algo.fit(trainset25)

# estimate the ratings for all the pairs (user, item) in testset25
predictions25KNN = algo.test(testset25)

pprint(predictions25KNN)

# the first user has uid=0 and first item iid=0
for (uid, iid, real, est, _) in predictions25KNN:
    if uid == 0:
        print(f'{uid} {iid} {real} {est}')

mae(predictions25KNN)

precisions, recalls = precision_recall_at_n(predictions25KNN, n=5, threshold=4)

# Precision and recall can then be averaged over all users
pre = sum(prec for prec in precisions.values()) / len(precisions)
recall = sum(rec for rec in recalls.values()) / len(recalls)
print("Precision:", pre)
print("Recall:", recall)
print("F1:", 2*pre*recall/(pre+recall))
