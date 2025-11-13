from collections import defaultdict
from pprint import pprint

import pandas as pd
import numpy as np
from surprise import KNNWithMeans, Reader
from surprise import Dataset
from surprise.accuracy import mae
from surprise.model_selection import train_test_split


# Load the movielens-100k dataset (download it if needed).
# data = Dataset.load_builtin('ml-100k')

#receive the prediction of Surprise,
# n -> items recommended per user
# threshold -> rating considered relevant
def precision_recall_at_n(predictions, n=10, threshold=3.5):
    """Return precision and recall at n metrics for each user"""

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


# FUNCIÓN EDITADA DEL PROGRAMA PRINCIPAL PARA QUE TRABAJE CON LA DATABASE QUE NOS PROPORCIONAN
def load_csv():
    # # Dataset Load

    # In[2]:

    # import csv file in python
    # >>> CAMBIO MÍNIMO: cargar MovieLens 100k desde data/ml-100k/u.data con Surprise <<<
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_from_file('data/ml-100k/u.data', reader=reader)
    return data

# -------------------
# MAIN PROGRAM

# Loading of the dataset
data = load_csv()
#
# # Dataset splitting in trainset and testset for 25% sparsity
# trainset25, testset25 = train_test_split(data, test_size=.25, random_state=22)

sim_options_KNN = {'name': "pearson",
                   'user_based': True  # compute similarities between users
                   }

# FIRST EXERCISE:
# A) Find out the value for K that minimizes the MAE with 25% of missing ratings.
# B) Sparsity Problem: find out the value for K that minimizes the MAE with 75% of missing ratings.

def evaluate_ks(data, test_size, ks):
    """Devuelve lista [(k, mae_k), ...] para un cierto porcentaje de test."""
    trainset, testset = train_test_split(data, test_size=test_size, random_state=22)
    resultados = []

    for k in ks:
        algo = KNNWithMeans(k=k, sim_options=sim_options_KNN, verbose=False)
        algo.fit(trainset)
        preds = algo.test(testset)
        m = mae(preds, verbose=False)  # verbose=False para que no imprima cada vez
        resultados.append((k, m))
        print(f"test_size={test_size}, k={k}, MAE={m:.4f}")

    return resultados


ks = [2, 5, 10, 15, 20]

# --- a) 25% missing ratings (test_size = 0.25) ---
resultados_25 = evaluate_ks(data, test_size=0.25, ks=ks)
best_k_25, best_mae_25 = min(resultados_25, key=lambda x: x[1])
print(f"\nMejor K para 25% missing: K={best_k_25}, MAE={best_mae_25:.4f}")

# --- b) 75% missing ratings (test_size = 0.75) ---
resultados_75 = evaluate_ks(data, test_size=0.75, ks=ks)
best_k_75, best_mae_75 = min(resultados_75, key=lambda x: x[1])
print(f"\nBest K for 75% missing: K={best_k_75}, MAE={best_mae_75:.4f}")