from collections import defaultdict
from pprint import pprint

import pandas as pd
import numpy as np
from surprise import KNNWithMeans, Reader
from surprise import Dataset
from surprise.accuracy import mae
from surprise.model_selection import train_test_split


# --------------------------------------------------------------------
# - Recibe la lista de predicciones de Surprise (uid, iid, true_r, est, details)
# - n: número de ítems recomendados por usuario (Top-N)
# - threshold: a partir de qué rating consideramos que un ítem es relevante
#              (en el enunciado: ratings 4 o 5 → threshold = 4.0)
# - Devuelve dos diccionarios:
#       precisions[uid] = precisión@N de ese usuario
#       recalls[uid]    = recall@N de ese usuario
# --------------------------------------------------------------------
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
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_from_file('data/ml-100k/u.data', reader=reader)
    return data

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
        # Crear algoritmo KNNWithMeans con ese k
        algo = KNNWithMeans(k=k, sim_options=sim_options_KNN, verbose=False)
        # Entrenar con el conjunto de entrenamiento
        algo.fit(trainset)
        # Predecir ratings del conjunto de test
        preds = algo.test(testset)
        # Calcular MAE sobre las predicciones
        m = mae(preds, verbose=False)
        resultados.append((k, m))
        print(f"[KNN] test_size={test_size}, k={k}, MAE={m:.4f}")
    return resultados

# --------------------------------------------------------------------
# A partir de una lista de predicciones (preds), calcula la media de
# precisión, recall y F1 para una lista de valores de N (n_values).
# Devuelve una lista:
#   [(N, precision_media, recall_media, F1_media), ...]
# --------------------------------------------------------------------
def topn_metrics(predictions, n_values, threshold=4.0):
    resultados = []

    for n in n_values:
        # Calculamos precisión y recall por usuario para este N
        precisions, recalls = precision_recall_at_n(predictions, n=n, threshold=threshold)

        # Media de precisión sobre todos los usuarios
        mean_prec = sum(precisions.values()) / len(precisions)

        # Media de recall sobre todos los usuarios
        mean_rec = sum(recalls.values()) / len(recalls)

        # F1 = 2 * (P * R) / (P + R). Si P+R = 0, definimos F1 = 0 para evitar división por cero.
        if (mean_prec + mean_rec) > 0:
            f1 = 2 * mean_prec * mean_rec / (mean_prec + mean_rec)
        else:
            f1 = 0.0

        print(f"N={n:3d} | Precision={mean_prec:.4f} | Recall={mean_rec:.4f} | F1={f1:.4f}")

        resultados.append((n, mean_prec, mean_rec, f1))

    return resultados

# -------------------
# MAIN PROGRAM

# Loading of the dataset
data = load_csv()

# 2) Definir valores de K a probar para MAE
ks = [2, 5, 10, 15, 20]

# --- a) 25% missing ratings (test_size = 0.25) ---
resultados_25 = evaluate_ks(data, test_size=0.25, ks=ks)
best_k_25, best_mae_25 = min(resultados_25, key=lambda x: x[1])
print(f"\nMejor K para 25% missing: K={best_k_25}, MAE={best_mae_25:.4f}")

# --- b) 75% missing ratings (test_size = 0.75) ---
resultados_75 = evaluate_ks(data, test_size=0.75, ks=ks)
best_k_75, best_mae_75 = min(resultados_75, key=lambda x: x[1])
print(f"\nBest K for 75% missing: K={best_k_75}, MAE={best_mae_75:.4f}")

# -----------------------------------------------------------
# PARTE 3: Top-N metrics (Precision, Recall, F1) para K-NN
#          con los mejores K, y N = 10..100
# -----------------------------------------------------------

# Valores de N (Top-N) que queremos evaluar: 10,20,...,100
n_values = list(range(10, 101, 10))

# ------------------------------
# 3A) Caso 25% missing (K = best_k_25)
# ------------------------------
# Volvemos a hacer el split con el mismo random_state para reproducir el mismo escenario
trainset25, testset25 = train_test_split(data, test_size=0.25, random_state=22)

# Creamos el modelo KNNWithMeans con el mejor K encontrado para este caso
algo_knn_25 = KNNWithMeans(k=best_k_25, sim_options=sim_options_KNN, verbose=False)
algo_knn_25.fit(trainset25)

# Obtenemos predicciones sobre el conjunto de test
preds_25 = algo_knn_25.test(testset25)

print("\n========== K-NN Top-N metrics (25% missing) ==========")
print(f"Usando K = {best_k_25} y threshold de relevancia = 4.0 (ratings 4 o 5)\n")
resultados_topn_25 = topn_metrics(preds_25, n_values, threshold=4.0)

# ------------------------------
# 3B) Caso 75% missing (K = best_k_75)
# ------------------------------
trainset75, testset75 = train_test_split(data, test_size=0.75, random_state=22)

algo_knn_75 = KNNWithMeans(k=best_k_75, sim_options=sim_options_KNN, verbose=False)
algo_knn_75.fit(trainset75)

preds_75 = algo_knn_75.test(testset75)

print("\n========== K-NN Top-N metrics (75% missing) ==========")
print(f"Usando K = {best_k_75} y threshold de relevancia = 4.0 (ratings 4 o 5)\n")
resultados_topn_75 = topn_metrics(preds_75, n_values, threshold=4.0)