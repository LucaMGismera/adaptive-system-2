from collections import defaultdict
from surprise import Reader, Dataset, SVD
from surprise.accuracy import mae
from surprise.model_selection import train_test_split


# --------------------------------------------------------------------
# FUNCIÓN: precision_recall_at_n
# --------------------------------------------------------------------
# Misma función que en KNN. Se mantiene igual para comparar modelos.
# --------------------------------------------------------------------
def precision_recall_at_n(predictions, n=10, threshold=4.0):
    """Return precision and recall at n metrics for each user"""

    user_est_true = defaultdict(list)

    # Agrupar predicciones por usuario
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = {}
    recalls = {}

    for uid, user_ratings in user_est_true.items():
        # Ordenar por rating estimado, de mayor a menor
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Conteo de ítems relevantes (rating real >= threshold)
        n_rel = sum(true_r >= threshold for (_, true_r) in user_ratings)

        # Conteo de ítems relevantes dentro del Top-N
        n_rel_and_rec = sum(true_r >= threshold for (_, true_r) in user_ratings[:n])

        # Precisión@n = ítems relevantes dentro del top N / N
        precisions[uid] = n_rel_and_rec / n

        # Recall@n = ítems relevantes recuperados / total de relevantes
        recalls[uid] = n_rel_and_rec / n_rel if n_rel > 0 else 0

    return precisions, recalls


# --------------------------------------------------------------------
# FUNCIÓN: topn_metrics
# --------------------------------------------------------------------
# Calcula precisión, recall y F1 medios para una lista de valores de N.
# --------------------------------------------------------------------
def topn_metrics(predictions, n_values, threshold=4.0):
    resultados = []

    for n in n_values:
        precisions, recalls = precision_recall_at_n(predictions, n=n, threshold=threshold)

        mean_prec = sum(precisions.values()) / len(precisions)
        mean_rec = sum(recalls.values()) / len(recalls)

        # F1 = media armónica de precisión y recall
        if mean_prec + mean_rec > 0:
            f1 = 2 * mean_prec * mean_rec / (mean_prec + mean_rec)
        else:
            f1 = 0

        print(f"N={n:3d} | Precision={mean_prec:.4f} | Recall={mean_rec:.4f} | F1={f1:.4f}")

        resultados.append((n, mean_prec, mean_rec, f1))

    return resultados


# --------------------------------------------------------------------
# FUNCIÓN: load_csv
# --------------------------------------------------------------------
def load_csv():
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    return Dataset.load_from_file('data/ml-100k/u.data', reader=reader)


# ===========================
#       MAIN PROGRAM
# ===========================

# 1) Cargar dataset
data = load_csv()

# Valores de N para Top-N
n_values = list(range(10, 101, 10))

# -------------------------------------------------------------
# A) Caso 25% missing  (test_size = 0.25) → mucha información
# -------------------------------------------------------------
train25, test25 = train_test_split(data, test_size=0.25, random_state=22)

# Crear modelo SVD (Funk variant)
algo_svd_25 = SVD(
    n_factors=100,   # dimensiones latentes
    n_epochs=20,     # número de iteraciones SGD
    random_state=3
)

# Entrenar
algo_svd_25.fit(train25)

# Obtener predicciones
preds_svd_25 = algo_svd_25.test(test25)

# MAE (solo informativo)
print("-----------------------------------------------------")
print("SVD – MAE para 25% missing")
print("-----------------------------------------------------")
mae_25 = mae(preds_svd_25, verbose=True)
print(f"SVD MAE (25% missing): {mae_25:.4f}\n")


# Top-N metrics
print("============= SVD Top-N Metrics (25% missing) =============")
resultados_svd_25 = topn_metrics(preds_svd_25, n_values, threshold=4.0)



# -------------------------------------------------------------
# B) Caso 75% missing  (test_size = 0.75) → alta esparsidad
# -------------------------------------------------------------
train75, test75 = train_test_split(data, test_size=0.75, random_state=22)

algo_svd_75 = SVD(
    n_factors=100,
    n_epochs=20,
    random_state=3
)

algo_svd_75.fit(train75)
preds_svd_75 = algo_svd_75.test(test75)

print("\n-----------------------------------------------------")
print("SVD – MAE para 75% missing")
print("-----------------------------------------------------")
mae_75 = mae(preds_svd_75, verbose=True)
print(f"SVD MAE (75% missing): {mae_75:.4f}\n")


# Top-N metrics
print("============= SVD Top-N Metrics (75% missing) =============")
resultados_svd_75 = topn_metrics(preds_svd_75, n_values, threshold=4.0)
