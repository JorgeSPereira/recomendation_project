"""
Módulo de Métricas de Recomendação
==================================

Contém funções para avaliar o desempenho do modelo de recomendação,
incluindo o cálculo de NDCG@K, Precision@K e Recall@K.
"""

import numpy as np


def evaluate_ndcg_at_k(model, train_matrix, test_gt_dict, k=10):
    """
    Calcula o NDCG@K médio para um conjunto de usuários de teste.

    Parâmetros:
    -----------
    model : AlternatingLeastSquares
        Modelo ALS treinado.
    train_matrix : csr_matrix
        Matriz de treino (User-Item) para que o implicit filtre itens já vistos.
    test_gt_dict : dict
        Dicionário mapeando user_idx -> lista de item_idx reais comprados no teste.
    k : int
        Número de recomendações a avaliar (Top K).

    Retorna:
    --------
    float
        O NDCG@K médio de todos os usuários avaliados.
    """
    user_ids = list(test_gt_dict.keys())
    if not user_ids:
        return 0.0

    # 1. Gerar recomendações em lote (muito mais rápido do que um por um)
    # ids: 2D numpy array de shape (len(user_ids), k) contendo os item_idx recomendados
    ids, _ = model.recommend(
        user_ids,
        train_matrix[user_ids],
        N=k,
        filter_already_liked_items=True
    )

    ndcg_list = []

    # 2. Calcular o NDCG para cada usuário
    for idx, user_idx in enumerate(user_ids):
        rec_list = ids[idx]
        gt_set = set(test_gt_dict[user_idx])

        # Calcular DCG@K
        dcg = 0.0
        for i, item_rec in enumerate(rec_list):
            if item_rec in gt_set:
                dcg += 1.0 / np.log2(i + 2)

        # Calcular IDCG@K (melhor ordenação possível dos acertos)
        num_hits_possible = min(len(gt_set), k)
        if num_hits_possible == 0:
            ndcg_list.append(0.0)
            continue

        idcg = sum(1.0 / np.log2(j + 2) for j in range(num_hits_possible))

        ndcg_list.append(dcg / idcg)

    return np.mean(ndcg_list)


def evaluate_precision_recall_at_k(model, train_matrix, test_gt_dict, k=10):
    """
    Calcula Precision@K e Recall@K médios para os usuários de teste.

    Parâmetros:
    -----------
    model : AlternatingLeastSquares
        Modelo ALS treinado.
    train_matrix : csr_matrix
        Matriz de treino (User-Item).
    test_gt_dict : dict
        Dicionário mapeando user_idx -> lista de item_idx reais comprados.
    k : int
        Número de recomendações.

    Retorna:
    --------
    (float, float)
        (Precision@K médio, Recall@K médio)
    """
    user_ids = list(test_gt_dict.keys())
    if not user_ids:
        return 0.0, 0.0

    ids, _ = model.recommend(
        user_ids,
        train_matrix[user_ids],
        N=k,
        filter_already_liked_items=True
    )

    precisions = []
    recalls = []

    for idx, user_idx in enumerate(user_ids):
        rec_list = ids[idx]
        gt_set = set(test_gt_dict[user_idx])

        # Encontrar interseção (acertos)
        hits = len(gt_set.intersection(rec_list))

        # Precision@K = acertos / total_recomendados (K)
        precisions.append(hits / k)

        # Recall@K = acertos / total_reais_comprados
        recalls.append(hits / len(gt_set) if len(gt_set) > 0 else 0.0)

    return np.mean(precisions), np.mean(recalls)
