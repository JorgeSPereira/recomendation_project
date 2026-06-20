"""
Script de Treinamento Final e Avaliação
======================================

Este script:
1. Carrega os melhores parâmetros e pesos obtidos na otimização (ou usa padrões).
2. Reconstrói a matriz final de treino aplicando os pesos de eventos otimizados.
3. Treina o modelo ALS final.
4. Calcula e exibe as métricas finais (NDCG@10, Precision@10, Recall@10) no teste.
5. Exporta o modelo treinado para 'models/als_model.pkl' para uso posterior.
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

# Importar o calculador de métricas
from src.metrics import evaluate_ndcg_at_k, evaluate_precision_recall_at_k

# Parâmetros e pesos padrão caso a otimização não tenha rodado
DEFAULT_PARAMS = {
    "factors": 64,
    "regularization": 0.05,
    "alpha": 15.0,
    "iterations": 15,
    "weight_cart": 3.0,
    "weight_buy": 5.0
}


def main():
    train_events_path = os.path.join("data", "processed", "train_events.csv")
    test_path = os.path.join("data", "processed", "test_interactions.csv")
    best_params_path = os.path.join("data", "processed", "best_params.json")
    models_dir = "models"
    model_export_path = os.path.join(models_dir, "als_model.pkl")

    # 1. Carregar dados
    if not os.path.exists(train_events_path) or not os.path.exists(test_path):
        print("ERRO: Arquivos de dados processados nao encontrados.")
        print("Por favor, execute: python prepare_data.py")
        sys.exit(1)

    print("Carregando eventos de treino e teste...")
    train_events_df = pd.read_csv(
        train_events_path,
        dtype={
            "user_idx": np.int32,
            "item_idx": np.int32,
            "event": "category"
        }
    )
    
    test_df = pd.read_csv(test_path)
    test_gt_dict = {}
    for _, row in test_df.iterrows():
        user_idx = int(row["user_idx"])
        item_list = [int(x) for x in str(row["item_idx_list"]).split()]
        test_gt_dict[user_idx] = item_list

    # Determinar dimensões da matriz
    num_users = train_events_df["user_idx"].max() + 1
    num_items = train_events_df["item_idx"].max() + 1

    # 2. Carregar parâmetros de otimização (ou defaults)
    if os.path.exists(best_params_path):
        print(f"Carregando hiperparametros e pesos otimizados de '{best_params_path}'...")
        with open(best_params_path, "r") as f:
            params = json.load(f)
    else:
        print("Hiperparametros otimizados nao encontrados. Utilizando configuracoes padrao...")
        params = DEFAULT_PARAMS

    print("Configuracoes utilizadas:")
    for param, val in params.items():
        print(f"  - {param}: {val}")

    # 3. Reconstruir a matriz esparsa usando os pesos carregados
    print("\nConstruindo a matriz final de treino com os pesos calibrados...")
    weight_cart = params.get("weight_cart", 3.0)
    weight_buy = params.get("weight_buy", 5.0)

    weights_map = {
        "view": 1.0,
        "addtocart": weight_cart,
        "transaction": weight_buy
    }

    train_events_df["weight"] = train_events_df["event"].map(weights_map).astype(np.float32)
    grouped_train = train_events_df.groupby(["user_idx", "item_idx"], as_index=False)["weight"].sum()

    train_matrix = csr_matrix(
        (grouped_train["weight"], (grouped_train["user_idx"], grouped_train["item_idx"])),
        shape=(num_users, num_items)
    )

    # 4. Treinar o modelo ALS
    print(f"Treinando o modelo ALS final ({num_users:,} usuarios x {num_items:,} itens)...")
    
    # Filtrar apenas parâmetros do ALS para a inicialização
    als_params = {
        "factors": int(params["factors"]),
        "regularization": float(params["regularization"]),
        "alpha": float(params["alpha"]),
        "iterations": int(params["iterations"]),
        "random_state": 42,
        "num_threads": 0
    }

    model = AlternatingLeastSquares(**als_params)
    model.fit(train_matrix, show_progress=True)

    # 5. Avaliar métricas finais no conjunto de teste
    print("\nAvaliando o modelo no conjunto de teste...")
    
    # Calcular NDCG@10
    ndcg_10 = evaluate_ndcg_at_k(model, train_matrix, test_gt_dict, k=10)
    
    # Calcular Precision@10 e Recall@10
    precision_10, recall_10 = evaluate_precision_recall_at_k(model, train_matrix, test_gt_dict, k=10)

    print("\n================ METRICAS FINAIS (Top 10) ================")
    print(f"  NDCG@10:      {ndcg_10:.5f}")
    print(f"  Precision@10: {precision_10:.5f}")
    print(f"  Recall@10:    {recall_10:.5f}")
    print("==========================================================")

    # 6. Salvar o modelo final
    os.makedirs(models_dir, exist_ok=True)
    with open(model_export_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModelo salvo com sucesso em '{model_export_path}'!")


if __name__ == "__main__":
    main()
