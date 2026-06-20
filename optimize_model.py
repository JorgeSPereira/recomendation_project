"""
Script de Otimização de Hiperparâmetros e Pesos (ALS + Optuna)
==============================================================

Este script utiliza a Otimização Bayesiana do Optuna para buscar:
1. Os melhores parâmetros do ALS (factors, regularization, alpha, iterations).
2. Os melhores pesos para os eventos (addtocart e transaction), mantendo o peso de view=1.0.

A cada trial, a matriz de treino CSR é montada dinamicamente com os pesos sugeridos.
"""

import os
import sys
import json
import argparse
import optuna
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

# Importar o calculador de métricas
from src.metrics import evaluate_ndcg_at_k

# Silenciar logs verbosos do Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description="Otimizacao do ALS e pesos de eventos com Optuna.")
    parser.add_argument("--trials", type=int, default=25, help="Numero de trials para a otimizacao. Padrao: 25.")
    args = parser.parse_args()

    train_events_path = os.path.join("data", "processed", "train_events.csv")
    test_path = os.path.join("data", "processed", "test_interactions.csv")
    best_params_path = os.path.join("data", "processed", "best_params.json")

    # 1. Carregar dados
    if not os.path.exists(train_events_path) or not os.path.exists(test_path):
        print("ERRO: Arquivos de dados processados nao encontrados.")
        print("Por favor, execute: python prepare_data.py")
        sys.exit(1)

    print("Carregando eventos de treino limpos...")
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

    print(f"Dados carregados. Usuarios: {num_users:,} | Itens: {num_items:,}")
    print(f"Iniciando otimizacao com {args.trials} trials (Budget estimado: ~15-20 min para 25 trials)...")

    # 2. Definir a funcao objetivo do Optuna
    def objective(trial):
        # Sugerir parametros do modelo
        factors = trial.suggest_int("factors", 32, 128, step=32)
        regularization = trial.suggest_float("regularization", 0.01, 0.2, log=True)
        alpha = trial.suggest_float("alpha", 5.0, 40.0)
        iterations = trial.suggest_int("iterations", 10, 20)

        # Sugerir pesos de feedback implícito
        weight_cart = trial.suggest_float("weight_cart", 2.0, 8.0)
        weight_buy = trial.suggest_float("weight_buy", 5.0, 30.0)

        # Mapear e construir a matriz esparsa de treino dinamicamente
        weights_map = {
            "view": 1.0,
            "addtocart": weight_cart,
            "transaction": weight_buy
        }
        
        # Aplicar os pesos sugeridos
        train_events_df["weight"] = train_events_df["event"].map(weights_map).astype(np.float32)
        
        # Agrupar por par usuário-item
        grouped = train_events_df.groupby(["user_idx", "item_idx"], as_index=False)["weight"].sum()

        # Construir matriz esparsa CSR
        train_matrix = csr_matrix(
            (grouped["weight"], (grouped["user_idx"], grouped["item_idx"])),
            shape=(num_users, num_items)
        )

        # Instanciar e treinar ALS
        model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            alpha=alpha,
            random_state=42,
            num_threads=0
        )
        
        model.fit(train_matrix, show_progress=False)

        # Avaliar NDCG@10
        ndcg = evaluate_ndcg_at_k(model, train_matrix, test_gt_dict, k=10)
        
        print(f"Trial {trial.number:02d} | NDCG@10: {ndcg:.5f} | model: factors={factors}, alpha={alpha:.1f} | weights: cart={weight_cart:.1f}, buy={weight_buy:.1f}")
        
        return ndcg

    # 3. Criar e rodar o estudo
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)

    print("\n--- Otimizacao Concluida ---")
    print(f"Melhor NDCG@10: {study.best_value:.5f}")
    print("Melhores parametros encontrados:")
    for param, val in study.best_params.items():
        print(f"  - {param}: {val}")

    # 4. Salvar os melhores parametros e pesos em JSON
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=4)
    print(f"\nMelhores parametros e pesos salvos em '{best_params_path}'.")


if __name__ == "__main__":
    main()
