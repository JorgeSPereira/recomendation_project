"""
Script de Preparação de Dados para Recomendação (ALS) com K-Core e Filtro de Bots
==============================================================================

Este script executa o pipeline de preparação de dados a partir do events.csv:
1. Divide os dados temporalmente (treino: histórico, teste: últimos 14 dias).
2. Remove bots do conjunto de treino (usuários com > 500 interações).
3. Aplica o filtro K-Core iterativo (K=5) no conjunto de treino.
4. Mapeia visitorid e itemid para índices contíguos (0 a N-1) baseado no treino limpo.
5. Exporta train_events.csv com os mapeamentos para otimização dinâmica de pesos.
6. Constrói e exporta a matriz esparsa de treino baseline (CSR).
7. Salva os arquivos de mapeamento e o Ground Truth de teste filtrado.
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

# Caminhos de arquivos
RAW_DATA_PATH = os.path.join("data", "raw", "events.csv")
PROCESSED_DIR = os.path.join("data", "processed")

# Limites de limpeza de dados
MAX_USER_INTERACTIONS = 500  # Limite para remover bots/outliers de cliques
K_CORE_THRESHOLD = 5         # Limite mínimo de interações por usuário e item

# Mapeamento inicial de pesos para a matriz baseline
EVENT_WEIGHTS_BASELINE = {
    "view": 1.0,
    "addtocart": 3.0,
    "transaction": 5.0
}

# Período de teste em dias (validação temporal estrita)
TEST_WINDOW_DAYS = 14


def main():
    print("Iniciando a preparacao dos dados (com K-Core e Filtro de Bots)...")

    # 1. Verificar se o arquivo bruto existe
    if not os.path.exists(RAW_DATA_PATH):
        print(f"ERRO: Arquivo '{RAW_DATA_PATH}' nao encontrado.")
        print("Execute primeiro: python setup_data.py")
        sys.exit(1)

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 2. Carregar events.csv otimizando tipos para economizar memória
    print("Carregando events.csv...")
    try:
        df = pd.read_csv(
            RAW_DATA_PATH,
            dtype={
                "timestamp": np.int64,
                "visitorid": np.int32,
                "event": "category",
                "itemid": np.int32,
                "transactionid": "float32"
            }
        )
    except Exception as e:
        print(f"ERRO ao carregar o arquivo: {e}")
        sys.exit(1)

    print(f"Dataset carregado com {len(df):,} interacoes.")

    # 3. Divisão Temporal Estrita
    print("Realizando a divisao temporal...")
    max_timestamp = df["timestamp"].max()
    window_ms = TEST_WINDOW_DAYS * 24 * 60 * 60 * 1000
    cutoff_timestamp = max_timestamp - window_ms

    train_df = df[df["timestamp"] < cutoff_timestamp].copy()
    test_df = df[df["timestamp"] >= cutoff_timestamp].copy()

    # --- ETAPA DE LIMPEZA DO TREINO ---
    print("\n--- Iniciando Limpeza do Conjunto de Treino ---")
    print(f"Treino original: {len(train_df):,} interacoes de {train_df['visitorid'].nunique():,} usuarios.")

    # 3.1. Remoção de Bots (Outliers)
    print(f"Removendo usuarios com mais de {MAX_USER_INTERACTIONS} interacoes (bots)...")
    user_counts = train_df["visitorid"].value_counts()
    normal_users = user_counts[user_counts <= MAX_USER_INTERACTIONS].index
    train_df = train_df[train_df["visitorid"].isin(normal_users)].copy()
    print(f"  Interacoes apos remocao de bots: {len(train_df):,} ({train_df['visitorid'].nunique():,} usuarios)")

    # 3.2. Filtro K-Core Iterativo (K=5)
    print(f"Aplicando filtro K-Core (K={K_CORE_THRESHOLD})...")
    iteration = 1
    while True:
        num_users_before = train_df["visitorid"].nunique()
        num_items_before = train_df["itemid"].nunique()
        
        # Filtrar usuários ativos
        u_counts = train_df["visitorid"].value_counts()
        keep_u = u_counts[u_counts >= K_CORE_THRESHOLD].index
        train_df = train_df[train_df["visitorid"].isin(keep_u)]
        
        # Filtrar itens ativos
        i_counts = train_df["itemid"].value_counts()
        keep_i = i_counts[i_counts >= K_CORE_THRESHOLD].index
        train_df = train_df[train_df["itemid"].isin(keep_i)]
        
        num_users_after = train_df["visitorid"].nunique()
        num_items_after = train_df["itemid"].nunique()
        
        print(f"  Iteracao {iteration:02d}: {num_users_after:,} usuarios e {num_items_after:,} itens restantes.")
        
        # Se nenhuma linha foi removida na iteração, o K-core está estabilizado
        if num_users_before == num_users_after and num_items_before == num_items_after:
            break
        iteration += 1

    train_df = train_df.copy()
    print(f"Treino limpo: {len(train_df):,} interacoes.")
    # ----------------------------------

    # 4. Mapeamento de IDs (Label Encoding baseado no Treino Limpo)
    print("\nGerando mapeamentos de IDs a partir do treino limpo...")
    unique_users = train_df["visitorid"].unique()
    unique_items = train_df["itemid"].unique()

    user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    item_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}

    # Salvar mapeamentos para uso posterior
    user_map_df = pd.DataFrame(list(user_to_idx.items()), columns=["visitorid", "user_idx"])
    item_map_df = pd.DataFrame(list(item_to_idx.items()), columns=["itemid", "item_idx"])

    user_map_df.to_csv(os.path.join(PROCESSED_DIR, "user_mapping.csv"), index=False)
    item_map_df.to_csv(os.path.join(PROCESSED_DIR, "item_mapping.csv"), index=False)
    print(f"Mapeamentos salvos. Total: {len(user_map_df):,} usuarios e {len(item_map_df):,} itens.")

    # Mapear treino
    train_df["user_idx"] = train_df["visitorid"].map(user_to_idx)
    train_df["item_idx"] = train_df["itemid"].map(item_to_idx)

    # 5. Salvar train_events.csv para otimização de pesos no Optuna
    train_events_df = train_df[["user_idx", "item_idx", "event"]].copy()
    train_events_path = os.path.join(PROCESSED_DIR, "train_events.csv")
    train_events_df.to_csv(train_events_path, index=False)
    print(f"Eventos de treino mapeados salvos em '{train_events_path}'.")

    # 6. Construir e salvar matriz de treino baseline
    print("Processando dados de treino baseline...")
    train_df["weight"] = train_df["event"].map(EVENT_WEIGHTS_BASELINE).astype(np.float32)
    grouped_train = train_df.groupby(["user_idx", "item_idx"], as_index=False)["weight"].sum()

    num_users = len(unique_users)
    num_items = len(unique_items)

    print(f"Construindo matriz de treino CSR baseline ({num_users} x {num_items})...")
    train_matrix = csr_matrix(
        (grouped_train["weight"], (grouped_train["user_idx"], grouped_train["item_idx"])),
        shape=(num_users, num_items)
    )

    train_matrix_path = os.path.join(PROCESSED_DIR, "train_matrix.npz")
    from scipy.sparse import save_npz
    save_npz(train_matrix_path, train_matrix)
    print(f"Matriz baseline salva em '{train_matrix_path}'.")

    # 7. Processamento do Teste (Ground Truth para Avaliação)
    print("Processando dados de teste (Ground Truth)...")
    # Apenas transações reais de compra
    test_transactions = test_df[test_df["event"] == "transaction"].copy()

    # Mapear IDs usando os mapeamentos do treino
    test_transactions["user_idx"] = test_transactions["visitorid"].map(user_to_idx)
    test_transactions["item_idx"] = test_transactions["itemid"].map(item_to_idx)

    # Remover transações com usuários ou itens frios (que foram removidos no K-core)
    test_transactions = test_transactions.dropna(subset=["user_idx", "item_idx"]).copy()
    test_transactions["user_idx"] = test_transactions["user_idx"].astype(np.int32)
    test_transactions["item_idx"] = test_transactions["item_idx"].astype(np.int32)

    # Agrupar compras por usuário
    test_gt = test_transactions.groupby("user_idx")["item_idx"].apply(list).reset_index()

    # Salvar Ground Truth de teste
    test_gt["item_idx_list"] = test_gt["item_idx"].apply(lambda x: " ".join(map(str, x)))
    test_gt = test_gt.drop(columns=["item_idx"])

    test_gt_path = os.path.join(PROCESSED_DIR, "test_interactions.csv")
    test_gt.to_csv(test_gt_path, index=False)
    print(f"Ground Truth de teste salvo em '{test_gt_path}'.")
    print(f"Total de {len(test_gt):,} usuarios ativos validos e avaliaveis no teste.")

    print("\nPreparacao de dados com K-core concluida com sucesso!")


if __name__ == "__main__":
    main()
