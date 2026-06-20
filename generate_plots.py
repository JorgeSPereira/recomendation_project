"""
Script de Geração de Visualizações para o Portfólio
===================================================

Este script analisa os dados originais e processados para gerar 4 gráficos:
1. Funil de Conversão (Implicit Feedback)
2. Efeito do K-Core e Bot Cleaning (Antes vs Depois)
3. Distribuição de Cauda Longa (Long Tail) dos Produtos
4. Melhoria da Métrica NDCG@10 (Sucesso do Modelo)

Os gráficos são salvos na pasta 'images/'.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração global de estilo para gráficos elegantes e modernos
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.titlesize": 16,
    "axes.edgecolor": "#CCCCCC",
    "patch.force_edgecolor": True,
    "patch.facecolor": "#3498db"
})

RAW_EVENTS_PATH = os.path.join("data", "raw", "events.csv")
PROCESSED_EVENTS_PATH = os.path.join("data", "processed", "train_events.csv")
IMAGES_DIR = "images"


def main():
    print("Iniciando geracao dos graficos de storytelling...")

    # Verificar se os dados existem
    if not os.path.exists(RAW_EVENTS_PATH) or not os.path.exists(PROCESSED_EVENTS_PATH):
        print("ERRO: Os dados brutos ou processados nao foram encontrados.")
        print("Certifique-se de ter rodado: python setup_data.py e python prepare_data.py")
        sys.exit(1)

    os.makedirs(IMAGES_DIR, exist_ok=True)

    # 1. Carregar Dados Brutos e Processados
    print("Carregando datasets (isso pode levar alguns segundos)...")
    raw_df = pd.read_csv(RAW_EVENTS_PATH, usecols=["visitorid", "itemid", "event"])
    train_events_df = pd.read_csv(PROCESSED_EVENTS_PATH)

    # =========================================================================
    # PLOT 1: Funil de Conversão (Feedback Implícito)
    # =========================================================================
    print("Gerando Plot 1: Funil de Conversao...")
    event_counts = raw_df["event"].value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2b5c8f", "#4682b4", "#f39c12"]  # Tons de azul com destaque laranja para compra
    
    sns.barplot(x=event_counts.index, y=event_counts.values, palette=colors, ax=ax, hue=event_counts.index, legend=False)
    
    # Adicionar anotações de contagem e porcentagem
    total = event_counts.sum()
    for i, val in enumerate(event_counts.values):
        pct = (val / total) * 100
        ax.text(i, val + (total * 0.005), f"{val:,}\n({pct:.2f}%)", ha="center", va="bottom", fontweight="bold")
    
    ax.set_title("Funil de Eventos no E-Commerce (Feedback Implicito)", pad=15)
    ax.set_xlabel("Tipo de Evento")
    ax.set_ylabel("Quantidade de Interacoes")
    ax.set_yscale("log")  # Escala logarítmica devido à enorme disparidade das visualizações
    
    plt.tight_layout()
    plot1_path = os.path.join(IMAGES_DIR, "funnel_events.png")
    plt.savefig(plot1_path, dpi=150)
    plt.close()
    print(f"  Saved: {plot1_path}")

    # =========================================================================
    # PLOT 2: Antes vs Depois da Limpeza (Bots e K-Core)
    # =========================================================================
    print("Gerando Plot 2: Antes vs Depois do K-Core...")
    
    # Métricas antes e depois
    users_before = raw_df["visitorid"].nunique()
    items_before = raw_df["itemid"].nunique()
    events_before = len(raw_df)

    # Para os dados depois, carregamos do mapeamento
    users_after = pd.read_csv(os.path.join("data", "processed", "user_mapping.csv")).shape[0]
    items_after = pd.read_csv(os.path.join("data", "processed", "item_mapping.csv")).shape[0]
    events_after = len(train_events_df)

    labels = ["Usuarios Unicos", "Itens Unicos", "Interacoes Totais"]
    before_vals = [users_before, items_before, events_before]
    after_vals = [users_after, items_after, events_after]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar(x - width/2, before_vals, width, label="Bruto (Original)", color="#95a5a6")
    rects2 = ax.bar(x + width/2, after_vals, width, label="Limpo (K-Core + Bots)", color="#27ae60")

    ax.set_title("Impacto do Tratamento de Dados (K-Core K=5 + Bot Removal)", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Contagem (Escala Log)")
    ax.set_yscale("log")
    ax.legend()

    # Adicionar rótulos nas barras
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:,}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="semibold")

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plot2_path = os.path.join(IMAGES_DIR, "before_after_cleaning.png")
    plt.savefig(plot2_path, dpi=150)
    plt.close()
    print(f"  Saved: {plot2_path}")

    # =========================================================================
    # PLOT 3: Distribuição de Cauda Longa (Long Tail)
    # =========================================================================
    print("Gerando Plot 3: Distribuição de Cauda Longa...")
    
    # Calcular contagem de interações por produto
    item_counts = raw_df["itemid"].value_counts().sort_values(ascending=False).values
    
    # Cumulative Sum
    cumulative_sum = np.cumsum(item_counts)
    cumulative_pct = (cumulative_sum / cumulative_sum[-1]) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plotar a curva de porcentagem cumulativa
    ax.plot(cumulative_pct, color="#2b5c8f", linewidth=2.5, label="Popularidade Cumulativa")
    
    # Achar o índice do produto onde acumulamos 80% das interações
    idx_80 = np.argmax(cumulative_pct >= 80)
    pct_items_80 = (idx_80 / len(item_counts)) * 100

    # Marcar a linha dos 80%
    ax.axhline(y=80, color="#e74c3c", linestyle="--", linewidth=1.5)
    ax.axvline(x=idx_80, color="#e74c3c", linestyle="--", linewidth=1.5)
    ax.scatter(idx_80, 80, color="#e74c3c", s=50, zorder=5)

    ax.set_title("Efeito Cauda Longa no E-Commerce", pad=15)
    ax.set_xlabel("Produtos Ordenados por Popularidade (Index)")
    ax.set_ylabel("Porcentagem Cumulativa das Interacoes (%)")
    ax.set_ylim(0, 105)
    ax.set_xlim(0, len(item_counts))
    
    # Anotação
    ax.text(idx_80 + (len(item_counts)*0.02), 65, 
            f"Regra de Pareto:\n apenas {pct_items_80:.1f}% dos produtos\n geram 80% de todos os\n cliques/compras!", 
            color="#c0392b", fontsize=10, bbox=dict(facecolor="#fdf2e9", edgecolor="#f5b041", boxstyle="round,pad=0.5"))

    plt.tight_layout()
    plot3_path = os.path.join(IMAGES_DIR, "long_tail.png")
    plt.savefig(plot3_path, dpi=150)
    plt.close()
    print(f"  Saved: {plot3_path}")

    # =========================================================================
    # PLOT 4: Evolução do NDCG@10 (Sucesso do Modelo)
    # =========================================================================
    print("Gerando Plot 4: Evolucao da Metrica NDCG@10...")
    
    baseline_ndcg = 0.02008
    optimized_ndcg = 0.02265
    improvement = ((optimized_ndcg - baseline_ndcg) / baseline_ndcg) * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    
    models = ["Baseline ALS (Dados Brutos)", "Cleaned & Optimized ALS\n(K-Core + Weights + Optuna)"]
    scores = [baseline_ndcg, optimized_ndcg]
    colors_bar = ["#95a5a6", "#2980b9"]

    sns.barplot(x=models, y=scores, palette=colors_bar, ax=ax, hue=models, legend=False)

    # Adicionar as notas no topo das barras
    for i, v in enumerate(scores):
        ax.text(i, v + 0.0005, f"{v:.5f}", ha="center", va="bottom", fontweight="bold", fontsize=11)

    # Anotação de melhoria
    ax.text(0.5, 0.010, f"+{improvement:.1f}% de Ganho", 
            ha="center", va="center", color="#27ae60", fontweight="bold", fontsize=14,
            bbox=dict(facecolor="#e8f8f5", edgecolor="#2ecc71", boxstyle="round,pad=0.6"))

    ax.set_title("Evolucao do NDCG@10 (Métrica de Validacao)", pad=15)
    ax.set_ylabel("NDCG@10")
    ax.set_ylim(0, 0.026)

    plt.tight_layout()
    plot4_path = os.path.join(IMAGES_DIR, "ndcg_evolution.png")
    plt.savefig(plot4_path, dpi=150)
    plt.close()
    print(f"  Saved: {plot4_path}")

    print("\nTodos os graficos de storytelling foram gerados com sucesso na pasta 'images/'!")


if __name__ == "__main__":
    main()
