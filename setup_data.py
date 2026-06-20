"""
Script de Ingestão de Dados — RetailRocket E-Commerce Dataset
=============================================================

Este script utiliza a API oficial do Kaggle para baixar e extrair
automaticamente o dataset 'retailrocket/ecommerce-dataset' no diretório
data/raw/ do projeto.

PRÉ-REQUISITOS
--------------
1. Ter uma conta no Kaggle (https://www.kaggle.com).
2. Gerar um token de API em:
       Kaggle → Account → API → Create New Token
   Isso fará o download do arquivo **kaggle.json**.
3. Copiar o arquivo kaggle.json para o local esperado:
       • Windows : C:\\Users\\<seu_usuario>\\.kaggle\\kaggle.json
       • Linux   : ~/.kaggle/kaggle.json
       • macOS   : ~/.kaggle/kaggle.json
4. Garantir que as permissões do arquivo estejam restritas (Linux/macOS):
       chmod 600 ~/.kaggle/kaggle.json

USO
---
    python setup_data.py
"""

import os
import sys
import zipfile
import glob

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
DATASET_SLUG = "retailrocket/ecommerce-dataset"
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")


def main():
    # 1. Garantir que o diretório de destino existe
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    # 2. Importar a API do Kaggle (verifica se está instalada)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print(
            "ERRO: A biblioteca 'kaggle' não está instalada.\n"
            "Execute:  pip install kaggle"
        )
        sys.exit(1)

    # 3. Autenticar (usa o kaggle.json configurado na máquina)
    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print(
            "ERRO: Falha na autenticação com a API do Kaggle.\n"
            "Verifique se o arquivo kaggle.json está no local correto:\n"
            "  • Windows : C:\\Users\\<usuario>\\.kaggle\\kaggle.json\n"
            "  • Linux   : ~/.kaggle/kaggle.json\n"
            f"\nDetalhes: {e}"
        )
        sys.exit(1)

    # 4. Baixar o dataset
    print(f"Baixando dataset '{DATASET_SLUG}' para '{RAW_DATA_DIR}' ...")
    try:
        api.dataset_download_files(DATASET_SLUG, path=RAW_DATA_DIR, unzip=True)
    except Exception as e:
        print(f"ERRO ao baixar o dataset: {e}")
        sys.exit(1)

    # 5. Verificar os arquivos extraídos
    csv_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))
    if csv_files:
        print("\n[OK] Download e extracao concluidos com sucesso!")
        print("Arquivos disponiveis em data/raw/:")
        for f in sorted(csv_files):
            size_mb = os.path.getsize(f) / (1024 * 1024)
            print(f"  - {os.path.basename(f)}  ({size_mb:.1f} MB)")
    else:
        print(
            "\n[AVISO] Nenhum arquivo .csv encontrado em data/raw/.\n"
            "Verifique se o download foi realizado corretamente."
        )
        sys.exit(1)

    print("\nSetup de dados finalizado. Proximo passo: explorar os dados em notebooks/.")


if __name__ == "__main__":
    main()
