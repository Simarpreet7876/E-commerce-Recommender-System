import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

def create_product_embeddings():
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / 'data' / 'processed' / 'user_product_interactions.parquet'
    models_path = project_root / 'models'
    models_path.mkdir(exist_ok=True)

    print("Loading data for embeddings...")
    df = pd.read_parquet(data_path)
    product_df = df[['product_id', 'category']].drop_duplicates('product_id').reset_index(drop=True)
    product_df['description'] = "Product from category: " + product_df['category'].fillna('')

    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Encoding product descriptions...")
    embeddings = model.encode(product_df['description'].tolist(), show_progress_bar=True)
    embeddings = np.array(embeddings, dtype='float32')

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss_index_path = models_path / 'faiss_index.bin'
    embeddings_path = models_path / 'product_embeddings.npy'
    product_id_map_path = models_path / 'product_id_faiss_map.csv'

    print(f"Saving FAISS index to {faiss_index_path}...")
    faiss.write_index(index, str(faiss_index_path))

    print(f"Saving embeddings to {embeddings_path}...")
    np.save(embeddings_path, embeddings)
    product_df[['product_id']].to_csv(product_id_map_path, index_label='faiss_index')

    print("✅ Product embeddings and FAISS index created.")

if __name__ == '__main__':
    create_product_embeddings()
