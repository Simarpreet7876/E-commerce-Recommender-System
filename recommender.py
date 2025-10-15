import pandas as pd
from scipy.sparse import coo_matrix
import scipy.sparse
from implicit.als import AlternatingLeastSquares
import pickle
from pathlib import Path
import json

def train_collaborative_filtering_model():
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / 'data' / 'processed' / 'user_product_interactions.parquet'
    models_path = project_root / 'models'
    models_path.mkdir(exist_ok=True)

    print("Loading processed data...")
    df = pd.read_parquet(data_path)

    df['user_id'] = df['user_id'].astype(str).str.strip()
    df['user_id_cat'] = df['user_id'].astype("category").cat.codes
    df['product_id_cat'] = df['product_id'].astype("category").cat.codes

    user_categories = df['user_id'].astype("category").cat.categories
    product_categories = df['product_id'].astype("category").cat.categories

    user_map = {str(idx): user for idx, user in enumerate(user_categories)}
    product_map = {str(idx): product for idx, product in enumerate(product_categories)}

    with open(models_path / 'user_map.json', 'w') as f:
        json.dump(user_map, f)
    with open(models_path / 'product_map.json', 'w') as f:
        json.dump(product_map, f)
    print("User and product mappings saved.")

    interaction_data = df.groupby(['user_id_cat', 'product_id_cat'])['interaction'].sum().reset_index()

    user_item_matrix = coo_matrix(
        (interaction_data['interaction'].astype(float),
         (interaction_data['user_id_cat'],
          interaction_data['product_id_cat']))
    ).tocsr()

    matrix_path = models_path / 'user_item_matrix.npz'
    scipy.sparse.save_npz(matrix_path, user_item_matrix)
    print(f"User-item matrix saved to {matrix_path}")

    item_user_matrix = user_item_matrix.T.tocsr()

    print("Training ALS model...")
    model = AlternatingLeastSquares(factors=64, regularization=0.05, iterations=50)
    model.fit(item_user_matrix)

    model_path = models_path / 'trained_cf_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"✅ ALS model trained and saved to {model_path}")

if __name__ == '__main__':
    train_collaborative_filtering_model()
