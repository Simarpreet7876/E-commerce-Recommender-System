# --- ADD THIS IMPORT AT THE TOP ---
from fastapi.middleware.cors import CORSMiddleware
# ------------------------------------

from fastapi import FastAPI, HTTPException
from pathlib import Path
import pickle
import json
import pandas as pd
import numpy as np
import faiss
import scipy.sparse
from src.llm.explain import generate_explanation_lm_studio
from src.api.schemas import RecommendationResponse, Recommendation, ExplanationResponse

app = FastAPI(title="E-commerce Recommender API")

# --- ADD THIS MIDDLEWARE CONFIGURATION ---
# This allows your HTML file (and any other website) to make requests to this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, including local files opened in a browser.
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.).
    allow_headers=["*"],  # Allows all request headers.
)
# -----------------------------------------


MODELS = {}

@app.on_event("startup")
def load_models():
    print("Loading models and data into memory...")
    project_root = Path(__file__).resolve().parents[2]
    models_path = project_root / 'models'

    # Load CF model
    with open(models_path / 'trained_cf_model.pkl', 'rb') as f:
        MODELS['als_model'] = pickle.load(f)

    # Load user and product maps
    with open(models_path / 'user_map.json', 'r') as f:
        loaded = json.load(f)
        MODELS['user_map'] = {int(k): v for k, v in loaded.items()}

    with open(models_path / 'product_map.json', 'r') as f:
        loaded = json.load(f)
        MODELS['product_map'] = {int(k): v for k, v in loaded.items()}

    # Reverse maps
    MODELS['user_map_rev'] = {v: k for k, v in MODELS['user_map'].items()}
    MODELS['product_map_rev'] = {v: k for k, v in MODELS['product_map'].items()}

    # Load FAISS & embeddings
    MODELS['faiss_index'] = faiss.read_index(str(models_path / 'faiss_index.bin'))
    MODELS['product_id_faiss_map'] = pd.read_csv(models_path / 'product_id_faiss_map.csv').set_index('faiss_index')
    MODELS['product_embeddings'] = np.load(models_path / 'product_embeddings.npy')

    # Load saved user-item sparse matrix
    MODELS['user_item_matrix'] = scipy.sparse.load_npz(models_path / 'user_item_matrix.npz')

    # Popular items for fallback
    data_path = project_root / 'data' / 'processed' / 'user_product_interactions.parquet'
    df = pd.read_parquet(data_path)
    MODELS['popular_items'] = df['product_id'].value_counts().nlargest(20).index.tolist()

    # Create a map from product_id to product_name
    print("Loading product name map...")
    product_name_map = df[['product_id', 'category']].drop_duplicates('product_id').set_index('product_id')
    MODELS['product_name_map'] = product_name_map['category'].to_dict()

    print("✅ Models loaded successfully.")


@app.get("/")
def read_root():
    return {"message": "Welcome. Visit /docs for documentation."}


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def get_recommendations(user_id: str, k: int = 10):
    user_id = user_id.strip()
    recs = []
    product_name_map = MODELS['product_name_map']

    # --- Collaborative Filtering (known users) ---
    if user_id in MODELS['user_map_rev']:
        user_idx = MODELS['user_map_rev'][user_id]
        if user_idx < MODELS['als_model'].user_factors.shape[0]:
            try:
                ids, scores = MODELS['als_model'].recommend(
                    user_idx,
                    MODELS['user_item_matrix'],
                    N=k
                )
                reverse_item_map = MODELS['product_map']
                for i, s in zip(ids, scores):
                    if int(i) in reverse_item_map:
                        product_id = reverse_item_map[int(i)]
                        product_name = product_name_map.get(product_id, "Name not available")
                        recs.append(
                            Recommendation(
                                product_id=product_id,
                                product_name=product_name,
                                score=float(s)
                            )
                        )
                    else:
                        print(f"Warning: ALS index {i} not found in product_map, skipping.")
            except Exception as e:
                print(f"ALS recommendation error for user {user_id}: {e}")

    # --- Cold-start / Fallback ---
    if len(recs) < k:
        num_needed = k - len(recs)
        print(f"Adding {num_needed} popular items for fallback for user {user_id}")
        current_recs_ids = {r.product_id for r in recs}
        fallback_items = [pid for pid in MODELS['popular_items'] if pid not in current_recs_ids]
        for pid in fallback_items[:num_needed]:
            product_name = product_name_map.get(str(pid), "Name not available")
            recs.append(
                Recommendation(
                    product_id=str(pid),
                    product_name=product_name,
                    score=0.0
                )
            )

    return RecommendationResponse(user_id=user_id, recommendations=recs[:k],
                                  source="collaborative_filtering" if user_id in MODELS['user_map_rev'] else "popularity_fallback")

@app.get("/explain/{user_id}/{product_id}", response_model=ExplanationResponse)
def get_explanation(user_id: str, product_id: str):
    user_id = user_id.strip()
    product_id = product_id.strip()

    if (product_id not in MODELS['product_map_rev']) and (product_id not in MODELS['product_id_faiss_map']['product_id'].values):
        raise HTTPException(status_code=404, detail=f"Product with ID '{product_id}' not found.")

    explanation_text = generate_explanation_lm_studio(user_id, product_id)
    return ExplanationResponse(
        user_id=user_id,
        product_id=product_id,
        explanation=explanation_text
    )
