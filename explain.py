import pandas as pd
import requests
from pathlib import Path

# Load processed data
try:
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / 'data' / 'processed' / 'user_product_interactions.parquet'
    df = pd.read_parquet(data_path)
    product_to_category = df[['product_id', 'category']].drop_duplicates().set_index('product_id')['category']
except FileNotFoundError:
    print("Warning: Processed data file not found. Explanations may be limited.")
    df = None
    product_to_category = None

def get_user_history_summary(user_id: str) -> str:
    if df is None:
        return "No purchase history available."
    user_history = df[df['user_id'] == user_id]
    if user_history.empty:
        return "This is a new user with no purchase history."
    top_categories = user_history['category'].value_counts().nlargest(3).index.tolist()
    if not top_categories:
        return "User has purchased items, but category info is missing."
    return f"This user frequently buys products from categories like: {', '.join(top_categories)}."

def get_product_summary(product_id: str) -> str:
    if product_to_category is None:
        return "Product information not available."
    try:
        return f"This is a product from the '{product_to_category.loc[product_id]}' category."
    except KeyError:
        return f"Product with ID {product_id} not found."

def generate_explanation_lm_studio(user_id: str, product_id: str) -> str:
    user_context = get_user_history_summary(user_id)
    product_context = get_product_summary(product_id)

    url = "http://127.0.0.1:1234/v1/chat/completions"

    system_prompt = (
        "You are a friendly e-commerce assistant. Your only job is to write a single, concise, and positive sentence "
        "explaining a product recommendation. Be creative. Do NOT use lists or multiple paragraphs. "
        "Directly state the reason in one sentence."
    )
    
    user_prompt = (
        f"User's past interest: {user_context}\n"
        f"Recommended product: {product_context}\n\n"
        "Generate one sentence explaining why they might like this. Example: 'Since you enjoy items for your home, you might like this for your active lifestyle.'"
    )

    payload = {
        "model": "microsoft/phi-3-mini-4k-instruct", # Make sure this matches your loaded model in LM Studio
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.9,
        "max_tokens": 50
    }

    try:
        # Increased timeout to give the model more time
        response = requests.post(url, json=payload, timeout=20)
        response.raise_for_status()
        choices = response.json().get('choices', [])
        if choices and 'message' in choices[0]:
            content = choices[0]['message'].get('content', '').strip()
            if content:
                return content
        return "No explanation available. This might be a new user or product."
    except requests.exceptions.RequestException as e:
        print(f"Error calling LM Studio API: {e}")
        return "Could not generate an explanation at this time."
    except Exception as e:
        print(f"Unexpected error in explanation generation: {e}")
        return "Could not generate an explanation at this time."
