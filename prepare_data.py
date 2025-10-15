import pandas as pd
from pathlib import Path

def prepare_olist_data():
    project_root = Path(__file__).resolve().parents[2]
    raw_data_path = project_root / 'data' / 'raw'
    processed_data_path = project_root / 'data' / 'processed'
    processed_data_path.mkdir(exist_ok=True)

    print("Loading datasets...")
    try:
        customers = pd.read_csv(raw_data_path / 'olist_customers_dataset.csv')
        orders = pd.read_csv(raw_data_path / 'olist_orders_dataset.csv')
        order_items = pd.read_csv(raw_data_path / 'olist_order_items_dataset.csv')
        products = pd.read_csv(raw_data_path / 'olist_products_dataset.csv')
        product_translation = pd.read_csv(raw_data_path / 'product_category_name_translation.csv')
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    print("Merging datasets...")
    df = pd.merge(orders, customers, on='customer_id')
    df = pd.merge(df, order_items, on='order_id')
    df = pd.merge(df, products, on='product_id')
    df = pd.merge(df, product_translation, on='product_category_name')

    print("Cleaning and feature engineering...")
    cols_to_keep = ['customer_unique_id', 'product_id', 'product_category_name_english', 'order_purchase_timestamp', 'price']
    df = df[cols_to_keep]
    
    df['customer_unique_id'] = df['customer_unique_id'].str.strip()
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['interaction'] = 1
    df.rename(columns={'customer_unique_id': 'user_id', 'product_category_name_english': 'category'}, inplace=True)
    df = df.sort_values(by=['user_id', 'order_purchase_timestamp'])

    output_path = processed_data_path / 'user_product_interactions.parquet'
    print(f"Saving processed data to {output_path}...")
    df.to_parquet(output_path, index=False)
    print("✅ Data preparation complete!")

if __name__ == '__main__':
    prepare_olist_data()
