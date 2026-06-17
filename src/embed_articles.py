import pandas as pd
import ollama
from tqdm import tqdm
import os

def generate_embeddings():
    input_file = "Data/prepared_articles_db.csv"
    output_file = "Data/articles_with_embeddings.parquet"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run prep_articles.py first.")
        return

    print("1. Loading prepared articles...")
    df = pd.read_csv(input_file)
    
    # Safety check: Ensure cluster_text exists and handle any accidental NaNs
    if 'cluster_text' not in df.columns or 'embedding_status' not in df.columns:
        print("Error: 'cluster_text' or 'embedding_status' column missing from input file.")
        return
        
    df['cluster_text'] = df['cluster_text'].fillna("").astype(str)
    
    # Filter for completely empty cluster_texts AND where status is 'Pending'
    pending_articles = df[(df['cluster_text'].str.strip() != "") & (df['embedding_status'] == 'Pending')].copy()
    
    if len(pending_articles) == 0:
        print("No pending articles found. Everything is already embedded and up to date.")
        return
        
    print(f"Found {len(pending_articles)} pending articles to embed out of {len(df)} total.")

    # ---------------------------------------------------------
    # 2. INITIALIZE OLLAMA TARGET
    # ---------------------------------------------------------
    model_name = "qwen2.5:3b"
    print(f"2. Connecting to local Ollama using model '{model_name}'...")
    
    # Quick connectivity check
    try:
        ollama.embeddings(model=model_name, prompt="test")
    except Exception as e:
        print(f"Error connecting to Ollama or model '{model_name}' not found.")
        print(f"Details: {e}")
        print(f"Please ensure Ollama is running and you have pulled the model using: 'ollama pull {model_name}'")
        return

    # ---------------------------------------------------------
    # 3. GENERATE EMBEDDINGS
    # ---------------------------------------------------------
    print("3. Generating embeddings... (This may take a moment)")
    texts_to_embed = pending_articles['cluster_text'].tolist()
    embeddings_list = []
    
    # Process each text individually. We use tqdm to recreate the progress bar
    for text in tqdm(texts_to_embed, desc="Embedding"):
        response = ollama.embeddings(model=model_name, prompt=text)
        embeddings_list.append(response['embedding'])
    
    # Assign the lists back to the dataframe
    pending_articles['embedding'] = embeddings_list
    pending_articles['embedding_status'] = "Embedded"

    # ---------------------------------------------------------
    # 4. APPEND TO PARQUET & UPDATE CSV
    # ---------------------------------------------------------
    print("4. Saving to Parquet format and updating CSV...")
    
    # Append to existing parquet file or create a new one if it doesn't exist
    if os.path.exists(output_file):
        existing_parquet = pd.read_parquet(output_file, engine='pyarrow')
        updated_parquet = pd.concat([existing_parquet, pending_articles], ignore_index=True)
        updated_parquet.to_parquet(output_file, index=False, engine='pyarrow')
        print(f"Appended {len(pending_articles)} new embeddings to existing Parquet file.")
    else:
        pending_articles.to_parquet(output_file, index=False, engine='pyarrow')
        print(f"Created new Parquet file with {len(pending_articles)} embeddings.")
    
    # Update the statuses in the original dataframe and save back to CSV
    df.loc[pending_articles.index, 'embedding_status'] = 'Embedded'
    df.to_csv(input_file, index=False)
    print(f"Updated {len(pending_articles)} statuses to 'Embedded' in {input_file}.")
    
    print(f"\nSUCCESS! Incremental update complete.")
    
    # Quick sanity check output
    sample_vector = pending_articles['embedding'].iloc[0]
    print(f"Sample Vector Dimension: {len(sample_vector)}")

if __name__ == "__main__":
    generate_embeddings()