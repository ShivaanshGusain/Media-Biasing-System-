import pandas as pd
import os
import json

# Set your paths (Adjust these to match your Kaggle setup)
DATA_DIR = "Data" 
# We output directly into your GitHub Pages folder (which we renamed to 'docs' earlier)
OUTPUT_DIR = "docs/api" 

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# DATA HEALING: Restore missing URLs to the master file before baking
# =====================================================================
master_path = os.path.join(DATA_DIR, "analysis_articles_master.csv")
prep_path = os.path.join(DATA_DIR, "prepared_articles_db.csv")

if os.path.exists(master_path) and os.path.exists(prep_path):
    print("Stitching URLs from prepared database back into master dataset...")
    try:
        master_df = pd.read_csv(master_path, low_memory=False)
        
        # Only run the merge if the URL column was stripped out
        if 'url' not in master_df.columns:
            prep_df = pd.read_csv(prep_path, low_memory=False)
            
            if 'article_id' in prep_df.columns and 'url' in prep_df.columns:
                # Isolate matching columns and drop duplicates to prevent row blowups
                url_mapping = prep_df[['article_id', 'url']].drop_duplicates(subset=['article_id'])
                
                # Sanitize IDs to prevent mismatch traps
                master_df['article_id'] = master_df['article_id'].astype(str).str.strip()
                url_mapping['article_id'] = url_mapping['article_id'].astype(str).str.strip()
                
                # Merge the URLs back in cleanly
                master_df = pd.merge(master_df, url_mapping, on='article_id', how='left')
                master_df.to_csv(master_path, index=False)
                print(" ✓ URLs successfully restored to analysis_articles_master.csv!")
            else:
                print(" ✗ Skipping patch: prepared_articles_db.csv missing critical columns.")
        else:
            print(" ✓ URL column already present in master dataset. Skipping patch.")
    except Exception as e:
        print(f" ✗ Warning: Failed to repair URL columns: {e}")
else:
    print(" ✗ URL Patch skipped: Missing master or prepared article source files.")


# =====================================================================
# API BAKE SYSTEM
# =====================================================================
CSV_MAP = {
    "coverage":     "bias_coverage_statistics.csv",
    "event_matrix": "matrix_event_coverage.csv",
    "overlap":      "matrix_pairwise_overlap.csv",
    "bias":         "event_target_bias_comparison.csv",
    "omission":     "target_omission_matrix.csv",
    "triples":      "explanation_triples.csv",
    "passages":     "passages.csv",
    "articles":     "analysis_articles_master.csv",
}

print("\nBaking static JSON API...")
meta = {}

for key, fname in CSV_MAP.items():
    csv_path = os.path.join(DATA_DIR, fname)
    json_path = os.path.join(OUTPUT_DIR, f"{key}.json")
    
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            df = df.fillna("") # Replace NaN with empty string just like Flask
            
            # Replicate your exact Flask JSON structure!
            api_response = {
                "key": key,
                "file": fname,
                "rows": len(df),
                "columns": list(df.columns),
                "data": df.to_dict(orient="records"),
            }
            
            # Save the JSON file
            with open(json_path, 'w') as f:
                json.dump(api_response, f)
            print(f" ✓ Created {key}.json")
            
            # Populate the meta dictionary
            stat = os.stat(csv_path)
            meta[key] = {
                "file": fname,
                "rows": len(df),
                "last_modified": stat.st_mtime,
                "last_modified_human": pd.Timestamp(stat.st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S"),
                "size_kb": round(stat.st_size / 1024, 1),
            }
        except Exception as e:
            print(f" ✗ Error processing {fname}: {e}")
            meta[key] = {"file": fname, "error": str(e)}
    else:
        print(f" ✗ MISSING {fname}")
        meta[key] = {"file": fname, "exists": False}

# Replicate your /api/meta endpoint
with open(os.path.join(OUTPUT_DIR, "meta.json"), 'w') as f:
    json.dump(meta, f)
    
print(" ✓ Created meta.json")
print("\nStatic bake complete! Ready for GitHub Pages.")