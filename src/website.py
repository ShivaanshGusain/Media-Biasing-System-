import pandas as pd
import os
import json

# Set your paths (Adjust these to match your Kaggle setup)
DATA_DIR = "Data" 
# We output directly into your GitHub Pages folder (which we renamed to 'docs' earlier)
OUTPUT_DIR = "docs/api" 

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

print("Baking static JSON API...")
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