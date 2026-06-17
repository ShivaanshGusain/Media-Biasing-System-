import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import re

def extract_basic_entities(text):
    if pd.isna(text): return set()
    entities = re.findall(r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', str(text)[1:])
    return set(entities)

def clean_tokens(text):
    if pd.isna(text) or not str(text).strip():
        return set()
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if len(t) > 2]
    stop = {"the", "and", "for", "with", "from", "that", "this", "have", "has", "had", "after", "into", "will", "were", "been", "their", "about", "over", "under", "said", "says", "more", "than", "what", "when", "where", "which", "while", "india", "indian", "news"}
    return set(t for t in tokens if t not in stop)

def token_overlap_score(text1, text2):
    a = clean_tokens(text1)
    b = clean_tokens(text2)
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)

def is_unknown(value):
    if value is None or pd.isna(value): return True
    v = str(value).strip().lower()
    return v in {"", "unknown", "none", "nan", "null", "n/a", "na"}

def safe_text(row, primary, fallback=None):
    value = row.get(primary, "")
    if not is_unknown(value): return str(value).strip()
    if fallback:
        fb = row.get(fallback, "")
        if not is_unknown(fb): return str(fb).strip()
    return ""

def is_low_quality_row(row):
    url = str(row.get("url", "")).lower()
    headline = safe_text(row, "headline", "headline_clean").lower()
    cluster_text = str(row.get("cluster_text", "")).lower()
    bad_url_patterns = ["liveblog", "web-stories", "/photos/", "/videos/", "/video/", "/gallery/"]
    if any(p in url for p in bad_url_patterns): return True
    if headline in {"", "unknown"}: return True
    if cluster_text.count("[SEP]") > 0:
        parts = [p.strip().lower() for p in cluster_text.split("[SEP]")]
        if len(parts) >= 2 and parts[0] and parts[1] and parts[0] == parts[1]: return True
    return False

def row_headline(row):
    return safe_text(row, "headline", "headline_clean")

def row_lead(row):
    return safe_text(row, "lead", "lead_clean")

def row_match_score(row_a, row_b, emb_a, emb_b):
    sim = cosine_similarity(emb_a.reshape(1, -1), emb_b.reshape(1, -1))[0, 0]
    h_a = row_headline(row_a)
    h_b = row_headline(row_b)
    headline_overlap = token_overlap_score(h_a, h_b)
    lead_overlap = token_overlap_score(row_lead(row_a), row_lead(row_b))
    ent_a = extract_basic_entities(h_a)
    ent_b = extract_basic_entities(h_b)
    ent_overlap = 0.0
    if ent_a and ent_b:
        ent_overlap = len(ent_a & ent_b) / min(len(ent_a), len(ent_b))
    return sim, headline_overlap, lead_overlap, ent_overlap

def get_centroid(embeddings_list):
    mean_vec = np.mean(embeddings_list, axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm == 0: return mean_vec
    return mean_vec / norm

def cluster_articles():
    input_file = "Data/articles_with_embeddings.parquet"
    output_file = "Data/clustered_events_db.csv"

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print("1. Loading embeddings...")
    df_parquet = pd.read_parquet(input_file)

    print("2. Loading existing clustered database (if any)...")
    existing_article_ids = set()
    clusters = []
    next_id = 1
    
    # --- INCREMENTAL LOGIC ---
    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
        df_existing["publish_date"] = pd.to_datetime(df_existing["publish_date_only"], errors="coerce")
        existing_article_ids = set(df_existing["article_id"].dropna().unique())
        
        # Merge the embeddings back in from the parquet file to reconstruct existing centroids
        df_existing_with_emb = df_existing.merge(df_parquet[['article_id', 'embedding']], on='article_id', how='left')
        
        # Rebuild existing cluster representations
        grouped = df_existing_with_emb.groupby("event_id")
        for event_id, group in grouped:
            if not str(event_id).startswith("EVT_"): continue
            
            # Keep track of the highest event_id to resume numbering seamlessly
            try:
                e_num = int(event_id.replace("EVT_", ""))
                if e_num >= next_id: next_id = e_num + 1
            except: pass
            
            valid_embs = [emb for emb in group['embedding'].values if isinstance(emb, np.ndarray)]
            if not valid_embs: continue
            
            centroid = get_centroid(valid_embs)
            rep_row = group.iloc[0] # first row acts as text representative
            
            clusters.append({
                "event_id": event_id,
                "centroid": centroid,
                "all_embeddings": valid_embs,
                "min_date": group["publish_date"].min(),
                "max_date": group["publish_date"].max(),
                "rep_row": rep_row 
            })
        print(f" -> Loaded {len(clusters)} existing clusters.")
    else:
        df_existing = pd.DataFrame()

    print("3. Preparing NEW fields...")
    # Only grab articles that are NOT in the existing CSV
    df_new = df_parquet[~df_parquet["article_id"].isin(existing_article_ids)].copy()
    
    if len(df_new) == 0:
        print("No new articles to cluster. Everything is up to date.")
        return
        
    print(f" -> Found {len(df_new)} new articles to cluster.")
    
    df_new["publish_date"] = pd.to_datetime(df_new["publish_date_only"], errors="coerce")
    df_new["low_quality"] = df_new.apply(is_low_quality_row, axis=1)

    work_df = df_new[~df_new["low_quality"]].copy()
    
    print("4. Sorting candidate rows by date...")
    work_df = work_df.sort_values(by=["publish_date", "article_id"], na_position="last").reset_index(drop=True)

    if len(work_df) > 0:
        embeddings = np.stack(work_df["embedding"].values)

    # --- TUNED THRESHOLDS ---
    HIGH_SIM_THRESHOLD = 0.995         # Increased from 0.98 to account for LLM vector anisotropy
    MODERATE_SIM_THRESHOLD = 0.85      
    HEADLINE_OVERLAP_THRESHOLD = 0.15  
    ENTITY_OVERLAP_THRESHOLD = 0.33    
    TIME_WINDOW_DAYS = 3               
    MAX_CLUSTER_SPAN_DAYS = 4          

    print("5. Assigning NEW articles to clusters...")
    assignments = {}

    for i in range(len(work_df)):
        row_i = work_df.iloc[i]
        emb_i = embeddings[i]
        date_i = row_i["publish_date"]
        art_id = row_i["article_id"]

        best_cluster_idx = None
        best_score = -1.0

        for c_idx, cluster in enumerate(clusters):
            cluster_min_date = cluster["min_date"]
            cluster_max_date = cluster["max_date"]

            if pd.isna(date_i) or pd.isna(cluster_min_date) or pd.isna(cluster_max_date):
                continue

            if (max(cluster_max_date, date_i) - min(cluster_min_date, date_i)).days > MAX_CLUSTER_SPAN_DAYS:
                continue
            
            if abs((date_i - cluster_max_date).days) > TIME_WINDOW_DAYS:
                continue

            centroid_emb = cluster["centroid"]
            row_rep = cluster["rep_row"]

            sim, h_overlap, l_overlap, ent_overlap = row_match_score(row_i, row_rep, emb_i, centroid_emb)

            if sim < MODERATE_SIM_THRESHOLD:
                continue

            # Strict requirement: Now catches similarities up to 0.995 to prevent the EVT_00001 dump
            if sim < HIGH_SIM_THRESHOLD:
                if ent_overlap < ENTITY_OVERLAP_THRESHOLD and h_overlap < HEADLINE_OVERLAP_THRESHOLD:
                    continue

            cluster_score = sim + (0.10 * h_overlap) + (0.20 * ent_overlap)
            
            if cluster_score > best_score:
                best_score = cluster_score
                best_cluster_idx = c_idx

        # Assign to the best found cluster or create a new one
        if best_cluster_idx is None:
            new_event_id = f"EVT_{next_id:05d}"
            next_id += 1
            
            clusters.append({
                "event_id": new_event_id,
                "centroid": emb_i,
                "all_embeddings": [emb_i],
                "min_date": date_i,
                "max_date": date_i,
                "rep_row": row_i
            })
            assignments[art_id] = new_event_id
        else:
            cluster = clusters[best_cluster_idx]
            assignments[art_id] = cluster["event_id"]
            
            # Update the Centroid dynamically
            cluster["all_embeddings"].append(emb_i)
            cluster["centroid"] = get_centroid(cluster["all_embeddings"])
            
            if not pd.isna(date_i):
                cluster["min_date"] = min(cluster["min_date"], date_i)
                cluster["max_date"] = max(cluster["max_date"], date_i)

    print("6. Writing event IDs to new DataFrame...")
    df_new["event_id"] = df_new["article_id"].map(assignments)

    # Assign standalone events for low-quality or skipped rows
    for idx in df_new.index:
        if pd.isna(df_new.at[idx, "event_id"]):
            df_new.at[idx, "event_id"] = f"EVT_{next_id:05d}"
            next_id += 1

    print("7. Saving and Appending to clustered database...")
    # Drop embedding arrays before saving to keep the CSV clean
    output_df_new = df_new.drop(columns=["embedding", "publish_date"], errors="ignore")
    
    if not df_existing.empty:
        df_final = pd.concat([df_existing, output_df_new], ignore_index=True)
    else:
        df_final = output_df_new
        
    df_final = df_final.sort_values(by=["event_id", "publish_date_only", "article_id"], na_position="last")
    df_final.to_csv(output_file, index=False)

    print(f"\nSUCCESS! Appended {len(df_new)} new articles.")
    print(f"Total database now contains {len(df_final)} articles across {df_final['event_id'].nunique()} events.")

if __name__ == "__main__":
    cluster_articles()