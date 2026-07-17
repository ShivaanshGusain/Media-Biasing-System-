import pandas as pd
import re
import difflib
import os
import hashlib


# ==========================================
# 1. ID + TEXT HELPERS
# ==========================================

def valid_article_id(value):
    if pd.isna(value):
        return False
    v = str(value).strip()
    return v not in {"", "nan", "None", "UNKNOWN", "Unknown","unknown"}


def make_article_id(url):
    if pd.isna(url) or not str(url).strip():
        return ""
    return hashlib.sha256(str(url).strip().encode("utf-8")).hexdigest()[:16]


def ensure_article_id(row, idx):
    existing = row.get("article_id", "")
    if valid_article_id(existing):
        return str(existing).strip()

    generated = make_article_id(row.get("url", ""))
    if generated:
        return generated

    return f"missingurl_{idx:06d}"


def is_unknown(value):
    if value is None or pd.isna(value):
        return True
    v = str(value).strip().lower()
    return v in {"", "unknown", "none", "nan", "null", "n/a", "na"}


def clean_text(value):
    if value is None or pd.isna(value):
        return ""
    
    v = str(value)
    
    replacements = {
        "â€”": "—",   # Em-dash
        "â€“": "-",   # En-dash
        "â€œ": '"',   # Left double quote
        "â€": '"',   # Right double quote
        "â€˜": "'",   # Left single quote
        "â€™": "'",   # Right single quote/apostrophe
        "â€¦": "...", # Ellipsis
        "Â": ""       # Stray circumflex
    }
    for bad, good in replacements.items():
        v = v.replace(bad, good)
        
    return re.sub(r"\s+", " ", v).strip()

def first_sentence(text, max_len=220):
    text = clean_text(text)
    if not text:
        return ""
    parts = re.split(r'(?<=[.!?])\s+', text)
    sent = parts[0].strip() if parts else text
    return sent[:max_len].strip()


# ==========================================
# 2. CLUSTERING HELPERS
# ==========================================

def get_first_n_sentences(text, n=3):
    if not text or pd.isna(text):
        return ""
    sentences = re.split(r'(?<=[.!?])\s+', str(text).strip())
    return " ".join(sentences[:n]).strip()


def is_too_similar(str1, str2, threshold=0.65):
    if not str1 or not str2:
        return False
    similarity = difflib.SequenceMatcher(None, str(str1).lower(), str(str2).lower()).ratio()
    is_subset = str(str1).lower() in str(str2).lower()
    return similarity > threshold or is_subset


def repair_headline_lead(row):
    headline = clean_text(row.get("headline", ""))
    lead = clean_text(row.get("lead", ""))
    first_para = clean_text(row.get("first_paragraph", ""))
    full_text = clean_text(row.get("full_text", ""))

    if not first_para and full_text:
        first_para = get_first_n_sentences(full_text, n=3)

    if is_unknown(headline):
        if lead and len(lead) > 15:
            headline = lead
            if first_para and not is_too_similar(headline, first_para, threshold=0.85):
                lead = first_para
            else:
                lead = ""
        elif first_para:
            headline = first_sentence(first_para)
        elif full_text:
            headline = first_sentence(full_text)

    if is_unknown(lead):
        if first_para and not is_too_similar(headline, first_para, threshold=0.85):
            lead = first_para
        elif full_text:
            candidate = get_first_n_sentences(full_text, n=2)
            if candidate and not is_too_similar(headline, candidate, threshold=0.85):
                lead = candidate

    if headline and lead and is_too_similar(headline, lead, threshold=0.85):
        lead = ""

    return pd.Series({
        "headline": headline,
        "lead": lead,
        "first_paragraph": first_para
    })


def build_cluster_text(row):
    headline = clean_text(row.get("headline", ""))
    lead = clean_text(row.get("lead", ""))
    first_para = clean_text(row.get("first_paragraph", ""))
    full_text = clean_text(row.get("full_text", ""))

    if not first_para and full_text:
        first_para = get_first_n_sentences(full_text, n=3)

    parts = []

    if headline and not is_unknown(headline):
        parts.append(headline)

    use_lead = False
    if lead and len(lead) > 15 and not is_too_similar(headline, lead):
        use_lead = True
        parts.append(lead)

    if first_para:
        if not use_lead or not is_too_similar(lead, first_para, threshold=0.8):
            parts.append(first_para)

    if not parts and full_text:
        parts.append(get_first_n_sentences(full_text, n=3))

    return " [SEP] ".join([p for p in parts if p])


# ==========================================
# 3. MAIN PROCESSING PIPELINE
# ==========================================

def prepare_data():
    input_file = "Data/canonical_articles_db.csv"
    output_file = "Data/prepared_articles_db.csv"

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run collector.py first.")
        return

    print("Loading raw input articles...")
    input_df = pd.read_csv(input_file, encoding="utf-8")

    # 1. Step 1: Ensure deterministic IDs on the raw incoming dataset right away
    if "article_id" not in input_df.columns:
        input_df["article_id"] = ""
    input_df["article_id"] = [ensure_article_id(row, i) for i, (_, row) in enumerate(input_df.iterrows())]

    # 2. Check for existing processed articles to enable incremental processing
    existing_df = None
    existing_ids = set()
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file, encoding="utf-8")
            if "article_id" in existing_df.columns:
                existing_ids = set(existing_df["article_id"].dropna().astype(str))
                print(f"Found existing output database with {len(existing_df)} processed rows.")
        except Exception as e:
            print(f"Warning: Could not read existing output file ({e}). Re-processing all.")

    # 3. Filter down to only completely NEW rows
    new_df = input_df[~input_df["article_id"].astype(str).isin(existing_ids)].copy()
    
    if len(new_df) == 0:
        print("🎉 No new articles to process. Everything is up to date!")
        return

    print(f"Found {len(new_df)} NEW articles to prepare (skipping {len(existing_ids)} already processed).")

    # 4. Only process the new items through the expensive cleaning/repair pipeline
    print("Repairing headline/lead fields for new rows...")
    new_df[["headline", "lead", "first_paragraph"]] = new_df.apply(repair_headline_lead, axis=1)

    print("Normalizing headlines for new rows...")
    new_df["lead_clean"] = new_df["lead"].astype(str).str.lower()
    new_df["lead_clean"] = new_df["lead_clean"].apply(
        lambda x: re.sub(r"[^a-z0-9\s]", "", x).strip()
    )
    new_df["headline_clean"] = new_df["headline"].astype(str).str.lower()
    new_df["headline_clean"] = new_df["headline_clean"].apply(
        lambda x: re.sub(r"[^a-z0-9\s]", "", x).strip()
    )

    print("Normalizing outlets for new rows...")
    new_df["clean_outlet"] = new_df["outlet"].astype(str).str.lower().str.replace(" ", "_", regex=False)

    print("Parsing dates for new rows...")
    new_df["publish_date_only"] = pd.to_datetime(new_df["publish_time"], errors="coerce").dt.date

    print("Building canonical cluster_text signatures for new rows...")
    new_df["cluster_text"] = new_df.apply(build_cluster_text, axis=1)

    print("Setting pipeline statuses for new rows...")
    new_df["embedding_status"] = "Pending"
    new_df["event_id"] = None

    # Organize core columns consistently
    core_cols = [
        "article_id",
        "event_id",
        "clean_outlet",
        "publish_date_only",
        "headline",
        "cluster_text",
        "embedding_status",
        "url",
        "full_text"
    ]
    final_cols = [c for c in core_cols if c in new_df.columns] + [c for c in new_df.columns if c not in core_cols]
    new_df = new_df[final_cols]

    # 5. Merge new rows back with the existing dataset (preserving previous statuses)
    if existing_df is not None:
        # Match column schemas dynamically just in case they differ slightly
        for col in final_cols:
            if col not in existing_df.columns:
                existing_df[col] = None
        existing_df = existing_df[final_cols]
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        final_df = new_df

    # 6. Perform an atomic write to prevent data corruption
    tmp_output = output_file + ".tmp"
    final_df.to_csv(tmp_output, index=False, encoding="utf-8")
    if os.path.exists(tmp_output):
        if os.path.exists(output_file):
            os.remove(output_file)
        os.rename(tmp_output, output_file)

    print(f"\nSUCCESS! {len(new_df)} new rows appended. Output now contains {len(final_df)} total records.")

    if len(new_df) > 0:
        print("\nSample 'cluster_text' output from new rows:")
        print("-" * 50)
        sample_text = str(new_df["cluster_text"].iloc[0])
        print(sample_text[:300] + "..." if len(sample_text) > 300 else sample_text)

if __name__ == "__main__":
    prepare_data()