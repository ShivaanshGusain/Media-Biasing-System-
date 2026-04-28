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

    print("Loading raw articles...")
    df = pd.read_csv(input_file, encoding="utf-8")

    print("Ensuring deterministic article_id...")
    if "article_id" not in df.columns:
        df["article_id"] = ""
    df["article_id"] = [ensure_article_id(row, i) for i, (_, row) in enumerate(df.iterrows())]

    print("Repairing headline/lead fields...")
    df[["headline", "lead", "first_paragraph"]] = df.apply(repair_headline_lead, axis=1)

    print("Normalizing headlines...")
    df["lead_clean"] = df["lead"].astype(str).str.lower()
    df["lead_clean"] = df["lead_clean"].apply(
        lambda x: re.sub(r"[^a-z0-9\s]", "", x).strip()
    )
    df["headline_clean"] = df["headline"].astype(str).str.lower()
    df["headline_clean"] = df["headline_clean"].apply(
        lambda x: re.sub(r"[^a-z0-9\s]", "", x).strip()
    )

    print("Normalizing outlets...")
    df["clean_outlet"] = df["outlet"].astype(str).str.lower().str.replace(" ", "_", regex=False)

    print("Parsing dates...")
    df["publish_date_only"] = pd.to_datetime(df["publish_time"], errors="coerce").dt.date

    print("Building canonical cluster_text signatures...")
    df["cluster_text"] = df.apply(build_cluster_text, axis=1)

    print("Setting initial pipeline statuses...")
    df["embedding_status"] = "Pending"
    df["event_id"] = None

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

    final_cols = [c for c in core_cols if c in df.columns] + [c for c in df.columns if c not in core_cols]
    df = df[final_cols]

    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"\nSUCCESS! {len(df)} articles enriched and saved to '{output_file}'.")

    if len(df) > 0:
        print("\nSample 'cluster_text' output:")
        print("-" * 50)
        sample_text = str(df["cluster_text"].iloc[0])
        print(sample_text[:300] + "..." if len(sample_text) > 300 else sample_text)


if __name__ == "__main__":
    prepare_data()