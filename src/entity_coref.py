import pandas as pd
import ollama
import json
import re
import os
from tqdm import tqdm

# Maps known canonical targets. Keys are lowercase substrings — if ANY key appears
# inside the LLM's output string, it maps to the value. Order matters: longer/more
# specific keys should come first so "nitish kumar" matches before bare "nitish".
CANONICAL_MAP = {
    # Parties
    "bharatiya janata party": "BJP",
    "saffron party": "BJP",
    "indian national congress": "Congress",
    "trinamool congress": "TMC",
    "rashtriya janata dal": "RJD",
    "janata dal (united)": "JD(U)",
    "janata dal united": "JD(U)",
    "aam aadmi party": "AAP",
    "shiv sena": "Shiv Sena",
    "national democratic alliance": "NDA",
    "united progressive alliance": "UPA",
    "india alliance": "INDIA Alliance",
    # Abbreviations (exact match only — keep these after the full names)
    "inc": "Congress",
    "jdu": "JD(U)",
    "aitc": "TMC",
    "bjp": "BJP",
    "rdj": "RJD",
    "aap": "AAP",
    "nda": "NDA",
    "upa": "UPA",
    "BRS": "BRS",
    "Zameer Ahmed Khan": "Zameer Ahmed Khan",
    # Politicians
    "narendra modi": "PM Modi",
    "prime minister modi": "PM Modi",
    "modi": "PM Modi",
    "pm modi": "PM Modi",
    "nitish kumar": "Nitish Kumar",
    "nitish": "Nitish Kumar",
    "lalu prasad yadav": "Lalu Prasad Yadav",
    "lalu prasad": "Lalu Prasad Yadav",
    "lalu": "Lalu Prasad Yadav",
    "rahul gandhi": "Rahul Gandhi",
    "rahul": "Rahul Gandhi",
    "amit shah": "Amit Shah",
    "samrat choudhary": "Samrat Choudhary",
    "samrat chaudhary": "Samrat Choudhary",
    "tejashwi yadav": "Tejashwi Yadav",
    "tejashwi": "Tejashwi Yadav",
    "arvind kejriwal": "Arvind Kejriwal",
    "kejriwal": "Arvind Kejriwal",
    "mamata banerjee": "Mamata Banerjee",
    "mamata": "Mamata Banerjee",
    "m.k. stalin": "MK Stalin",
    "mk stalin": "MK Stalin",
    "stalin": "MK Stalin",
    # Institutions
    "election commission of india": "EC",
    "election commission": "EC",
    "supreme court of india": "SC",
    "supreme court": "SC",
    "central bureau of investigation": "CBI",
    "enforcement directorate": "ED",
    "reserve bank": "RBI",
    "parliament": "Parliament",
    "National Democratic Alliance":"NDA",
    # Regions / States (kept as-is, just normalizing casing)
    "bihar": "Bihar",
    "tamil nadu": "Tamil Nadu",
    "west bengal": "West Bengal",
    "delhi": "Delhi",
    "uttar pradesh": "Uttar Pradesh",
}

# These tokens after a name tell us the LLM added a descriptor — strip them
DESCRIPTOR_STRIP_RE = re.compile(
    r'\s*[\(\[].*?[\)\]]'          # anything in brackets: (political party), [BJP leader]
    r'|\s*,?\s*(chief minister|cm|prime minister|pm|president|mp|mla|leader|chief|spokesperson|minister|party|politician|indian|of india)\b.*',
    re.IGNORECASE
)


def clean_llm_canonical(raw: str) -> str:
    """
    Strip descriptive suffixes the LLM adds to canonical names.
    'BJP (Indian political party)' -> 'BJP'
    'JD(U) chief Nitish Kumar'     -> 'Nitish Kumar'  (then CANONICAL_MAP handles it)
    'BJPâ€™s Samrat Choudhary'    -> 'Samrat Choudhary'
    """
    if not raw:
        return ""
    # Fix common encoding artifact (â€™ = right apostrophe)
    cleaned = raw.replace("â€™", "'").replace("â€˜", "'").strip()
    # Remove possessives: "BJP's Samrat Choudhary" -> "Samrat Choudhary"
    cleaned = re.sub(r"^[A-Z][A-Z()/]+\s*'s\s+", "", cleaned)
    # Strip bracketed descriptors and title prefixes
    cleaned = DESCRIPTOR_STRIP_RE.sub("", cleaned).strip().strip(",").strip()
    return cleaned


def standardize_target(raw: str) -> str:
    """
    1. Clean the raw LLM string.
    2. Try exact match in CANONICAL_MAP.
    3. Try substring match (longest key wins).
    4. Fall back to the cleaned string as-is.
    """
    if not raw:
        return "Unknown"

    cleaned = clean_llm_canonical(raw)
    lower = cleaned.lower()

    # Exact match first
    if lower in CANONICAL_MAP:
        return CANONICAL_MAP[lower]

    # Substring match — find all keys that appear in the cleaned string,
    # take the longest one (most specific) to avoid "bjp" matching inside "cbjp-splinter")
    matches = [(k, v) for k, v in CANONICAL_MAP.items() if k in lower]
    if matches:
        best_key, best_val = max(matches, key=lambda x: len(x[0]))
        return best_val

    return cleaned if cleaned else "Unknown"


def extract_json_from_llm(response_text: str) -> list:
    """Robustly extracts a JSON array from LLM output, ignoring prose filler."""
    try:
        match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return []
    except json.JSONDecodeError:
        return []


def run_entity_extraction():
    passages_file = "Data/passages.csv"
    master_file   = "Data/analysis_articles_master.csv"
    output_file   = "Data/passage_entities.csv"

    if not os.path.exists(passages_file):
        print(f"ERROR: Cannot find '{passages_file}'. Run Step 6 first.")
        return

    print("Loading data...")
    passages_df = pd.read_csv(passages_file)
    master_df   = pd.read_csv(master_file)[['article_id', 'headline']]
    df = pd.merge(passages_df, master_df, on='article_id', how='left')

    entity_records = []
    grouped_articles = df.groupby('article_id')

    print(f"Processing {len(df)} passages using Ollama (qwen2.5:3b)...")

    for article_id, group in tqdm(grouped_articles, desc="Articles"):
        article_headline = group['headline'].iloc[0]
        article_memory   = set()
        group = group.sort_values('passage_id')

        for _, row in group.iterrows():
            passage_text = row['passage_text']

            prompt = f"""You are a Political NLP system. Extract political entities from the passage below.
Rules:
- Return ONLY a valid JSON array — no prose, no markdown, no explanation.
- Each object must have exactly two keys: "mention" and "canonical".
- "mention": the exact span of text as it appears in the passage.
- "canonical": the person or party's BARE SHORT NAME only. No titles, no party labels in brackets, no descriptors.
  CORRECT:   "BJP",  "Nitish Kumar",  "PM Modi",  "EC",  "Congress"
  INCORRECT: "BJP (Indian political party)", "JD(U) chief Nitish Kumar", "BJP's Samrat Choudhary"
- Resolve pronouns (he/she/they/the minister/the party) using the headline and memory below.

Headline: "{article_headline}"
Entities seen earlier in this article: {sorted(article_memory)}

Passage:
"{passage_text}"
"""
            try:
                response = ollama.chat(
                    model='qwen2.5:3b',
                    messages=[
                        {'role': 'system', 'content': 'You output only valid JSON arrays. No markdown, no text outside the array.'},
                        {'role': 'user',   'content': prompt}
                    ]
                )

                extracted_data = extract_json_from_llm(response['message']['content'])

                for item in extracted_data:
                    mention     = str(item.get('mention', '')).strip()
                    raw_canonical = str(item.get('canonical', '')).strip()

                    if not mention or not raw_canonical:
                        continue

                    final_target = standardize_target(raw_canonical)
                    if final_target == "Unknown":
                        continue

                    article_memory.add(final_target)

                    entity_records.append({
                        'article_id':      row['article_id'],
                        'event_id':        row['event_id'],
                        'passage_id':      row['passage_id'],
                        'entity_mention':  mention,
                        'canonical_target': final_target
                    })

            except Exception:
                pass

    entities_df = pd.DataFrame(entity_records)
    entities_df.to_csv(output_file, index=False)

    print("\n=========================================================")
    print("            ENTITY & COREF RESOLUTION COMPLETE           ")
    print("=========================================================")
    print(f"Total entity mentions mapped:        {len(entities_df):,}")
    print(f"Unique canonical targets identified: {entities_df['canonical_target'].nunique():,}")
    print(f"Saved to -> '{output_file}'")
    print("=========================================================\n")
    print("Top canonical targets:")
    print(entities_df['canonical_target'].value_counts().head(10).to_string())


if __name__ == "__main__":
    run_entity_extraction()