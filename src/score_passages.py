import pandas as pd
import ollama
import json
import os
from tqdm import tqdm

def clean_entities_for_scoring(df):
    """Unifies fragmented LLM outputs into single, clean targets."""
    if df.empty: return df
    
    cleanup_map = {
        # PM Modi variations
        "Modi": "PM Modi", "PMModi": "PM Modi", "Narendra Modi": "PM Modi", 
        # Shah variations
        "Shah": "Amit Shah", "SHAH": "Amit Shah", 
        # Mamata variations
        "MB": "Mamata Banerjee", "Banerjee": "Mamata Banerjee", "Mamata": "Mamata Banerjee",
        # Nitish variations
        "Kumar": "Nitish Kumar",
        # Party variations
        "JD": "JD(U)", "Con": "Congress", "Cong": "Congress", 
        "Loksabha": "Lok Sabha", "LS": "Lok Sabha",
        "SP": "Samajwadi Party", "aiadmk": "AIADMK", "dmk": "DMK", "TMC": "Trinamool Congress"
    }
    
    # Apply cleanup map
    df['canonical_target'] = df['canonical_target'].replace(cleanup_map)
    
    # Filter out obvious non-political entities that slipped through
    ignore_list = ['None', 'Delhi', 'Tamil Nadu', 'Bengal', 'Del', 'Constitution', 'Home', '2026']
    df = df[~df['canonical_target'].isin(ignore_list)]
    
    return df

def score_passages_incrementally():
    input_file = "Data/passages.csv"
    output_file = "Data/passage_scores.csv"

    if not os.path.exists(input_file):
        print(f"ERROR: Cannot find '{input_file}'. Please run segment_passage.py first.")
        return

    # ---------------------------------------------------------
    # 1. CHECK WHAT HAS ALREADY BEEN SCORED
    # ---------------------------------------------------------
    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
        scored_passage_ids = set(df_existing['passage_id'].dropna().unique())
    else:
        df_existing = pd.DataFrame()
        scored_passage_ids = set()

    # ---------------------------------------------------------
    # 2. ISOLATE NEW PASSAGES
    # ---------------------------------------------------------
    print(f"Loading {input_file}...")
    df_passages = pd.read_csv(input_file)
    
    df_new = df_passages[~df_passages['passage_id'].isin(scored_passage_ids)].copy()

    if df_new.empty:
        print("No new passages to score. 'passage_scores.csv' is completely up to date!")
        return

    print(f"Found {len(df_new)} NEW passages to score (out of {len(df_passages)} total).")
    print("Connecting to local Ollama LLM...")

    # ---------------------------------------------------------
    # 3. LLM SCORING (Only for new passages)
    # ---------------------------------------------------------
    new_scored_records = []

    # System prompt forces strict JSON output with a root OBJECT
    system_prompt = """You are an expert political science AI. 
    Analyze the text and extract political framing.
    Return ONLY a valid JSON object with a single key "records".
    "records" must contain an array of objects.
    Each object must have exactly these keys:
    "target" (Name of politician or party)
    "sentiment" (Positive, Negative, or Neutral)
    "frame" (e.g. Corruption, Development, Protest, Election Strategy, Violence, Law & Order, Policy)
    "confidence" (0.0 to 1.0)
    "evidence_span" (The exact short quote proving this)"""

    for _, row in tqdm(df_new.iterrows(), total=len(df_new)):
        passage_id = row['passage_id']
        text = str(row['passage_text'])

        prompt = f"Text to analyze:\n{text}"

        try:
            # ADDED: format='json' to force valid JSON output from the model
            response = ollama.chat(model='qwen2.5:3b', messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ], format='json')
            
            raw_output = response['message']['content']
            
            # Because of format='json', we can parse it directly!
            parsed_json = json.loads(raw_output)
            results = parsed_json.get('records', [])
            
            for r in results:
                new_scored_records.append({
                    'passage_id': passage_id,
                    'article_id': row['article_id'],
                    'event_id': row['event_id'],
                    'clean_outlet': row['clean_outlet'],
                    'framing_label': r.get('frame', 'Unknown'),
                    'canonical_target': r.get('target', 'Unknown'),
                    'sentiment': r.get('sentiment', 'Neutral'),
                    'confidence': r.get('confidence', 0.5),
                    'evidence_span': r.get('evidence_span', '')
                })

        except Exception as e:
            # Mark it as Parse Error so we don't get stuck re-trying it forever
            new_scored_records.append({
                'passage_id': passage_id,
                'article_id': row['article_id'],
                'event_id': row['event_id'],
                'clean_outlet': row['clean_outlet'],
                'framing_label': 'Parse Error',
                'canonical_target': 'Unknown',
                'sentiment': 'Neutral',
                'confidence': 0.0,
                'evidence_span': f'LLM Parse Error: {str(e)}'
            })

    # ---------------------------------------------------------
    # 4. CLEAN AND SAVE
    # ---------------------------------------------------------
    df_new_scores = pd.DataFrame(new_scored_records)
    
    if not df_new_scores.empty:
        df_new_scores = clean_entities_for_scoring(df_new_scores)

    if not df_existing.empty:
        df_final = pd.concat([df_existing, df_new_scores], ignore_index=True)
    else:
        df_final = df_new_scores

    # Atomic Write: Save to a temp file, then rename (prevents server crashes)
    temp_file = output_file + ".tmp"
    df_final.to_csv(temp_file, index=False)
    os.replace(temp_file, output_file)

    print("\n=========================================================")
    print("                PASSAGE SCORING COMPLETE                 ")
    print("=========================================================")
    print(f"Processed {len(df_new)} new passages.")
    print(f"Generated {len(df_new_scores):,} new score records.")
    print(f"Total scored records in database now: {len(df_final):,}")
    print(f"Saved to -> '{output_file}'")
    print("=========================================================\n")

if __name__ == "__main__":
    score_passages_incrementally()