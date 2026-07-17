import pandas as pd
import ollama
import json
import re
import os
from tqdm import tqdm

def extract_json_triple(response_text):
    """Extracts a JSON object containing the triple from the LLM output."""
    try:
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return None
    except json.JSONDecodeError:
        return None

def generate_explanation_triples():
    scores_file = "Data/passage_scores.csv"
    output_file = "Data/explanation_triples.csv"

    if not os.path.exists(scores_file):
        print(f"ERROR: Cannot find '{scores_file}'. Run Step 8 first.")
        return

    print("Loading passage scores...")
    df = pd.read_csv(scores_file)

    # 1. Filter for "High-Signal" Passages ONLY
    # We only want to explain strong bias (Positive/Negative) or aggressive framing.
    # We skip "Neutral" sentiment with "Neutral/Factual" framing to save compute time.
    strong_sentiment = df['sentiment'].isin(['Negative', 'Positive'])
    strong_framing = df['framing_label'].isin(['Attack/Defence', 'Victimhood', 'Communal Tension', 'Nationalism', 'Corruption'])
    
    # Also ensure there is actual text evidence to extract a triple from
    has_evidence = df['evidence_span'].notna() & (df['evidence_span'] != 'N/A - No targets detected') & (df['evidence_span'] != 'LLM Parse Error')

    high_signal_df = df[(strong_sentiment | strong_framing) & has_evidence].copy()
    
    # Take only the top clearest examples (Confidence >= 0.8)
    high_signal_df = high_signal_df[high_signal_df['confidence'] >= 0.8]

    # 2. Check for existing processed triples to enable incremental processing
    existing_df = None
    processed_passage_ids = set()
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
            if 'passage_id' in existing_df.columns:
                processed_passage_ids = set(existing_df['passage_id'].dropna().astype(str))
                print(f"Found existing triples database with {len(processed_passage_ids)} processed passages.")
        except Exception as e:
            print(f"Warning: Could not read existing triples file ({e}). Re-processing all high-signal passages.")

    # 3. Filter high-signal passages down to ONLY new, unprocessed ones
    new_high_signal_df = high_signal_df[~high_signal_df['passage_id'].astype(str).isin(processed_passage_ids)].copy()

    if len(new_high_signal_df) == 0:
        print("🎉 No new high-signal passages to process. Everything is up to date!")
        return

    print(f"Found {len(new_high_signal_df)} NEW high-signal passages for triple extraction (skipping {len(processed_passage_ids)} already processed).")
    
    triple_records = []

    for _, row in tqdm(new_high_signal_df.iterrows(), total=len(new_high_signal_df), desc="Extracting Triples"):
        evidence = row['evidence_span']
        target = row['canonical_target']
        sentiment = row['sentiment']
        framing = row['framing_label']

        prompt = f"""
You are a linguistic extraction tool for a UI dashboard. 
Extract a SINGLE, short Subject-Relation-Object (SVO) triple from the "Evidence Quote" below that explains why the target ({target}) received a {sentiment} sentiment score under a {framing} frame.

Evidence Quote: "{evidence}"

Rules:
1. Subject: Who is doing the action?
2. Relation: The verb/action (e.g., "accused", "praised", "attacked", "defended").
3. Object: Who or what is receiving the action?
4. Keep them incredibly short (1-4 words each).

Return ONLY valid JSON in this format:
{{
  "subject": "Congress",
  "relation": "accused",
  "object": "Modi of bulldozing"
}}
"""
        try:
            response = ollama.chat(model='qwen2.5:3b', messages=[
                {'role': 'system', 'content': 'You output pure JSON objects.'},
                {'role': 'user', 'content': prompt}
            ])
            
            triple_data = extract_json_triple(response['message']['content'])
            
            if triple_data:
                triple_records.append({
                    'passage_id': row['passage_id'],
                    'article_id': row['article_id'],
                    'event_id': row['event_id'],
                    'clean_outlet': row['clean_outlet'],
                    'canonical_target': target,
                    'framing_label': framing,
                    'sentiment': sentiment,
                    'evidence_span': evidence,
                    'subject': triple_data.get('subject', 'Unknown'),
                    'relation': triple_data.get('relation', 'Unknown'),
                    'object': triple_data.get('object', 'Unknown')
                })
        except Exception as e:
            pass # Skip on LLM timeout

    # 4. Create a clean DataFrame from the new extractions
    new_triples_df = pd.DataFrame(triple_records)

    # 5. Safely concatenate the fresh updates with the historical corpus
    if existing_df is not None:
        # Match schemas dynamically
        for col in new_triples_df.columns:
            if col not in existing_df.columns:
                existing_df[col] = None
        existing_df = existing_df[new_triples_df.columns]
        final_df = pd.concat([existing_df, new_triples_df], ignore_index=True)
    else:
        final_df = new_triples_df

    # 6. Perform an atomic write to prevent data corruption
    tmp_output = output_file + ".tmp"
    final_df.to_csv(tmp_output, index=False)
    if os.path.exists(tmp_output):
        if os.path.exists(output_file):
            os.remove(output_file)
        os.rename(tmp_output, output_file)

    print("\n=========================================================")
    print("             EXPLANATION TRIPLES GENERATED               ")
    print("=========================================================")
    print(f"Successfully appended {len(new_triples_df)} new UI Explanation Cards.")
    print(f"Total triples database now contains: {len(final_df)} records.")
    print(f"Saved to -> '{output_file}'")
    if len(new_triples_df) > 0:
        print("\nSneak Peek (New Triples):")
        peek = new_triples_df[['subject', 'relation', 'object']].head(3)
        for _, r in peek.iterrows():
            print(f"[{r['subject']}] -> [{r['relation']}] -> [{r['object']}]")
    print("=========================================================\n")

if __name__ == "__main__":
    generate_explanation_triples()