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

    print(f"Filtered down to {len(high_signal_df)} high-signal passages for triple extraction...")
    
    triple_records = []

    for _, row in tqdm(high_signal_df.iterrows(), total=len(high_signal_df), desc="Extracting Triples"):
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

    # Save the output
    triples_df = pd.DataFrame(triple_records)
    triples_df.to_csv(output_file, index=False)

    print("\n=========================================================")
    print("             EXPLANATION TRIPLES GENERATED               ")
    print("=========================================================")
    print(f"Successfully generated {len(triples_df)} UI Explanation Cards.")
    print(f"Saved to -> '{output_file}'")
    if len(triples_df) > 0:
        print("\nSneak Peek:")
        peek = triples_df[['subject', 'relation', 'object']].head(3)
        for _, r in peek.iterrows():
            print(f"[{r['subject']}] -> [{r['relation']}] -> [{r['object']}]")
    print("=========================================================\n")

if __name__ == "__main__":
    generate_explanation_triples()