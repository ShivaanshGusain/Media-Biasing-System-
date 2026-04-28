import pandas as pd
import os
import re

def segment_passages_incrementally():
    # Adjust paths if your files are inside a "Data/" folder
    input_file = "Data/corpus_shared_events.csv"
    output_file = "Data/passages.csv"

    if not os.path.exists(input_file):
        print(f"ERROR: Cannot find '{input_file}'. Please run split_corpora first.")
        return

    # ---------------------------------------------------------
    # 1. CHECK WHAT HAS ALREADY BEEN PROCESSED
    # ---------------------------------------------------------
    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
        # Get a set of article IDs we have already segmented
        processed_article_ids = set(df_existing['article_id'].dropna().unique())
    else:
        df_existing = pd.DataFrame()
        processed_article_ids = set()

    # ---------------------------------------------------------
    # 2. ISOLATE NEW ARTICLES
    # ---------------------------------------------------------
    print(f"Loading {input_file}...")
    df_in = pd.read_csv(input_file)
    
    df_new = df_in[~df_in['article_id'].isin(processed_article_ids)].copy()

    if df_new.empty:
        print("✅ No new articles to segment. 'passages.csv' is completely up to date!")
        return

    print(f"Found {len(df_new)} NEW articles to segment (out of {len(df_in)} total shared articles).")

    # ---------------------------------------------------------
    # 3. SEGMENTATION LOGIC (Your excellent chunking rules)
    # ---------------------------------------------------------
    passages_data = []

    for _, row in df_new.iterrows():
        article_id = row['article_id']
        event_id = row['event_id']
        outlet = row['clean_outlet']
        pub_time = row.get('publish_time', '')
        
        # Take full_text or fall back to headline
        text = str(row.get('full_text', ''))
        if text.strip().lower() in ['nan', 'none', '']:
            text = str(row.get('headline', ''))

        # Split into initial paragraph units
        raw_paragraphs = re.split(r'\n+', text)

        passage_count = 1
        for p in raw_paragraphs:
            p_clean = p.strip()

            # Filter out obvious junk/metadata lines
            if len(p_clean) < 30 or "Read more:" in p_clean or "Click here" in p_clean:
                continue

            # If the paragraph is massive, split it by sentences
            if len(p_clean) > 800:
                sentences = re.split(r'(?<=[.!?])\s+', p_clean)
                current_chunk = ""
                for s in sentences:
                    if len(current_chunk) + len(s) < 600:
                        current_chunk += s + " "
                    else:
                        passage_id = f"{article_id}_p{passage_count:03d}"
                        passages_data.append({
                            'passage_id': passage_id,
                            'article_id': article_id,
                            'event_id': event_id,
                            'clean_outlet': outlet,
                            'publish_time': pub_time,
                            'passage_text': current_chunk.strip()
                        })
                        passage_count += 1
                        current_chunk = s + " "
                
                # Catch the remainder
                if current_chunk.strip():
                    passage_id = f"{article_id}_p{passage_count:03d}"
                    passages_data.append({
                        'passage_id': passage_id,
                        'article_id': article_id,
                        'event_id': event_id,
                        'clean_outlet': outlet,
                        'publish_time': pub_time,
                        'passage_text': current_chunk.strip()
                    })
                    passage_count += 1
            else:
                # Standard paragraph length -> keep as one passage unit
                passage_id = f"{article_id}_p{passage_count:03d}"
                passages_data.append({
                    'passage_id': passage_id,
                    'article_id': article_id,
                    'event_id': event_id,
                    'clean_outlet': outlet,
                    'publish_time': pub_time,
                    'passage_text': p_clean
                })
                passage_count += 1

    # ---------------------------------------------------------
    # 4. APPEND AND SAVE
    # ---------------------------------------------------------
    df_new_passages = pd.DataFrame(passages_data)
    
    if not df_existing.empty:
        df_final = pd.concat([df_existing, df_new_passages], ignore_index=True)
    else:
        df_final = df_new_passages

    # Atomic Write: Save to a temp file, then rename (prevents server crashes)
    temp_file = output_file + ".tmp"
    df_final.to_csv(temp_file, index=False)
    os.replace(temp_file, output_file)

    print("\n=========================================================")
    print("                 SEGMENTATION COMPLETE                   ")
    print("=========================================================")
    print(f"Processed {len(df_new)} new articles.")
    print(f"Generated {len(df_new_passages):,} new passage units.")
    print(f"Total passages in database now: {len(df_final):,}")
    print(f"Saved to -> '{output_file}'")
    print("=========================================================\n")

if __name__ == "__main__":
    segment_passages_incrementally()