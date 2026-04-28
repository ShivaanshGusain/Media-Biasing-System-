import pandas as pd
import os

def split_corpora():
    input_file = "Data/analysis_articles_master.csv"
    
    if not os.path.exists(input_file):
        print(f"ERROR: Cannot find '{input_file}'. Please ensure it is in the directory.")
        return
        
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    
    # 1. Clean boolean columns to be safe
    for col in ['failed_coherence_audit', 'is_template_junk']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map({'true': True, 'false': False}).fillna(False)
            
    # 2. Filter out the absolute junk
    # We don't want to analyze incoherent text or generic index pages at all
    print("Filtering out audited junk and template pages...")
    valid_df = df[(~df['failed_coherence_audit']) & (~df['is_template_junk'])].copy()
    
    # 3. Create the Shared-Event Corpus (The "Framing Bias" Arena)
    # Events covered by 2 or more unique outlets
    shared_mask = valid_df['unique_outlet_count'] >= 2
    shared_corpus = valid_df[shared_mask].copy()
    
    # 4. Create the Exclusive-Event Corpus (The "Editorial Agenda" Arena)
    # Events covered by exactly 1 outlet
    exclusive_mask = valid_df['unique_outlet_count'] == 1
    exclusive_corpus = valid_df[exclusive_mask].copy()
    
    # 5. Save the corpora
    shared_file = "Data/corpus_shared_events.csv"
    exclusive_file = "Data/corpus_exclusive_events.csv"
    
    shared_corpus.to_csv(shared_file, index=False)
    exclusive_corpus.to_csv(exclusive_file, index=False)
    
    # 6. Print the summary
    print("\n=========================================================")
    print("                 CORPORA SPLIT COMPLETE                  ")
    print("=========================================================")
    print(f"1. SHARED CORPUS -> Saved to '{shared_file}'")
    print(f"   - Total Articles: {len(shared_corpus)}")
    print(f"   - Total Events:   {shared_corpus['event_id'].nunique()}")
    print("   * Use this for direct cross-outlet framing comparisons.")
    print("---------------------------------------------------------")
    print(f"2. EXCLUSIVE CORPUS -> Saved to '{exclusive_file}'")
    print(f"   - Total Articles: {len(exclusive_corpus)}")
    print(f"   - Total Events:   {exclusive_corpus['event_id'].nunique()}")
    print("   * Use this to study what stories specific outlets push alone.")
    print("=========================================================\n")

if __name__ == "__main__":
    split_corpora()