import pandas as pd
import os

def apply_event_filtering():
    input_csv = "Data/analysis_articles_master.csv"
    output_csv = "Data/filtered_analysis_master.csv"

    if not os.path.exists(input_csv):
        print(f"ERROR: '{input_csv}' not found. Please run Step 1 first.")
        return

    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)

    # 1. Ensure safety of boolean columns (handling potential string 'True'/'False')
    bool_cols = ['failed_coherence_audit', 'is_template_junk', 'is_low_quality_event']
    for col in bool_cols:
        if col in df.columns:
            # Convert to actual booleans in case they were saved as strings
            df[col] = df[col].astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False}).fillna(False)

    # 2. Apply Filtering Masks based on your policy
    
    # Tier A: Coherent, not junk, at least 2 outlets
    tier_a_mask = (
        (~df['failed_coherence_audit']) & 
        (~df['is_template_junk']) & 
        (df['unique_outlet_count'] >= 2)
    )

    # Tier B: Tier A + not a low-quality event
    tier_b_mask = tier_a_mask & (~df['is_low_quality_event'])

    # Tier C: Tier B + at least 3 outlets
    tier_c_mask = tier_b_mask & (df['unique_outlet_count'] >= 3)

    # 3. Assign the Tiers (We apply lowest to highest so the highest tier overrides)
    df['analysis_tier'] = 'Junk / Singleton'
    df.loc[tier_a_mask, 'analysis_tier'] = 'Tier A (Experimental)'
    df.loc[tier_b_mask, 'analysis_tier'] = 'Tier B (Standard)'
    df.loc[tier_c_mask, 'analysis_tier'] = 'Tier C (High Confidence)'

    # 4. Save the filtered master dataset
    df.to_csv(output_csv, index=False)
    
    # 5. Print the Audit Report (Transitioning nicely into Step 3)
    print("\n=========================================")
    print("        STEP 3: DATASET AUDIT REPORT       ")
    print("=========================================")
    print(f"Total initial articles: {len(df)}")
    print(f"Total unique events:    {df['event_id'].nunique()}")
    
    print("\n--- Event Distribution by Tier ---")
    events_per_tier = df.groupby('analysis_tier')['event_id'].nunique()
    for tier, count in events_per_tier.items():
        print(f"{tier.ljust(25)}: {count} events")

    print("\n--- Article Distribution by Tier ---")
    articles_per_tier = df['analysis_tier'].value_counts()
    for tier, count in articles_per_tier.items():
        print(f"{tier.ljust(25)}: {count} articles")

    print("\n--- Why events were disqualified (Junk/Singleton reasons) ---")
    singletons = len(df[df['unique_outlet_count'] < 2]['event_id'].unique())
    failed_coherence = len(df[df['failed_coherence_audit'] == True]['event_id'].unique())
    template_junk = len(df[df['is_template_junk'] == True]['event_id'].unique())
    
    print(f"Single-outlet events:       {singletons}")
    print(f"Failed coherence audit:     {failed_coherence}")
    print(f"Flagged as template junk:   {template_junk}")
    
    print(f"\nSUCCESS: Labeled dataset saved to -> '{output_csv}'")
    print("Note: You can now filter this dataset easily using the 'analysis_tier' column.")

if __name__ == "__main__":
    apply_event_filtering()