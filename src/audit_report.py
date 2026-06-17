import pandas as pd
import os

def run_dataset_audit():
    articles_csv = "Data/bias_article_details.csv"
    events_csv = "Data/audited_canonical_events.csv"

    # Checking if file exists
    if not os.path.exists(articles_csv) or not os.path.exists(events_csv):
        print(f"ERROR: Missing input files. Ensure '{articles_csv}' and '{events_csv}' are present.")
        return

    print("Loading datasets for audit...\n")
    articles_df = pd.read_csv(articles_csv)
    events_df = pd.read_csv(events_csv)

    # Clean boolean columns in events_df just in case they were saved as strings
    bool_cols = ['failed_coherence_audit', 'is_template_junk', 'is_low_quality_event']
    for col in bool_cols:
        if col in events_df.columns:
            events_df[col] = events_df[col].astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False}).fillna(False)

    # 2. Calculate Audit Metrics
    total_articles = len(articles_df)
    total_events = len(events_df)

    # Determine outlet counts per event (if not already perfectly accurate in events_df)
    # We can calculate it directly from the articles_df to be safe
    outlet_counts = articles_df.groupby('event_id')['clean_outlet'].nunique().reset_index()
    outlet_counts.rename(columns={'clean_outlet': 'actual_unique_outlets'}, inplace=True)
    events_df = pd.merge(events_df, outlet_counts, on='event_id', how='left')

    # Calculate shares
    single_outlet_events = len(events_df[events_df['actual_unique_outlets'] <= 1])
    low_quality_events = len(events_df[events_df['is_low_quality_event'] == True])
    failed_coherence = len(events_df[events_df['failed_coherence_audit'] == True])
    template_junk = len(events_df[events_df['is_template_junk'] == True])

    # Calculate the Gold Standard Denominator (Tier A + B + C equivalent)
    valid_events_mask = (
        (events_df['actual_unique_outlets'] >= 2) & 
        (events_df['failed_coherence_audit'] == False) & 
        (events_df['is_template_junk'] == False)
    )
    valid_events = events_df[valid_events_mask]['event_id'].tolist()
    
    # How many articles belong to these valid events?
    valid_articles_df = articles_df[articles_df['event_id'].isin(valid_events)]
    total_valid_articles = len(valid_articles_df)

    # 3. Print the Report
    print("=========================================================")
    print("                 DATASET AUDIT REPORT                    ")
    print("=========================================================")
    print(f"1. Total articles in bias_article_details:      {total_articles:,}")
    print(f"2. Total unique events in audited events table: {total_events:,}")
    print("---------------------------------------------------------")
    
    print("EVENT QUALITY BREAKDOWN:")
    print(f" - Single-Outlet Events:   {single_outlet_events:,} ({(single_outlet_events/total_events)*100:.1f}%)")
    if 'is_low_quality_event' in events_df.columns:
        print(f" - Low Quality Flags:      {low_quality_events:,} ({(low_quality_events/total_events)*100:.1f}%)")
    print(f" - Failed Coherence:       {failed_coherence:,} ({(failed_coherence/total_events)*100:.1f}%)")
    print(f" - Template/Junk Pages:    {template_junk:,} ({(template_junk/total_events)*100:.1f}%)")
    
    print("---------------------------------------------------------")
    print("THE DENOMINATOR (Clean, Multi-Outlet, Non-Junk Data):")
    print(f" -> Total Valid Events:    {len(valid_events):,} ({(len(valid_events)/total_events)*100:.1f}% of original events)")
    print(f" -> Total Valid Articles:  {total_valid_articles:,} ({(total_valid_articles/total_articles)*100:.1f}% of original articles)")
    print("=========================================================\n")
    
    print("CONCLUSION:")
    print(f"Your downstream NLP scripts (segmentation, entities, scoring) should ONLY process")
    print(f"the {total_valid_articles:,} articles belonging to these {len(valid_events):,} valid events.")
    print("Processing anything else will introduce noise into your bias calculations.")

if __name__ == "__main__":
    run_dataset_audit()