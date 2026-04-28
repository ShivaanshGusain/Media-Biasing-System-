import pandas as pd
import os

def create_master_table():
    articles_csv = "Data/bias_article_details.csv"
    events_csv = "Data/audited_canonical_events.csv"
    output_csv = "Data/analysis_articles_master.csv"

    # 1. Verify files exist
    if not os.path.exists(articles_csv) or not os.path.exists(events_csv):
        print(f"ERROR: Missing input files. Ensure '{articles_csv}' and '{events_csv}' are in the same folder.")
        return

    print(f"Loading {articles_csv}...")
    articles_df = pd.read_csv(articles_csv)
    
    print(f"Loading {events_csv}...")
    events_df = pd.read_csv(events_csv)

    # 2. Merge the tables on event_id
    # We use an 'inner' join so we only keep articles that belong to audited events
    print("Merging tables on 'event_id'...")
    master_df = pd.merge(articles_df, events_df, on="event_id", how="inner", suffixes=("", "_event"))

    # 3. Standardize column names if needed
    # (Mapping names from earlier scripts to the standardized Step 1 output)
    rename_mapping = {
        "total_articles_published": "article_count",
        "unique_outlets_covering": "unique_outlet_count"
    }
    master_df.rename(columns=rename_mapping, inplace=True)

    # 4. Define the target schema
    target_columns = [
        "article_id",
        "event_id",
        "clean_outlet",
        "publish_time", 
        "publish_date_only",
        "headline", 
        "full_text",
        "low_quality", 
        "is_wire_copy",
        "article_count", 
        "unique_outlet_count",
        "cluster_confidence",
        "is_low_quality_event",
        "failed_coherence_audit",
        "is_template_junk"
    ]

    # Keep only the columns that actually exist (to prevent KeyErrors if a column is missing)
    final_columns = [col for col in target_columns if col in master_df.columns]
    master_df = master_df[final_columns]

    # 5. Save the Output
    master_df.to_csv(output_csv, index=False)
    print(f"\nSUCCESS! Master table created with {len(master_df)} articles.")
    print(f"Saved to -> '{output_csv}'")

    # 6. Print a quick diagnostic audit
    print("\n--- Diagnostic Audit ---")
    print(f"Total Unique Events: {master_df['event_id'].nunique()}")
    print(f"Total Unique Outlets: {master_df['clean_outlet'].nunique()}")
    print("\nMissing Values Check:")
    print(master_df.isnull().sum()[master_df.isnull().sum() > 0]) # Only print cols with missing data

if __name__ == "__main__":
    create_master_table()