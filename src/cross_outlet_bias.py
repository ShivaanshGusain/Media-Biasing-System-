import pandas as pd
import os

def normalize_entities(df):
    """Final cleanup of LLM entity fragmentation before aggregation."""
    cleanup_map = {
        "Prime Minister Narendra Modi": "PM Modi",
        "Narendra Modi": "PM Modi",
        "Congress party": "Congress",
        "Government of India": "Central Government",
        "Lower House of the Indian Parliament": "Lok Sabha"
    }
    df['canonical_target'] = df['canonical_target'].replace(cleanup_map)
    return df

def generate_bias_comparisons():
    scores_file = "Data/passage_scores.csv"
    output_file = "Data/event_target_bias_comparison.csv"

    if not os.path.exists(scores_file):
        print(f"ERROR: Cannot find '{scores_file}'. Run Step 8 first.")
        return

    print("Loading passage scores...")
    df = pd.read_csv(scores_file)

    # 1. Clean up missing values and entity aliases
    df['canonical_target'] = df['canonical_target'].fillna('None')
    df = normalize_entities(df)

    # 2. Filter out the Background/No-Target passages
    # We only want to compare how actual political entities were treated
    active_df = df[df['canonical_target'] != 'None'].copy()

    print("Aggregating framing and sentiment data...")
    
    # 3. Create the Aggregation Logic
    # We group by Event, Target, and Outlet to see exactly how an outlet treated a specific entity in a specific event
    grouped = active_df.groupby(['event_id', 'canonical_target', 'clean_outlet'])

    comparison_records = []

    for (event_id, target, outlet), group in grouped:
        total_mentions = len(group)
        
        # Calculate Sentiment Percentages
        positive_pct = (len(group[group['sentiment'] == 'Positive']) / total_mentions) * 100
        negative_pct = (len(group[group['sentiment'] == 'Negative']) / total_mentions) * 100
        neutral_pct = (len(group[group['sentiment'] == 'Neutral']) / total_mentions) * 100
        
        # Determine the Dominant Frame
        # What was the primary lens this outlet used to view this target?
        dominant_frame = group['framing_label'].mode().iloc[0] if not group.empty else "None"
        
        # Determine Dominant Sentiment
        # If negative > 50%, it's heavily negative. If positive > 50%, it's heavily positive. Otherwise mixed/neutral.
        dominant_sentiment = "Neutral/Mixed"
        if negative_pct >= 50: dominant_sentiment = "Negative"
        elif positive_pct >= 50: dominant_sentiment = "Positive"

        comparison_records.append({
            'event_id': event_id,
            'canonical_target': target,
            'clean_outlet': outlet,
            'passage_mentions': total_mentions,
            'dominant_sentiment': dominant_sentiment,
            'dominant_framing': dominant_frame,
            '%_Negative': round(negative_pct, 1),
            '%_Positive': round(positive_pct, 1),
            '%_Neutral': round(neutral_pct, 1)
        })

    comparison_df = pd.DataFrame(comparison_records)

    # 4. Create the "Omission Matrix" (Who omitted a major target?)
    # If a target is mentioned by at least one outlet in an event, did the other outlets mention them?
    print("Generating Target Omission Analysis...")
    
    # Pivot so columns are outlets, and rows are Event+Target
    # Values are the number of times the outlet mentioned the target
    omission_pivot = pd.pivot_table(
        comparison_df,
        values='passage_mentions',
        index=['event_id', 'canonical_target'],
        columns='clean_outlet',
        aggfunc='sum',
        fill_value=0
    ).reset_index()

    # Save the outputs
    comparison_df.to_csv(output_file, index=False)
    omission_pivot.to_csv("Data/target_omission_matrix.csv", index=False)

    print("\n=========================================================")
    print("            CROSS-OUTLET BIAS ANALYSIS COMPLETE          ")
    print("=========================================================")
    print(f"Generated comparison data for {len(comparison_df)} Event-Target-Outlet combinations.")
    print(f"\nOutputs generated:")
    print(f"1. '{output_file}'")
    print(f"   -> Shows exactly how Outlet A vs Outlet B framed the same person in the same event.")
    print(f"2. 'target_omission_matrix.csv'")
    print(f"   -> Shows if an outlet completely scrubbed/omitted a politician from a story that others mentioned.")
    print("=========================================================\n")

if __name__ == "__main__":
    generate_bias_comparisons()