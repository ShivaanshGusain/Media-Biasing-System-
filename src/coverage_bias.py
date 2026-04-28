import pandas as pd
import numpy as np

def compute_coverage_bias():
    input_file = "Data/analysis_articles_master.csv"
    
    print("Loading master dataset...")
    df = pd.read_csv(input_file)
    
    # Clean boolean columns
    for col in ['failed_coherence_audit', 'is_template_junk']:
        df[col] = df[col].astype(str).str.lower().map({'true': True, 'false': False}).fillna(False)
        
    # Convert publish_time to datetime objects
    df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
    
    # Filter out absolute junk/templates, but KEEP singletons for exclusive-rate calculation
    valid_df = df[(~df['failed_coherence_audit']) & (~df['is_template_junk'])].copy()
    
    print("\n1. Building Event-Outlet Coverage Matrix...")
    # Binary Matrix: 1 if outlet covered event, 0 if missed
    coverage_matrix = pd.pivot_table(
        valid_df, 
        values='article_id', 
        index='event_id', 
        columns='clean_outlet', 
        aggfunc=lambda x: 1, 
        fill_value=0
    )
    
    print("2. Calculating Missed vs Covered Events...")
    # We only penalize an outlet for "missing" an event if it was a SHARED event (Tier A/B/C)
    # i.e., an event covered by at least 2 other outlets.
    shared_events = coverage_matrix[coverage_matrix.sum(axis=1) >= 2].index
    shared_matrix = coverage_matrix.loc[shared_events]
    
    total_shared_events = len(shared_events)
    
    stats = []
    outlets = coverage_matrix.columns
    
    print("3. Calculating Delay-to-Cover...")
    # Find the earliest publish time for every event (the "breaking" time)
    event_start_times = valid_df.groupby('event_id')['publish_time'].min()
    
    for outlet in outlets:
        # Exclusive events (events ONLY covered by this outlet)
        exclusive_events = coverage_matrix[(coverage_matrix[outlet] == 1) & (coverage_matrix.sum(axis=1) == 1)].shape[0]
        total_covered_by_outlet = coverage_matrix[outlet].sum()
        
        # Coverage of Shared Events
        covered_shared = shared_matrix[outlet].sum()
        missed_shared = total_shared_events - covered_shared
        coverage_share = (covered_shared / total_shared_events) * 100 if total_shared_events > 0 else 0
        exclusive_rate = (exclusive_events / total_covered_by_outlet) * 100 if total_covered_by_outlet > 0 else 0
        
        # Delay to cover (Average hours behind the breaking outlet)
        outlet_articles = valid_df[valid_df['clean_outlet'] == outlet]
        delays = []
        for _, row in outlet_articles.iterrows():
            if pd.notna(row['publish_time']):
                start_time = event_start_times[row['event_id']]
                delay_hours = (row['publish_time'] - start_time).total_seconds() / 3600.0
                delays.append(delay_hours)
                
        avg_delay = np.mean(delays) if delays else np.nan
        
        stats.append({
            'Outlet': outlet,
            'Total_Events_Covered': total_covered_by_outlet,
            'Exclusive_Events': exclusive_events,
            'Exclusive_Rate_%': round(exclusive_rate, 1),
            'Shared_Events_Covered': covered_shared,
            'Shared_Events_Missed': missed_shared,
            'Major_Coverage_Share_%': round(coverage_share, 1),
            'Avg_Delay_Hours': round(avg_delay, 2) if pd.notna(avg_delay) else "N/A"
        })

    outlet_stats_df = pd.DataFrame(stats).sort_values(by='Major_Coverage_Share_%', ascending=False)
    
    print("4. Calculating Pairwise Outlet Overlap...")
    # Matrix multiplication to get overlap (dot product of transposed matrix)
    overlap_matrix = coverage_matrix.T.dot(coverage_matrix)
    
    # Save Outputs
    coverage_matrix.to_csv("Data/matrix_event_coverage.csv")
    outlet_stats_df.to_csv("Data/bias_coverage_statistics.csv", index=False)
    overlap_matrix.to_csv("Data/matrix_pairwise_overlap.csv")
    
    print("\n=========================================================")
    print("                 COVERAGE BIAS RESULTS                   ")
    print("=========================================================")
    print(outlet_stats_df.to_string(index=False))
    print("=========================================================")
    print("\nSUCCESS! Outputs generated:")
    print(" -> 'matrix_event_coverage.csv' (The raw binary 1/0 matrix)")
    print(" -> 'bias_coverage_statistics.csv' (Missed, Share, Delay, Exclusive metrics)")
    print(" -> 'matrix_pairwise_overlap.csv' (How often outlets agree on what is news)")

if __name__ == "__main__":
    compute_coverage_bias()