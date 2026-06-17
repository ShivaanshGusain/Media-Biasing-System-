import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_visualizations():
    # Set global plotting style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    print("Generating Graph 1: Coverage Bias...")
    try:
        cov_df = pd.read_csv("Data/bias_coverage_statistics.csv")
        cov_df = cov_df.sort_values('Major_Coverage_Share_%', ascending=True)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Major_Coverage_Share_%', y='Outlet', data=cov_df, palette='viridis')
        plt.title('Agenda Setting: Major Coverage Share by Outlet', fontsize=16, fontweight='bold')
        plt.xlabel('Percentage of Major Events Covered (%)')
        plt.ylabel('News Outlet')
        plt.tight_layout()
        plt.savefig("Data/graph_1_coverage_bias.png", dpi=300)
        plt.close()
        print(" -> Saved 'graph_1_coverage_bias.png'")
    except Exception as e:
        print(f"Skipped Coverage Graph: {e}")

    print("Generating Graph 2: Framing Bias (Sentiment Stack)...")
    try:
        bias_df = pd.read_csv("Data/event_target_bias_comparison.csv")
        
        # Pick the most frequently mentioned target to visualize
        top_target = bias_df['canonical_target'].value_counts().index[0]
        target_df = bias_df[bias_df['canonical_target'] == top_target]
        
        # Aggregate sentiment across all events for this target
        agg_target = target_df.groupby('clean_outlet')[['%_Positive', '%_Neutral', '%_Negative']].mean().reset_index()
        
        # Normalize to ensure they sum to exactly 100%
        agg_target['Total'] = agg_target['%_Positive'] + agg_target['%_Neutral'] + agg_target['%_Negative']
        for col in ['%_Positive', '%_Neutral', '%_Negative']:
            agg_target[col] = (agg_target[col] / agg_target['Total']) * 100
            
        agg_target = agg_target.sort_values('%_Negative', ascending=True)

        # Plot 100% Stacked Bar
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plotting sequentially to stack them
        ax.barh(agg_target['clean_outlet'], agg_target['%_Positive'], color='#2ecc71', label='Positive')
        ax.barh(agg_target['clean_outlet'], agg_target['%_Neutral'], left=agg_target['%_Positive'], color='#95a5a6', label='Neutral')
        ax.barh(agg_target['clean_outlet'], agg_target['%_Negative'], left=agg_target['%_Positive'] + agg_target['%_Neutral'], color='#e74c3c', label='Negative')

        plt.title(f'Framing Bias: Sentiment Analysis toward "{top_target}"', fontsize=16, fontweight='bold')
        plt.xlabel('Percentage of Passages (%)')
        plt.ylabel('News Outlet')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.tight_layout()
        plt.savefig("Data/graph_2_framing_bias.png", dpi=300)
        plt.close()
        print(f" -> Saved 'graph_2_framing_bias.png' (Showing data for {top_target})")
    except Exception as e:
        print(f"Skipped Framing Graph: {e}")

    print("Generating Graph 3: Omission Matrix Heatmap...")
    try:
        omission_df = pd.read_csv("Data/target_omission_matrix.csv")
        
        # FIX: Aggregate mentions across ALL events for each target
        agg_df = omission_df.drop(columns=['event_id']).groupby('canonical_target').sum()
        
        # Calculate total mentions to find the top 15 targets overall
        agg_df['Total_Mentions'] = agg_df.sum(axis=1)
        top_omissions = agg_df.sort_values('Total_Mentions', ascending=False).head(20)
        
        # Drop the Total_Mentions column for the final heatmap
        heatmap_data = top_omissions.drop(columns=['Total_Mentions'])
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, cmap="Reds", annot=True, fmt="d", linewidths=.5, cbar_kws={'label': 'Number of Passages'})
        plt.title('Omission Bias: Mentions of Key Targets by Outlet (Aggregated)', fontsize=16, fontweight='bold')
        plt.xlabel('News Outlet')
        plt.ylabel('Political Target')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig("graph_3_omission_heatmap.png", dpi=300)
        plt.close()
        print(" -> Saved 'graph_3_omission_heatmap.png'")
    except Exception as e:
        print(f"Skipped Omission Graph: {e}")

if __name__ == "__main__":
    generate_visualizations()