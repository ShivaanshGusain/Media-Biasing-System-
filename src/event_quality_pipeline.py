import pandas as pd
import numpy as np
import re
from collections import Counter

class EventQualityPipeline:
    def __init__(self, input_csv):
        print(f"Loading data from {input_csv}...")
        self.df = pd.read_csv(input_csv)
        # Ensure text columns are strings to prevent NaN errors
        for col in ['url', 'headline', 'full_text', 'first_paragraph', 'lead']:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna("").astype(str)

    def step_1_article_validation(self):
        """Filters out navigation pages, liveblogs, and low-quality text."""
        print("Running Stage 1: Article Validation...")
        
        # 1. URL Rejection Rules
        bad_url_patterns = r'(/tag/|/category/|/liveblog/|/epaper/|/search/|/homepage|/video/|/gallery/)'
        self.df['is_bad_url'] = self.df['url'].str.contains(bad_url_patterns, flags=re.IGNORECASE, regex=True)

        # 2. Boilerplate Rejection Rules (Add your specific outlet boilerplates here)
        boilerplate_patterns = r'(Stay informed with our latest|TOI News Desk comprises|India News Latest Updates|Download the app|Click here to read more)'
        self.df['is_boilerplate'] = self.df['full_text'].str.contains(boilerplate_patterns, flags=re.IGNORECASE, regex=True)

        # 3. Content Length and Repetitiveness
        def check_text_quality(text):
            words = text.split()
            if len(words) < 60:  # Too short to be a full article
                return 'too_short'
            
            # Check for high repetition (e.g., SEO spam or broken scrape)
            vocab = set(words)
            if len(vocab) / len(words) < 0.25 and len(words) > 100:
                return 'too_repetitive'
            return 'pass'

        self.df['text_quality_flag'] = self.df['full_text'].apply(check_text_quality)

        # 4. Semantic Misalignment (Headline vs Body)
        # Fallback heuristic: Does the headline share at least 1 meaningful word with the first 2 paragraphs?
        def check_alignment(row):
            head_words = set([w for w in re.sub(r"[^a-z0-9\s]", "", row['headline'].lower()).split() if len(w) > 3])
            body_words = set(re.sub(r"[^a-z0-9\s]", "", row['full_text'][:1000].lower()).split())
            if head_words and not head_words.intersection(body_words):
                return True # Misaligned
            return False
            
        self.df['is_misaligned'] = self.df.apply(check_alignment, axis=1)

        # Apply Filters
        initial_len = len(self.df)
        self.df['is_valid_article'] = ~(
            self.df['is_bad_url'] | 
            self.df['is_boilerplate'] | 
            (self.df['text_quality_flag'] != 'pass') | 
            self.df['is_misaligned']
        )
        
        valid_df = self.df[self.df['is_valid_article']].copy()
        print(f" -> Removed {initial_len - len(valid_df)} invalid articles.")
        self.df = valid_df

    def step_2_wire_copy_detection(self):
        """Tags articles that are likely just wire agency copy-pastes."""
        print("Running Stage 2: Wire-Copy Detection...")
        
        # Look for standard wire agency tags in the lead or first paragraph
        wire_patterns = r'\b(PTI|ANI|Reuters|IANS|Bloomberg|AP)\b'
        
        def is_wire(row):
            search_text = row['first_paragraph'] + " " + row['lead']
            if re.search(wire_patterns, search_text):
                return True
            return False
            
        self.df['is_wire_copy'] = self.df.apply(is_wire, axis=1)
        print(f" -> Flagged {self.df['is_wire_copy'].sum()} articles as wire-copy.")

    def step_3_cluster_coherence(self):
        """Calculates a coherence score for existing clusters."""
        print("Running Stage 3: Cluster Coherence Scoring...")
        # Note: If you have embeddings here, you would calculate variance.
        # Without embeddings, we use the ratio of unique outlets to total articles 
        # and headline overlap as a proxy for coherence.
        
        cluster_scores = {}
        for event_id, group in self.df.groupby('event_id'):
            if len(group) == 1:
                cluster_scores[event_id] = 1.0 # Perfect coherence for singleton
                continue
                
            # Measure how similar the headlines are (Jaccard overlap proxy)
            all_head_words = group['headline'].str.lower().str.split().explode()
            common_words = all_head_words.value_counts().head(5).sum() / len(all_head_words)
            
            # Penalize massive clusters that span too many days
            date_span = pd.to_datetime(group['publish_date_only']).max() - pd.to_datetime(group['publish_date_only']).min()
            span_penalty = 1.0 if date_span.days <= 3 else 0.5
            
            cluster_scores[event_id] = round(min(common_words * span_penalty * 2, 1.0), 2)
            
        self.df['cluster_confidence'] = self.df['event_id'].map(cluster_scores)

    def step_4_generate_canonical_events(self, output_file="Data/canonical_events_master.csv"):
        """Rolls up the clean articles into the final Event schema for Bias Analysis."""
        print("Running Stage 4: Generating Canonical Event Records...")
        
        # We only want to summarize events based on original reporting (exclude wire copies for outlet count)
        original_reporting = self.df[~self.df['is_wire_copy']]
        
        events = []
        for event_id, group in self.df.groupby('event_id'):
            orig_group = original_reporting[original_reporting['event_id'] == event_id]
            
            # Determine representative headline (longest one is usually most descriptive)
            rep_headline = group.loc[group['headline'].str.len().idxmax(), 'headline']
            
            # Event Date (earliest publish date in cluster)
            event_date = group['publish_date_only'].min()
            
            # Determine if it's a low-quality event (e.g., only wire copies, or very low coherence)
            conf = group['cluster_confidence'].iloc[0]
            is_low_quality = bool(conf < 0.40 or len(orig_group) == 0)
            
            # Simple summary (first 300 chars of the longest article)
            longest_text = group.loc[group['full_text'].str.len().idxmax(), 'full_text']
            event_summary = longest_text[:300] + "..." if len(longest_text) > 300 else longest_text
            
            events.append({
                'event_id': event_id,
                'event_title': rep_headline, # Using rep_headline as title
                'event_date': event_date,
                'article_count': len(group),
                'unique_outlet_count': orig_group['clean_outlet'].nunique(), # Only counting original reports
                'outlets_covering': list(orig_group['clean_outlet'].unique()),
                'representative_headline': rep_headline,
                'event_summary': event_summary,
                'cluster_confidence': conf,
                'is_low_quality_event': is_low_quality
            })
            
        canonical_df = pd.DataFrame(events)
        
        # Sort by biggest, most trusted events first
        canonical_df = canonical_df.sort_values(
            by=['unique_outlet_count', 'cluster_confidence'], 
            ascending=[False, False]
        )
        
        canonical_df.to_csv(output_file, index=False)
        print(f"\nSUCCESS! Canonical Event list saved to {output_file}")
        print(canonical_df.head(3).to_string(columns=['event_id', 'event_title', 'unique_outlet_count', 'cluster_confidence']))

    def run_all(self):
        self.step_1_article_validation()
        self.step_2_wire_copy_detection()
        self.step_3_cluster_coherence()
        self.step_4_generate_canonical_events()

if __name__ == "__main__":
    # Assuming 'bias_article_details.csv' contains your currently clustered data
    pipeline = EventQualityPipeline("Data/bias_article_details.csv")
    pipeline.run_all()