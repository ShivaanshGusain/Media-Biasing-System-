import pandas as pd
import re

class CanonicalEventAudit:
    def __init__(self, input_csv="Data/canonical_events_master.csv"):
        print(f"Loading canonical events from {input_csv}...")
        self.df = pd.read_csv(input_csv)
        
        # Ensure text columns are strings
        for col in ['event_title', 'representative_headline', 'event_summary']:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna("").astype(str)

    def _extract_keywords(self, text):
        # Remove punctuation and lowercase
        clean_text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        return set([w for w in clean_text.split() if len(w) > 3])

    def check_event_coherence(self):
        print("1. Auditing Event Coherence (Headline vs Summary)...")
        
        def is_incoherent(row):
            head_words = self._extract_keywords(row['representative_headline'])
            sum_words = self._extract_keywords(row['event_summary'])
            
            if not head_words or not sum_words:
                return True # Empty text is incoherent
                
            # If there is absolutely zero semantic overlap between the headline 
            # and the first 300 words of the text, it's a hallucinated cluster.
            overlap = head_words.intersection(sum_words)
            if len(overlap) == 0:
                return True
                
            return False

        self.df['failed_coherence_audit'] = self.df.apply(is_incoherent, axis=1)

    def detect_template_junk(self):
        print("2. Sweeping for Page-Template Junk...")
        
        # Aggressive regex for non-news structural pages
        junk_patterns = r'(?i)(' \
                        r'search results for|' \
                        r'category archive|' \
                        r'latest news updates|' \
                        r'home page|' \
                        r'subscribe now|' \
                        r'all rights reserved|' \
                        r'click here to read|' \
                        r'live updates:|' \
                        r'epaper|' \
                        r'top stories of the day|' \
                        r'sign in to read' \
                        r')'
        
        def is_junk_template(row):
            combined_text = row['representative_headline'] + " " + row['event_summary']
            if re.search(junk_patterns, combined_text):
                return True
            return False
            
        self.df['is_template_junk'] = self.df.apply(is_junk_template, axis=1)

    def recompute_strict_quality(self):
        """
        Calculates the final 'is_low_quality_event' flag using strict, multi-layered rules.
        """
        print("3. Recomputing Strict Event Quality...")
        
        def strict_evaluation(row):
            # 1. Did it fail the new coherence check?
            if row.get('failed_coherence_audit', False):
                return True
                
            # 2. Did it get flagged as a junk template?
            if row.get('is_template_junk', False):
                return True
                
            # 3. Was the original cluster confidence too low? (Raising the bar to 0.45)
            if row.get('cluster_confidence', 1.0) < 0.45:
                return True
                
            # 4. Is it an isolated event covered by only 1 outlet?
            # (For bias analysis, single-outlet events are often noise, op-eds, or ultra-local news)
            if row.get('unique_outlet_count', 1) < 2:
                return True
                
            return False

        self.df['is_low_quality_event'] = self.df.apply(strict_evaluation, axis=1)
        
        bad_count = self.df['is_low_quality_event'].sum()
        good_count = len(self.df) - bad_count
        print(f" -> Audit Complete: {good_count} Verified Events, {bad_count} Low-Quality/Junk Events.")

    def run_audit(self, output_file="Data/audited_canonical_events.csv"):
        self.check_event_coherence()
        self.detect_template_junk()
        self.recompute_strict_quality()
        
        # Save the audited file
        self.df.to_csv(output_file, index=False)
        print(f"\nSaved final audited dataset to -> {output_file}")
        
        # Optional: Print a sample of what got rejected for debugging
        rejected = self.df[self.df['is_low_quality_event'] == True]
        if not rejected.empty:
            print("\nSample of Rejected Events:")
            print(rejected[['representative_headline', 'failed_coherence_audit', 'is_template_junk']].head(5).to_string())

if __name__ == "__main__":
    audit = CanonicalEventAudit()
    audit.run_audit()