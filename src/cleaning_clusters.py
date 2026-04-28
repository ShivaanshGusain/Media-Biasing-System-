import pandas as pd
import re
import spacy

# Load SpaCy for accurate Entity Extraction
try:
    print("Loading SpaCy NLP model...")
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except OSError:
    print("WARNING: SpaCy model not found. Using Regex fallback.")
    SPACY_AVAILABLE = False

def extract_entities(text):
    """Extracts true political entities (People, Orgs, Locations)"""
    if pd.isna(text) or not str(text).strip(): 
        return set()
    
    text = str(text)
    
    if SPACY_AVAILABLE:
        # Take the first 1000 chars to keep it fast
        doc = nlp(text[:1000])
        # Keep only People, Organizations, and Geo-Political Entities
        valid_labels = {'PERSON', 'ORG', 'GPE'}
        entities = {ent.text.lower().strip() for ent in doc.ents if ent.label_ in valid_labels}
        return entities
    else:
        # Regex Fallback
        entities = re.findall(r'\b[A-Z][a-z]+\b', text[1:])
        return set([e.lower() for e in entities])


def prepare_for_bias_analysis(clustered_csv="clustered_events_db.csv"):
    print("1. Loading clustered data...")
    try:
        df = pd.read_csv(clustered_csv)
    except FileNotFoundError:
        print(f"Error: {clustered_csv} not found.")
        return
        
    df['publish_date'] = pd.to_datetime(df['publish_date_only'], errors='coerce')
    
    # ==========================================
    # FIX 1: SAME-OUTLET DUPLICATES
    # ==========================================
    print("2. Fixing Same-Outlet Duplicates...")
    df = df.sort_values('publish_date')
    df = df.drop_duplicates(subset=['event_id', 'clean_outlet'], keep='first').copy()

    # ==========================================
    # FIX 2: DROP WIRE AGENCIES (News aggregators)
    # ==========================================
    print("3. Dropping Wire-Only Events...")
    wire_agencies = ['pti', 'ani', 'ians', 'reuters', 'unknown']
    
    # Find events where ALL articles are from wire agencies
    event_outlets = df.groupby('event_id')['clean_outlet'].apply(set)
    events_to_drop = []
    
    for evt, outlets in event_outlets.items():
        if all(out in wire_agencies for out in outlets):
            events_to_drop.append(evt)
            
    df = df[~df['event_id'].isin(events_to_drop)].copy()

    # ==========================================
    # FIX 3: SMART MERGE (Time-Bounded Entity Overlap)
    # ==========================================
    print("4. Extracting Entities for Event Merging...")
    
    # Combine all text for an event to get its overall entities and average date
    event_data = {}
    for evt, group in df.groupby('event_id'):
        combined_text = " ".join(group['headline'].fillna("") + " " + group['cluster_text'].fillna(""))
        avg_date = group['publish_date'].mean()
        event_data[evt] = {
            'entities': extract_entities(combined_text),
            'date': avg_date
        }

    print("5. Checking for fragmented clusters (Time-Bounded)...")
    merge_map = {}
    event_ids = list(event_data.keys())
    
    # $O(N \times k)$ loop: Only compare events within 3 days of each other
    for i in range(len(event_ids)):
        evt1 = event_ids[i]
        ent1 = event_data[evt1]['entities']
        date1 = event_data[evt1]['date']
        
        if len(ent1) < 3: continue # Skip events with too little context
            
        for j in range(i + 1, len(event_ids)):
            evt2 = event_ids[j]
            date2 = event_data[evt2]['date']
            
            # THE CRITICAL FIX: Skip if events are more than 3 days apart
            if pd.notna(date1) and pd.notna(date2):
                if abs((date1 - date2).days) > 3:
                    continue
                    
            ent2 = event_data[evt2]['entities']
            if len(ent2) < 3: continue
                
            # Calculate Overlap
            overlap = len(ent1 & ent2)
            smaller_set_size = min(len(ent1), len(ent2))
            overlap_score = overlap / smaller_set_size if smaller_set_size > 0 else 0
            
            if overlap_score >= 0.75:
                # Map the newer event to the older event ID
                merge_map[evt2] = evt1

    # Apply the merges
    if merge_map:
        print(f"   -> Merged {len(merge_map)} fragmented clusters back together.")
        df['event_id'] = df['event_id'].replace(merge_map)
    else:
        print("   -> No fragmented clusters found.")

    # ==========================================
    # 5. GENERATE FINAL SUMMARIES
    # ==========================================
    print("6. Generating Bias Analysis outputs...")
    event_summary = df.groupby('event_id').agg(
        total_articles_published=('article_id', 'count'),
        unique_outlets_covering=('clean_outlet', 'nunique'),
        list_of_outlets=('clean_outlet', lambda x: list(set(x))),
        main_headline=('headline', 'first')
    ).reset_index().sort_values(by='unique_outlets_covering', ascending=False)

    event_summary.to_csv("bias_event_summary.csv", index=False)
    df.to_csv("bias_article_details.csv", index=False)
    
    print("\n🎉 SUCCESS! Data is cleaned and ready for Bias Analysis.")
    print("Output 1: 'bias_event_summary.csv' (Use this for your dashboard events list)")
    print("Output 2: 'bias_article_details.csv' (Use this for your next passage-segmentation step)")

if __name__ == "__main__":
    prepare_for_bias_analysis()