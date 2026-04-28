import json
import requests
import feedparser
from bs4 import BeautifulSoup
import trafilatura
from newspaper import Article
from readability import Document
import dateparser
from datetime import datetime,timedelta
import pandas as pd
import time
import random
import hashlib
import os
import re
HEADERS = {'User-Agent': 'Mozilla/5.0'}

def Date(raw_html):
    """A brute-force fallback to find hidden dates in HTML source."""
    if not raw_html:
        return None
        
    soup = BeautifulSoup(raw_html, 'html.parser')
    
    # Target 1: The standard OpenGraph / Article tags
    meta_tags = [
        soup.find('meta', property='article:published_time'),
        soup.find('meta', attrs={'itemprop': 'datePublished'}),
        soup.find('meta', attrs={'name': 'pubdate'}),
        soup.find('meta', attrs={'name': 'publish-date'})
    ]
    
    for tag in meta_tags:
        if tag and tag.get('content'):
            return tag['content']
            
    # Target 2: Look inside the JSON-LD schema (Very common on IE and The Hindu)
    for script in soup.find_all('script', type='application/ld+json'):
        if script.string and 'datePublished' in script.string:
            try:
                # Use regex to quickly yank the date out of the JSON string
                match = re.search(r'"datePublished"\s*:\s*"([^"]+)"', script.string)
                if match:
                    return match.group(1)
            except Exception:
                pass
                
    return None

def matches_patterns(url, config):
    patterns = config.get('url_patterns') or []
    single = config.get('url_pattern')
    if patterns:
        return any(p in url for p in patterns)
    if single:
        return single in url
    return True


def discover_links(config):
    discovered_urls = set()

    for rss_url in config.get('rss_feeds', []):
        try:
            feed = feedparser.parse(rss_url)
            for entry in feed.entries:
                link = getattr(entry, 'link', None)
                if link and matches_patterns(link, config):
                    discovered_urls.add(link)
        except Exception as e:
            print(f'Error parsing RSS {rss_url}: {e}')

    for sitemap_url in config.get('sitemaps', []):
        try:
            response = requests.get(sitemap_url, headers=HEADERS, timeout=20)
            soup = BeautifulSoup(response.text, 'xml')
            for loc in soup.find_all('loc'):
                href = loc.get_text(strip=True)
                if href and matches_patterns(href, config):
                    discovered_urls.add(href)
        except Exception as e:
            print(f'Error scraping sitemap {sitemap_url}: {e}')

    for section_url in config.get('section_pages', []):
        try:
            response = requests.get(section_url, headers=HEADERS, timeout=20)
            soup = BeautifulSoup(response.text, 'html.parser')
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                if href.startswith('/'):
                    base = section_url.split('/', 3)[:3]
                    href = '/'.join(base) + href
                if matches_patterns(href, config):
                    discovered_urls.add(href)
        except Exception as e:
            print(f'Error scraping section page {section_url}: {e}')

    return sorted(discovered_urls)


def extract_article_text(url):
    raw_html = None
    extracted_title, extracted_text, extracted_date, extractor = None, None, None, 'Failed'

    # Try Trafilatura
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            raw_html = downloaded
            extracted = trafilatura.extract(downloaded, include_comments=False, output_format='json')
            if extracted:
                data = json.loads(extracted)
                extracted_title = data.get('title')
                extracted_text = data.get('text')
                extracted_date = data.get('date')
                extractor = 'Trafilatura'
    except Exception:
        pass

    # If Trafilatura failed to get the text, try Newspaper3k
    if not extracted_text:
        try:
            article = Article(url)
            if raw_html:
                article.set_html(raw_html)
                article.parse()
            else:
                article.download()
                article.parse()
                raw_html = article.html # Save html for our fallback
                
            if article.text:
                extracted_title = article.title
                extracted_text = article.text
                extracted_date = str(article.publish_date) if article.publish_date else None
                extractor = 'Newspaper3k'
        except Exception:
            pass

    # --- THE DATE FALLBACK ---
    # If we have text, but no date, deploy the brute-force finder
    if extracted_text and not extracted_date and raw_html:
        found_date = Date(raw_html)
        if found_date:
            extracted_date = found_date
            
    return extracted_title, extracted_text, extracted_date, extractor


def normalize_article(outlet_name, url, title, text, raw_date, extractor_used):
    if not text or len(text) < 100:
        return None

    clean_date = None
    if raw_date:
        parsed_date = dateparser.parse(str(raw_date))
        if parsed_date:
            clean_date = parsed_date.strftime('%Y-%m-%d %H:%M:%S')

    paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 30]
    first_paragraph = paragraphs[0] if paragraphs else ''
    lead = first_paragraph[:250] + '...' if len(first_paragraph) > 250 else first_paragraph

    # --- NEW: Generate a unique, deterministic ID based on the URL ---
    # We take the first 16 characters of a SHA-256 hash to keep it clean but unique
    article_id = hashlib.sha256(url.encode('utf-8')).hexdigest()[:16]

    return {
        'article_id': article_id,
        'outlet': outlet_name,
        'url': url,
        'headline': title.strip() if title else 'Unknown',
        'publish_time': clean_date,
        'lead': lead,
        'first_paragraph': first_paragraph,
        'full_text': text,
        'extractor_used': extractor_used,
        'scrape_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def run_pipeline():
    with open('config.json', 'r') as f:
        configs = json.load(f)

    seen_urls = set()
    csv_filename = 'Data/canonical_articles_db.csv'
    
    week_def = datetime.now() - timedelta(days=7)
    if os.path.exists(csv_filename):
        try:
            existing_df = pd.read_csv(csv_filename)
            if 'url' in existing_df.columns:
                seen_urls = set(existing_df['url'].tolist())
                print(f"Loaded {len(seen_urls)} previously scraped URLs.")
                
        except Exception as e:
            print(f"Could not read existing CSV: {e}")

    for site_key, site_config in configs.items():
        print(f"\n--- Processing {site_config['outlet_name']} ---")
        urls = discover_links(site_config)
        
        print(f"Found {len(urls)} total URLs.")

        for url in urls: 
            url = url.replace('%e2%81%a0', '').replace('\u2060', '')
            if url in seen_urls:
                continue

            print(f"Scraping: {url}", end=" ") # Keeps the print on one line
            title, text, raw_date, extractor = extract_article_text(url)
                        
            canonical_record = normalize_article(
                site_config['outlet_name'], url, title, text, raw_date, extractor
            )
            
            if canonical_record:
                pub_time_str = canonical_record.get('publish_time')
                if pub_time_str:
                    pub_time = datetime.strptime(pub_time_str, '%Y-%m-%d %H:%M:%S')
                    
                    if pub_time < week_def:
                        print(f"--> Skipped (Too Old: Published {pub_time.strftime('%b %d, %Y')})")
                        seen_urls.add(url) # Add to seen so we ignore it next week
                        continue
                # --- LIVE SAVE: Write to CSV immediately ---
                df = pd.DataFrame([canonical_record])
                file_exists = os.path.isfile(csv_filename)
                
                try:
                    df.to_csv(csv_filename, mode='a', index=False, header=not file_exists)
                    print("--> Saved!") # Confirms it hit the CSV
                    seen_urls.add(url)
                except PermissionError:
                    print(f"\nERROR: Could not save {url}. Is {csv_filename} open?")
            else:
                print("--> Failed (No Text/Skipped)")
            
            time.sleep(random.uniform(1, 3)) 

    print("\nPipeline finished! All new articles saved to database.")
if __name__ == '__main__':
    run_pipeline()