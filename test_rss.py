import requests
import feedparser

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
}

rss_urls = {
    "Indian Express": [
        "https://indianexpress.com/section/political-pulse/feed/",
        "https://indianexpress.com/section/india/feed/",
        "https://indianexpress.com/feed/",
    ],
    "News18": [
        "https://www.news18.com/rss/politics.xml",
        "https://www.news18.com/rss/india.xml",
        "https://www.news18.com/commonfeeds/v1/eng/rss/politics.xml",
        "https://www.news18.com/commonfeeds/v1/eng/rss/india.xml",
    ],
    "Firstpost": [
        "https://www.firstpost.com/rss/india.xml",
        "https://www.firstpost.com/feed",
        "https://www.firstpost.com/rss/politics.xml",
    ],
    "NDTV": [
        "https://feeds.feedburner.com/ndtvnews-india-news",
        "https://feeds.feedburner.com/ndtvnews-top-stories",
    ],
    "New Indian Express": [
        "https://www.newindianexpress.com/rss",
        "https://www.newindianexpress.com/nation/rssfeed.xml",
        "https://www.newindianexpress.com/feed",
    ],
    "Telegraph India": [
        "https://www.telegraphindia.com/rss",
        "https://www.telegraphindia.com/india/rss",
    ],
    "Deccan Herald": [
        "https://www.deccanherald.com/rss",
        "https://www.deccanherald.com/india/rss",
        "https://www.deccanherald.com/feeds/rss/national/india-politics.rss",
    ],
    "India Today": [
        "https://www.indiatoday.in/rss/home",
        "https://www.indiatoday.in/rss/1206578",
    ],
    "ThePrint": [
        "https://theprint.in/feed/",
        "https://theprint.in/politics/feed/",
        "https://theprint.in/india/feed/",
    ],
    "Times of India": [
        "https://timesofindia.indiatimes.com/rssfeeds/296589292.cms",
        "https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms",
    ],
    "Hindustan Times": [
        "https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml",
    ],
    "The Hindu": [
        "https://www.thehindu.com/news/national/feeder/default.rss",
    ],
}

for outlet, urls in rss_urls.items():
    print(f"\n{'='*60}")
    print(f"  {outlet}")
    print(f"{'='*60}")
    for url in urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            status = r.status_code
            if status == 200:
                feed = feedparser.parse(r.text)
                n_entries = len(feed.entries)
                sample = feed.entries[0].link if feed.entries else "N/A"
                print(f"  OK  {status} | {n_entries:3d} entries | {url}")
                if n_entries > 0:
                    print(f"       Sample: {sample}")
            else:
                print(f"  ERR {status} |           | {url}")
        except Exception as e:
            print(f"  EXC {str(e)[:50]:50s} | {url}")
