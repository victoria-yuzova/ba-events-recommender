
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, List, Callable
from urllib.parse import urljoin, urlparse
import hashlib
import os
import pandas as pd
from bs4 import BeautifulSoup
import requests
import json



# Standard headers to fetch a website
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

# Avoid hanging forever on slow/unresponsive sites.
# (connect timeout, read timeout) in seconds
REQUEST_TIMEOUT = (10, 30)


def fetch_website_contents(url):
    """
    Return the title and contents of the website at the given url;
    truncate to 2,000 characters as a sensible limit
    """
    try:
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException:
        return "No title found\n\n"
    soup = BeautifulSoup(response.content, "html.parser")
    title = soup.title.string if soup.title else "No title found"
    if soup.body:
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        text = soup.body.get_text(separator="\n", strip=True)
    else:
        text = ""
    return (title + "\n\n" + text)[:2_000]


def fetch_website_links(url):
    """
    Return the links on the webiste at the given url
    I realize this is inefficient as we're parsing twice! This is to keep the code in the lab simple.
    Feel free to use a class and optimize it!
    """
    try:
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException:
        return []
    soup = BeautifulSoup(response.content, "html.parser")
    links = [link.get("href") for link in soup.find_all("a")]
    return [link for link in links if link]


import pandas as pd
from urllib.parse import urljoin, urlparse

JUNK_PREFIXES = ("mailto:", "tel:", "javascript:")
JUNK_SUBSTRINGS = (
    "cdn-cgi/l/email-protection",
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "privacy",
    "terminos",
    "terms",
    "cookies",
    "contacto",
    "prensa",
)



# Keep these near the function (or import from a config module)
JUNK_PREFIXES = ("#", "mailto:", "tel:", "javascript:")
JUNK_SUBSTRINGS = (
    "cdn-cgi/l/email-protection",
    # optional extra junk patterns (safe defaults)
    "/privacy", "privacy", "/terminos", "terminos", "/terms", "terms", "cookies",
)

def extract_links_from_homepages(
    homepage_urls: List[str],
    fetch_website_links_fn: Callable[[str], list],
    *,
    keep_same_domain: bool = True,
    out_dir: str = "../data/raw",
    filename: str = "01_all_links.csv",
    run_ts: Optional[str] = None,
) -> pd.DataFrame:
    """
    Step 1: Extract and clean ALL candidate links from homepage URLs.

    Saves to a date-partitioned path:
        {out_dir}/{run_date}/{filename}

    Returns a DataFrame with:
        run_date, scraped_at, page_url, event_url_raw, event_url_abs, link_id
    """
    # Timestamp + run_date
    if run_ts is None:
        run_ts = datetime.utcnow().isoformat(timespec="seconds")
    run_date = run_ts[:10]  # YYYY-MM-DD

    rows = []

    for page_url in homepage_urls:
        hrefs = fetch_website_links_fn(page_url) or []
        homepage_domain = urlparse(page_url).netloc

        for href in hrefs:
            if not href:
                continue

            href_str = str(href).strip()

            # Drop junk prefixes (anchors, mailto, etc.)
            if href_str.startswith(JUNK_PREFIXES):
                continue

            # Drop obvious junk substrings
            if any(j in href_str.lower() for j in JUNK_SUBSTRINGS):
                continue

            # Convert to absolute URL
            abs_url = urljoin(page_url, href_str)

            # Optional: keep only same-domain links
            if keep_same_domain and urlparse(abs_url).netloc != homepage_domain:
                continue

            # Drop "homepage links to itself"
            if abs_url.rstrip("/") == page_url.rstrip("/"):
                continue

            link_id = hashlib.md5(abs_url.encode("utf-8")).hexdigest()

            rows.append(
                {
                    "run_date": run_date,
                    "scraped_at": run_ts,
                    "page_url": page_url,
                    "event_url_raw": href_str,
                    "event_url_abs": abs_url,
                    "link_id": link_id,
                }
            )

    df = pd.DataFrame(rows)

    # Always write a file for the run (even if empty) — helps debugging/history
    save_folder = os.path.join(out_dir, run_date)
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, filename)

    df = df.drop_duplicates(subset=["page_url", "event_url_abs"]).reset_index(drop=True)
    df.to_csv(save_path, index=False)

    print(f"Saved {len(df)} rows to: {save_path}")
    return df




event_system_prompt = """
You extract structured event info from a Buenos Aires cultural events webpage.

Input: JSON with:
- url: string
- homepage_url: string
- content: string (title + cleaned text, may be truncated)

Return ONLY valid JSON with this schema:
{
  "url": "<string>",
  "homepage_url": "<string>",
  "page_type": "event_detail" | "listing" | "ticket" | "pdf" | "other",
  "title": "<string or null>",
  "summary": "<string or null>",
  "category": "theatre" | "music" | "exhibition" | "cinema" | "dance" | "talk" | "workshop" | "other",
  "start_date": "<YYYY-MM-DD or null>",
  "start_time": "<HH:MM or null>",
  "venue": "<string or null>",
  "price": "<string or null>",
  "is_free": true | false | null,
  "tags": ["<string>", "..."],
  "confidence": <number between 0 and 1>
}

Rules:
- If content is too thin/unclear, set fields to null and lower confidence.
- Do not invent specifics (date/time/price) if not present.
""".strip()

def classify_one_url(openai_client, model, url, homepage_url, content):
    payload = {"url": url, "homepage_url": homepage_url, "content": content}
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": event_system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content)

def build_classified_events_dataset(
    df_relevant: pd.DataFrame,
    fetch_website_contents_fn,
    *,
    openai_client,
    model: str = "gpt-4.1-mini",
    out_dir: str = "../data/raw",
    filename: str = "03_events.csv",
    limit: int = 50,  # start small
):
    run_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_date = run_ts[:10]

    rows = []
    df = df_relevant.copy()

    # expected columns: url, homepage_url
    df = df.drop_duplicates(subset=["homepage_url", "url"]).reset_index(drop=True)

    for i, r in df.head(limit).iterrows():
        url = r["url"]
        homepage_url = r["homepage_url"]

        # Fast routing: PDFs don’t need fetch_website_contents (optional)
        if str(url).lower().endswith(".pdf"):
            rows.append({
                "url": url,
                "homepage_url": homepage_url,
                "page_type": "pdf",
                "title": None,
                "summary": None,
                "category": "other",
                "start_date": None,
                "start_time": None,
                "venue": None,
                "price": None,
                "is_free": None,
                "tags": [],
                "confidence": 0.3,
                "run_date": run_date,
                "extracted_at": run_ts,
            })
            continue

        content = fetch_website_contents_fn(url)
        out = classify_one_url(openai_client, model, url, homepage_url, content)

        out["run_date"] = run_date
        out["extracted_at"] = run_ts
        rows.append(out)

    df_events = pd.DataFrame(rows)

    save_folder = os.path.join(out_dir, run_date)
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, filename)
    df_events.to_csv(save_path, index=False)
    print(f"Saved {len(df_events)} rows to: {save_path}")

    return df_events