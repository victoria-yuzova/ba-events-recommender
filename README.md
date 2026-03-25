# BA Events Recommender

An end-to-end ML pipeline that scrapes cultural events in Buenos Aires 
and recommends what to do this weekend based on your preferences.

## 🎯 Problem

Buenos Aires has a rich cultural scene spread across dozens of 
institutions with no unified discovery layer. This project builds 
a personal recommender that aggregates events from multiple sources 
and ranks them by similarity to your taste.

## 🏗 Pipeline

1. **Scraping** — extracts links from 4 cultural institution homepages
2. **LLM classification** — uses GPT-4.1-mini to extract structured 
   event data (title, category, venue, price, tags, summary)
3. **Cleaning** — filters to high-confidence event_detail rows, 
   parses features, merges user ratings
4. **Recommender** — TF-IDF vectorization on summaries + tags, 
   cosine similarity ranking against liked events
5. **App** — Streamlit interface with category filter, 
   top 10 recommendations with links

## 📊 Current Dataset

- 4 sources: Complejo Teatral, MALBA, Bellas Artes, Turismo BA
- ~103 events scraped per run
- 36 clean events after filtering
- 46 manually labeled events (ground truth)

## 🛠 Tech Stack

- Python, BeautifulSoup, Requests
- OpenAI API (GPT-4.1-mini) for event extraction
- pandas, scikit-learn (TF-IDF, cosine similarity, logistic regression)
- matplotlib, seaborn
- Streamlit

## 📁 Structure
```
notebooks/
  01_scraper.ipynb             # data collection pipeline
  02_cleaning_and_features     # cleaning, feature engineering
  03_recommender.ipynb         # model and recommendations
src/
  scraper.py                   # scraping functions
  homepage_urls.json           # source URLs
data/
  raw/                         # scraped events by run date
  processed/                   # clean dataset, user ratings
app.py                         # Streamlit demo app
```

## ⚠️ Known Limitations

- 3 JS-rendered sites (Teatro Cervantes, Palacio Libertad, 
  CC Recoleta) excluded — Playwright integration planned
- Dataset is small (36 events) — cosine similarity works but 
  supervised model evaluation is not meaningful at this size
- Recommendations currently based on one user's ratings — 
  personalization flow planned

## 🚀 Next Steps

- User rating flow in app for personalized recommendations
- Playwright integration for JS-rendered sites
- Scheduled pipeline runs to accumulate data over time