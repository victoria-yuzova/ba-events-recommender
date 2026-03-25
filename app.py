import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🎭 BA Events Recommender")
st.write("Top cultural events in Buenos Aires based on your taste.")
st.sidebar.header("Filters")



# Load data
df = pd.read_csv("data/processed/events_clean.csv")
df['tags'] = df['tags'].apply(ast.literal_eval)
df['text'] = df['summary'] + ' ' + df['tags'].apply(lambda x: ' '.join(x))

# Add this after the title for a cleaner header

category = st.sidebar.multiselect(
    "Filter by category",
    options=df['category'].unique(),
    default=df['category'].unique()
)

# Build model
tfidf = TfidfVectorizer(max_features=100)
tfidf_matrix = tfidf.fit_transform(df['text'])
similarity_matrix = cosine_similarity(tfidf_matrix)

liked_indices = df[df['liked'] == 1].index.tolist()
scores = similarity_matrix[liked_indices].mean(axis=0)
df['rec_score'] = scores

# Show recommendations
st.subheader("🎯 Recommended for you")
top = (df[df['liked'] == 0]
       .sort_values('rec_score', ascending=False)
       .head(10))
top = top[top['category'].isin(category)]

for _, row in top.iterrows():
    st.markdown(f"### {row['title']}")
    st.write(f"**{row['category']}** | Free: {row['is_free']} | Score: {row['rec_score']:.3f}")
    st.write(row['summary'])
    st.markdown(f"[View event]({row['url']})")
    st.divider()