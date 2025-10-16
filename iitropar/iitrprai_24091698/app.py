import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset

df = pd.read_csv("movie_dataset.csv")

# Features we will use
features = ['genres', 'keywords', 'title', 'cast', 'director']

# Fill NaN values with empty string
for feature in features:
    df[feature] = df[feature].fillna('')

# Combine features
def combine_features(row):
    return row['title'] + " " + row['genres'] + " " + row['director'] + " " + row['keywords'] + " " + row['cast']

df["combined_features"] = df.apply(combine_features, axis=1)

# Vectorize
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])

# Cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix)

# Helper functions
def get_title_from_index(index):
    return df.loc[index, "title"]

def get_index_from_title(title):
    result = df[df.title.str.lower() == title.lower()]
    if not result.empty:
        return result.index[0]
    else:
        return None

def recommend_movies(movie_title, num_recommendations=5):
    movie_index = get_index_from_title(movie_title)
    if movie_index is None:
        return []
    similar_movies = list(enumerate(cosine_sim[movie_index]))
    sorted_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    return [(get_title_from_index(i), score) for i, score in sorted_movies]

# Streamlit App

st.title("🎬 Movie Recommendation System")
st.write("Find movies similar to your favorite one using content-based filtering.")

# Input from user
movie_name = st.text_input("Enter a movie title:")


if st.button("Recommend"):
    if movie_name.strip() == "":
        st.warning("⚠️ Please enter a movie title.")
    else:
        movie_index = get_index_from_title(movie_name)
        if movie_index is not None:
            # Display movie details
            movie_row = df.loc[movie_index]
            st.subheader(f"Details for **{movie_row['title']}**:")
            st.markdown(f"**Genres:** {movie_row['genres']}")
            st.markdown(f"**Director:** {movie_row['director']}")
            st.markdown(f"**Cast:** {movie_row['cast']}")
            st.markdown(f"**Keywords:** {movie_row['keywords']}")
            st.markdown("---")
            # Show recommendations
            recommendations = recommend_movies(movie_name, num_recommendations=10)
            st.subheader(f"Movies similar to **{movie_name}**:")
            for title, score in recommendations:
                st.write(f"- {title} (similarity: {score:.2f})")
        else:
            st.error("❌ Movie not found in dataset. Please try another title.")
        
#streamlit run app.py