import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from typing import Tuple
import os


@st.cache_data
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load dataset. Prefers movie_ratings.csv, falls back to movie_ratings_EC.csv.
    Returns original dataframe and exploded (by genre).
    """
    if os.path.exists("movie_ratings.csv"):
        csv_path = "movie_ratings.csv"
    elif os.path.exists("movie_ratings_EC.csv"):
        csv_path = "movie_ratings_EC.csv"
    else:
        raise FileNotFoundError(
            "No dataset file found. Expected 'movie_ratings.csv' or 'movie_ratings_EC.csv' in the current directory."
        )

    df = pd.read_csv(csv_path)

    # Ensure genres column is properly exploded
    if "genres" in df.columns:
        df_exp = df.assign(genres=df["genres"].str.split("|")).explode("genres")
    elif "genre" in df.columns:  # in case pre-cleaned version has single genre
        df_exp = df.rename(columns={"genre": "genres"})
    else:
        raise ValueError("No genre column found in dataset.")

    return df, df_exp


def main():
    st.title("🎬 MovieLens Ratings Dashboard")
    st.markdown("Analyze Movie ratings dataset interactively.")

    # Load dataset
    df, df_exp = load_data()

    # Sidebar filters
    st.sidebar.header("Filters")
    age_range = st.sidebar.slider("Age Range", int(df["age"].min()), int(df["age"].max()), (18, 50))
    genders = st.sidebar.multiselect("Gender", df["gender"].unique(), default=list(df["gender"].unique()))
    occupations = st.sidebar.multiselect("Occupation", df["occupation"].unique(), default=list(df["occupation"].unique()))
    selected_genres = st.sidebar.multiselect("Genres", df_exp["genres"].unique(), default=list(df_exp["genres"].unique()))

    # Filtered dataset
    filtered = df_exp[
        (df_exp["age"].between(age_range[0], age_range[1]))
        & (df_exp["gender"].isin(genders))
        & (df_exp["occupation"].isin(occupations))
        & (df_exp["genres"].isin(selected_genres))
    ]

    st.subheader("1. Breakdown of Genres")
    genre_counts = filtered["genres"].value_counts().reset_index()
    genre_counts.columns = ["Genre", "Count"]
    fig1 = px.bar(genre_counts, x="Genre", y="Count", title="Genre Breakdown (Filtered)", color="Genre")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("2. Genres with Highest Viewer Satisfaction")
    genre_ratings = filtered.groupby("genres")["rating"].mean().reset_index()
    fig2 = px.bar(genre_ratings.sort_values("rating", ascending=False), x="genres", y="rating",
                  title="Average Rating by Genre", color="rating", color_continuous_scale="Blues")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("3. Mean Rating by Movie Release Year")
    ratings_by_year = filtered.groupby("year")["rating"].mean().reset_index()
    fig3 = px.line(ratings_by_year, x="year", y="rating", title="Mean Rating Across Release Years")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("4. Top Rated Movies")
    min_ratings = st.sidebar.slider("Minimum Ratings Threshold", 50, 500, 50, step=50)

    movie_stats = filtered.groupby("title").agg(
        mean_rating=("rating", "mean"),
        count=("rating", "count")
    ).reset_index()

    top_movies = movie_stats[movie_stats["count"] >= min_ratings].sort_values(
        by="mean_rating", ascending=False
    ).head(5)

    st.write(f"Top 5 Movies with at least {min_ratings} ratings:")
    st.table(top_movies)


if __name__ == "__main__":
    main()
