"""
MovieLens 200k Ratings Dashboard
--------------------------------

This Streamlit application provides an interactive dashboard to explore the MovieLens
200k ratings dataset.  Users can filter the data by demographic attributes and
genres, then view visualizations answering key analytical questions:

1. What’s the breakdown of genres for the movies that were rated?
2. Which genres have the highest viewer satisfaction (highest ratings)?
3. How does the mean rating change across movie release years?
4. What are the 5 best‑rated movies that have at least 50 ratings? At least 150 ratings?

The app assumes the data file ``movie_ratings_EC.csv`` resides in the same
directory.  It explodes the pipe‑separated ``genres`` column into individual
genre values on load to enable per‑genre analysis.  A sidebar offers
interactive filters for age, gender, occupation, genres, and a minimum
sample threshold when computing genre satisfaction.

To run the dashboard locally, install the dependencies listed in
``requirements.txt`` and execute ``streamlit run streamlit_app.py`` from
this directory.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from typing import Tuple, List


@st.cache_data
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the MovieLens dataset and return both the original DataFrame and an
    exploded version where each row corresponds to a single (rating, genre)
    combination.

    The function will attempt to read ``movie_ratings.csv`` (the pre‑cleaned
    dataset); if this file is not present it falls back to ``movie_ratings_EC.csv``.
    The dataset is expected to contain either a ``genres`` column with
    pipe‑separated strings or a pre‑split ``genre`` column.  In either case
    the returned ``df_exp`` will include a ``genre`` column suitable for
    aggregation.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple of two DataFrames: ``df`` is the original data and
        ``df_exp`` is the exploded version with a ``genre`` column.
    """
    import os
    # Determine which data file to load
    candidates = [
        os.path.join(os.path.dirname(__file__), "movie_ratings.csv"),
        os.path.join(os.path.dirname(__file__), "movie_ratings_EC.csv"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            csv_path = path
            break
    else:
        raise FileNotFoundError(
            "No dataset file found. Expected 'movie_ratings.csv' or 'movie_ratings_EC.csv' in the current directory."
        )
    df = pd.read_csv(csv_path)
    # If a pre‑split 'genre' column exists use it directly, otherwise split the 'genres' column
    if "genre" in df.columns:
        # Ensure the genre field is a single genre per row.  Some datasets may still
        # contain pipe‑separated strings; handle both cases uniformly.
        if df["genre"].dtype == object:
            # Split strings on '|' to handle any multi‑genre entries
            df["genre_list"] = df["genre"].astype(str).str.split("|")
        else:
            df["genre_list"] = df["genre"].apply(lambda x: [x])
    elif "genres" in df.columns:
        df["genre_list"] = df["genres"].astype(str).str.split("|")
    else:
        raise KeyError("Dataset must contain either a 'genres' or 'genre' column")
    df_exp = df.explode("genre_list").rename(columns={"genre_list": "genre"})
    return df, df_exp


def filter_data(
    df_exp: pd.DataFrame,
    age_range: Tuple[int, int],
    genders: List[str],
    occupations: List[str],
    selected_genres: List[str],
) -> pd.DataFrame:
    """Apply user‑selected filters to the exploded data.

    Parameters
    ----------
    df_exp : pd.DataFrame
        Exploded DataFrame containing ratings with a ``genre`` column.
    age_range : Tuple[int, int]
        Tuple specifying the inclusive lower and upper bounds for user age.
    genders : list of str
        List of genders to include (e.g., ``['M', 'F']``).
    occupations : list of str
        List of occupations to include.
    selected_genres : list of str
        List of genres to include.  Rows whose ``genre`` is not in this list
        will be excluded.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame meeting all criteria.
    """
    mask = (
        (df_exp["age"] >= age_range[0])
        & (df_exp["age"] <= age_range[1])
        & (df_exp["gender"].isin(genders))
        & (df_exp["occupation"].isin(occupations))
        & (df_exp["genre"].isin(selected_genres))
    )
    return df_exp.loc[mask].copy()


def genre_breakdown_chart(df_filtered: pd.DataFrame) -> None:
    """Render a bar chart showing the distribution of ratings by genre.

    Parameters
    ----------
    df_filtered : pd.DataFrame
        DataFrame filtered according to user selections.  Must include a
        ``genre`` column.
    """
    genre_counts = (
        df_filtered.groupby("genre")["movie_id"].count().sort_values(ascending=False)
    )
    fig = px.bar(
        genre_counts,
        x=genre_counts.index,
        y=genre_counts.values,
        labels={"x": "Genre", "y": "Number of Ratings"},
        title="Distribution of Ratings by Genre",
    )
    fig.update_layout(
        xaxis_title="Genre",
        yaxis_title="Number of ratings",
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig, use_container_width=True)
    # Narrative insight
    top_genres = genre_counts.head(3)
    st.markdown(
        f"The **{top_genres.index[0]}** genre has the highest number of ratings"
        f" (≈{top_genres.iloc[0]:,}), followed by **{top_genres.index[1]}**"
        f" (≈{top_genres.iloc[1]:,}) and **{top_genres.index[2]}**"
        f" (≈{top_genres.iloc[2]:,}).  This suggests these genres are most popular among the filtered sample."
    )


def genre_satisfaction_chart(df_filtered: pd.DataFrame, min_ratings: int) -> None:
    """Render a bar chart showing mean rating by genre with a minimum sample threshold.

    Parameters
    ----------
    df_filtered : pd.DataFrame
        Filtered DataFrame including ``genre`` and ``rating`` columns.
    min_ratings : int
        Minimum number of ratings required for a genre to be included in the
        satisfaction comparison.  Genres with fewer ratings are excluded to
        avoid small‑sample noise.
    """
    genre_stats = (
        df_filtered.groupby("genre")["rating"]
        .agg(["count", "mean"])
        .rename(columns={"count": "num_ratings", "mean": "avg_rating"})
    )
    genre_stats = genre_stats[genre_stats["num_ratings"] >= min_ratings]
    genre_stats_sorted = genre_stats.sort_values("avg_rating", ascending=False)
    fig = px.bar(
        genre_stats_sorted,
        x=genre_stats_sorted.index,
        y="avg_rating",
        labels={"x": "Genre", "avg_rating": "Average Rating"},
        title=f"Genres by Average Rating (minimum {min_ratings} ratings)",
    )
    fig.update_layout(
        xaxis_title="Genre",
        yaxis_title="Average rating (1–5)",
        xaxis_tickangle=-45,
        yaxis_range=[df_filtered["rating"].min() - 0.1, df_filtered["rating"].max() + 0.1],
    )
    st.plotly_chart(fig, use_container_width=True)
    # Narrative insight
    if not genre_stats_sorted.empty:
        top = genre_stats_sorted.head(3)
        st.markdown(
            f"After requiring at least **{min_ratings}** ratings, the top genre by average rating is **{top.index[0]}**"
            f" (mean ≈{top.iloc[0]['avg_rating']:.2f}), followed by **{top.index[1]}**"
            f" (≈{top.iloc[1]['avg_rating']:.2f}) and **{top.index[2]}**"
            f" (≈{top.iloc[2]['avg_rating']:.2f})."
        )
    else:
        st.info("No genres meet the minimum rating threshold.")


def year_trend_chart(df_filtered: pd.DataFrame) -> None:
    """Render a line chart showing mean rating by release year.

    Parameters
    ----------
    df_filtered : pd.DataFrame
        Filtered DataFrame including ``year`` and ``rating`` columns.
    """
    # Remove missing years and ensure numeric type
    df_year = df_filtered.dropna(subset=["year"]).copy()
    df_year["year"] = df_year["year"].astype(int)
    year_stats = (
        df_year.groupby("year")["rating"].mean().reset_index().sort_values("year")
    )
    fig = px.line(
        year_stats,
        x="year",
        y="rating",
        labels={"year": "Release Year", "rating": "Mean Rating"},
        title="Mean Rating by Movie Release Year",
    )
    fig.update_layout(
        xaxis_title="Release year",
        yaxis_title="Mean rating (1–5)",
        yaxis_range=[df_filtered["rating"].min() - 0.1, df_filtered["rating"].max() + 0.1],
    )
    st.plotly_chart(fig, use_container_width=True)
    # Narrative insight
    earliest = int(year_stats.iloc[0]["year"])
    latest = int(year_stats.iloc[-1]["year"])
    high_year = year_stats.loc[year_stats["rating"].idxmax()]
    low_year = year_stats.loc[year_stats["rating"].idxmin()]
    st.markdown(
        f"Across the selected sample, movies released around **{int(high_year['year'])}** achieve the highest average ratings"
        f" (≈{high_year['rating']:.2f}), while those from **{int(low_year['year'])}** show the lowest"
        f" (≈{low_year['rating']:.2f})."
    )


def top_movies_section(df_filtered: pd.DataFrame) -> None:
    """Display the top 5 best‑rated movies at two minimum rating thresholds.

    Parameters
    ----------
    df_filtered : pd.DataFrame
        Filtered DataFrame including ``title`` and ``rating`` columns.
    """
    # Compute movie statistics
    movie_stats = (
        df_filtered.groupby("title")["rating"]
        .agg(["count", "mean"])
        .rename(columns={"count": "num_ratings", "mean": "avg_rating"})
    )
    def display_top_movies(min_count: int, caption: str) -> None:
        subset = movie_stats[movie_stats["num_ratings"] >= min_count]
        top5 = subset.sort_values(["avg_rating", "num_ratings"], ascending=[False, False]).head(5)
        if top5.empty:
            st.info(f"No movies have at least {min_count} ratings in the filtered dataset.")
        else:
            st.subheader(caption)
            st.dataframe(
                top5.reset_index().style.format(
                    {"avg_rating": "{:.2f}", "num_ratings": "{:,.0f}"}
                ),
                hide_index=True,
            )
            top_title = top5.index[0]
            st.markdown(
                f"The highest rated movie with at least **{min_count}** ratings is **{top_title}**"
                f" (mean ≈{top5.iloc[0]['avg_rating']:.2f} from {top5.iloc[0]['num_ratings']} ratings)."
            )

    display_top_movies(50, "Top 5 movies (≥ 50 ratings)")
    display_top_movies(150, "Top 5 movies (≥ 150 ratings)")


def main() -> None:
    # Load data
    df, df_exp = load_data()
    st.title("🎬 MovieLens 200k Ratings Dashboard")
    st.markdown(
        """
        Use the controls in the sidebar to filter the dataset by demographic
        attributes and genres.  The charts and tables below will update
        automatically to reflect your selections.
        """
    )

    # Sidebar filters
    st.sidebar.header("Filters")
    # Age filter
    min_age = int(df_exp["age"].min())
    max_age = int(df_exp["age"].max())
    age_range = st.sidebar.slider(
        "Age range", min_value=min_age, max_value=max_age, value=(min_age, max_age)
    )
    # Gender filter
    genders = list(df_exp["gender"].dropna().unique())
    selected_genders = st.sidebar.multiselect(
        "Gender", options=sorted(genders), default=sorted(genders)
    )
    # Occupation filter
    occupations = list(df_exp["occupation"].dropna().unique())
    selected_occupations = st.sidebar.multiselect(
        "Occupation", options=sorted(occupations), default=sorted(occupations)
    )
    # Genre filter
    all_genres = sorted(df_exp["genre"].dropna().unique())
    selected_genres = st.sidebar.multiselect(
        "Genres", options=all_genres, default=all_genres
    )
    # Minimum ratings threshold for satisfaction chart
    min_ratings = st.sidebar.slider(
        "Minimum ratings per genre", min_value=10, max_value=200, value=50, step=10
    )

    # Apply filters
    df_filtered = filter_data(
        df_exp,
        age_range=age_range,
        genders=selected_genders,
        occupations=selected_occupations,
        selected_genres=selected_genres,
    )

    # Display summary stats
    st.markdown(
        f"**{len(df_filtered):,}** ratings match the selected filters ("
        f"out of {len(df_exp):,} total ratings in the dataset)."
    )
    # Section: Genre breakdown
    st.header("1. Breakdown of genres")
    st.write(
        "This chart shows the number of ratings for each genre within the filtered sample."
    )
    genre_breakdown_chart(df_filtered)

    # Section: Viewer satisfaction by genre
    st.header("2. Viewer satisfaction by genre")
    st.write(
        "We compute the mean rating for each genre.  To avoid unreliable results with too few samples, use the slider to set a minimum number of ratings per genre."
    )
    genre_satisfaction_chart(df_filtered, min_ratings=min_ratings)

    # Section: Mean rating over release years
    st.header("3. Mean rating across release years")
    st.write(
        "This line chart plots the average rating of movies by their release year."
    )
    year_trend_chart(df_filtered)

    # Section: Best rated movies
    st.header("4. Best‑rated movies")
    st.write(
        "Below are the top five movies by average rating, first requiring at least 50 ratings and then 150 ratings."
    )
    top_movies_section(df_filtered)


if __name__ == "__main__":
    main()