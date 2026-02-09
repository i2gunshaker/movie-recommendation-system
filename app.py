import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer

# ------------------------
# Streamlit Page Configuration
# ------------------------
st.set_page_config(
    page_title="Movie Recommendation System 🎬",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------
# Custom CSS for Enhanced Styling
# ------------------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        body {
            background: linear-gradient(to bottom, #000000, #1a1a1a);
            color: #ffffff;
            font-family: 'Poppins', sans-serif;
        }
        /* Sidebar Styling */
        .css-6qob1r { background-color: #181818 !important; border-right: 1px solid #2a2a2a; }
        /* Hero Section */
        .hero-container { text-align: center; padding: 40px 0; }
        .hero-title { font-size: 3rem; font-weight: 700; color: #ffffff; margin-bottom: 10px; animation: fadeInDown 1s ease-out; }
        .hero-subtitle { font-size: 1.2rem; color: #b3b3b3; margin-bottom: 30px; animation: fadeInUp 1s ease-out; }
        @keyframes fadeInDown { from {opacity:0; transform: translateY(-20px);} to {opacity:1; transform: translateY(0);} }
        @keyframes fadeInUp { from {opacity:0; transform: translateY(20px);} to {opacity:1; transform: translateY(0);} }
        /* Movie Details */
        .movie-detail-container { display: flex; flex-direction: row; justify-content: space-between; background: linear-gradient(to right, #000000, #1a1a1a); border-radius: 12px; padding: 30px; margin-bottom: 40px; box-shadow: 0 4px 20px rgba(0,0,0,0.5); gap: 20px; animation: fadeIn 0.5s ease-in; }
        @keyframes fadeIn { from {opacity:0;} to {opacity:1;} }
        .poster-container { flex: 0 0 40%; display: flex; align-items: center; justify-content: center; }
        .poster-container img { max-width: 100%; height: auto; border-radius: 10px; transition: transform 0.3s ease; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
        .poster-container img:hover { transform: scale(1.03); }
        .details-container { flex: 0 0 60%; display: flex; flex-direction: column; justify-content: center; background: rgba(255, 255, 255, 0.05); border-radius: 10px; padding: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); animation: fadeInRight 1s ease-out; }
        @keyframes fadeInRight { from {opacity:0; transform: translateX(20px);} to {opacity:1; transform: translateX(0);} }
        .movie-title { font-size: 2.4rem; font-weight: 700; margin-bottom: 10px; color: #ffffff; animation: popIn 0.5s ease-in-out; }
        @keyframes popIn { 0% {transform: scale(0.8); opacity: 0;} 100% {transform: scale(1); opacity: 1;} }
        .genres-row { margin-bottom: 15px; display: flex; flex-wrap: wrap; gap: 10px; }
        .genre-badge { background-color: #333333; color: #f1c40f; font-size: 0.9rem; padding: 5px 10px; border-radius: 20px; transition: background-color 0.3s ease; }
        .genre-badge:hover { background-color: #f1c40f; color: #333333; }
        .rating-row { display: flex; align-items: center; margin-bottom: 8px; color: #f1c40f; font-size: 1.1rem; }
        .rating-row .star { margin-right: 5px; animation: shine 2s infinite; }
        @keyframes shine { 0% { text-shadow: 0 0 5px #f1c40f; } 50% { text-shadow: 0 0 20px #f1c40f; } 100% { text-shadow: 0 0 5px #f1c40f; } }
        .release-year { font-size: 0.95rem; color: #bfbfbf; margin-bottom: 15px; }
        .synopsis { font-size: 1rem; line-height: 1.5; color: #e0e0e0; }
        @media(max-width: 900px) {
            .movie-detail-container { flex-direction: column; align-items: center; }
            .poster-container, .details-container { flex: 0 0 100%; text-align: center; }
            .poster-container { margin-bottom: 20px; }
            .details-container { padding: 15px; }
        }
        /* Recommendation Cards - Grid Layout */
        .rec-card {
            background-color: #1e1e1e;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
            margin: 10px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            cursor: pointer;
        }
        .rec-card:hover {
            transform: translateY(-5px) scale(1.02);
            box-shadow: 0 8px 20px rgba(0,0,0,0.4);
        }
        .rec-card img {
            width: 100%;
            height: auto;
            border-bottom: 1px solid #333;
        }
        .rec-card-content {
            padding: 10px;
        }
        .rec-title {
            font-size: 1.1rem;
            font-weight: 600;
            margin: 5px 0;
            color: #ffffff;
        }
        .rec-rating {
            color: #f1c40f;
            font-size: 0.95rem;
        }
        .rec-genres { margin-top: 5px; }
        /* Favorites Button */
        .fav-button {
            background-color: #e50914;
            color: #fff;
            border: none;
            padding: 5px 10px;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .fav-button:hover { background-color: #b20710; }
    </style>
""", unsafe_allow_html=True)

# ------------------------
# Caching and Data Loading Functions
# ------------------------
@st.cache_data(show_spinner=False)
def load_and_validate_data(filepath):
    """Load the dataset and validate required columns."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"File not found at {filepath}. Please check the path and try again.")
        st.stop()
    except pd.errors.ParserError:
        st.error("Error parsing the CSV file. Please ensure it's formatted correctly.")
        st.stop()

    required_cols = ["Title", "Genres", "Description", "Poster URL", "IMDb Rating", "Release Year", "Director", "Cast"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        st.stop()

    df['Title_Year'] = df['Title'] + " (" + df['Release Year'].astype(int).astype(str) + ")"
    return df

@st.cache_data(show_spinner=False)
def preprocess(df):
    """Normalize ratings and prepare combined feature strings."""
    scaler = MinMaxScaler()
    df['Normalized IMDb Rating'] = scaler.fit_transform(df[['IMDb Rating']])
    # Combine relevant features into a single string for embedding
    df['Combined_Features'] = df.apply(
        lambda row: f"{row['Genres']} {row['Description']} {row['Director']} {row['Cast']}", axis=1
    )
    return df

@st.cache_resource(show_spinner=False)
def get_embeddings(df):
    """Generate embeddings for combined features."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    combined_features = df['Combined_Features'].tolist()
    embeddings = model.encode(combined_features, convert_to_tensor=False, show_progress_bar=True)
    return embeddings

def compute_similarity(embeddings, df, weights):
    """
    Compute cosine similarity based on embeddings and incorporate IMDb rating similarity.
    The weights dict should include keys: 'genres', 'description', 'director', 'cast', 'rating'.
    """
    sim_matrix = cosine_similarity(embeddings)
    ratings_sim = cosine_similarity(df[['Normalized IMDb Rating']])
    combined_sim = (
        (weights['genres'] + weights['description'] + weights['director'] + weights['cast']) * sim_matrix
        + weights['rating'] * ratings_sim
    )
    return combined_sim

def get_recommendations(title_year, df, similarity_matrix, top_n=9):
    """Retrieve top N movie recommendations based on similarity."""
    if title_year not in df['Title_Year'].values:
        return None
    idx = df.index[df['Title_Year'] == title_year][0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices]

def get_unique_genres(df):
    """Extract unique genres from the dataframe."""
    genres_series = df['Genres'].dropna().str.split(',')
    unique_genres = sorted({genre.strip() for sublist in genres_series for genre in sublist})
    return unique_genres

# ------------------------
# Session State for Favorites
# ------------------------
if 'favorites' not in st.session_state:
    st.session_state.favorites = []

def add_to_favorites(movie):
    if movie['Title_Year'] not in st.session_state.favorites:
        st.session_state.favorites.append(movie['Title_Year'])
        st.success(f"Added '{movie['Title']}' to favorites!")
    else:
        st.info(f"'{movie['Title']}' is already in your favorites.")

# ------------------------
# Sidebar Configuration and User Options
# ------------------------
st.sidebar.title("🎬 Movie Recommendation System")

# Data file path (update as needed)
data_filepath = "/Users/i2gunshaker/Downloads/top20k_movies_project2.csv"  
movies_df = load_and_validate_data(data_filepath)
movies_df = preprocess(movies_df)
embeddings = get_embeddings(movies_df)
unique_genres = get_unique_genres(movies_df)

# Advanced Filters
st.sidebar.subheader("🔍 Filter Movies")
min_year = int(movies_df['Release Year'].min())
max_year = int(movies_df['Release Year'].max())
year_range = st.sidebar.slider("Select Release Year Range", min_year, max_year, (min_year, max_year), step=1)

min_rating = float(movies_df['IMDb Rating'].min())
max_rating = float(movies_df['IMDb Rating'].max())
rating_range = st.sidebar.slider("Select IMDb Rating Range", min_rating, max_rating, (min_rating, max_rating), step=0.1)

filtered_df = movies_df[
    (movies_df['Release Year'] >= year_range[0]) &
    (movies_df['Release Year'] <= year_range[1]) &
    (movies_df['IMDb Rating'] >= rating_range[0]) &
    (movies_df['IMDb Rating'] <= rating_range[1])
]

# Selection Method and Search
st.sidebar.subheader("🎞️ Select Movie")
selection_method = st.sidebar.radio("Choose Selection Method:", ["Select a Movie", "Select by Genre", "Search by Title"])
selected_movie = None

if selection_method == "Select a Movie":
    movie_options = filtered_df['Title_Year'].sort_values().tolist()
    if movie_options:
        selected_movie = st.sidebar.selectbox("Select a movie you like:", movie_options, index=0)
    else:
        st.sidebar.write("No movies found with current filters.")

elif selection_method == "Select by Genre":
    selected_genre = st.sidebar.selectbox("Select a genre:", unique_genres)
    genre_filtered_df = filtered_df[filtered_df['Genres'].str.contains(selected_genre, case=False, na=False)]
    genre_filtered_df = genre_filtered_df.sort_values(by='IMDb Rating', ascending=False).head(100)
    if genre_filtered_df.empty:
        st.sidebar.write("No movies found for the selected genre.")
    else:
        selected_movie = st.sidebar.selectbox("Select a movie from the top 100:", genre_filtered_df['Title_Year'], index=0)

elif selection_method == "Search by Title":
    search_term = st.sidebar.text_input("Enter part of the movie title:")
    if search_term:
        search_results = filtered_df[filtered_df['Title'].str.contains(search_term, case=False, na=False)]
        if not search_results.empty:
            selected_movie = st.sidebar.selectbox("Search Results:", search_results['Title_Year'], index=0)
        else:
            st.sidebar.write("No movies match your search.")

# Use default weights (no adjustment)
default_weights = {'genres': 0.4, 'description': 0.3, 'director': 0.2, 'cast': 0.1, 'rating': 0.1}
user_weights = default_weights

# Recompute the similarity matrix with default weights
similarity_matrix = compute_similarity(embeddings, movies_df, user_weights)

# Favorites Section in Sidebar
st.sidebar.subheader("❤️ Favorites")
if st.session_state.favorites:
    for fav in st.session_state.favorites:
        st.sidebar.markdown(f"- {fav}")
else:
    st.sidebar.write("No favorites yet.")

# ------------------------
# Main Page Content
# ------------------------
st.markdown("""
    <div class='hero-container'>
        <div class='hero-title'>🎬 Absolute Cinema</div>
        <div class='hero-subtitle'>🔍 Explore, Discover, and Enjoy</div>
    </div>
""", unsafe_allow_html=True)

# If a movie is selected, display details and recommendations
if selected_movie:
    with st.spinner("Fetching recommendations..."):
        recommendations = get_recommendations(selected_movie, movies_df, similarity_matrix)

    selected = movies_df[movies_df['Title_Year'] == selected_movie].iloc[0]

    # Display Selected Movie Details
    poster_html = (f"<img src='{selected['Poster URL']}' alt='Poster'>"
                   if pd.notnull(selected['Poster URL']) and selected['Poster URL'].startswith("http")
                   else "<div style='width:100%;height:300px;background:#333;display:flex;align-items:center;justify-content:center;color:#fff;border-radius:10px;'>No Image Available</div>")
    genres_badges = "".join([f"<span class='genre-badge'>{g.strip()}</span>" for g in selected['Genres'].split(",")]) if pd.notnull(selected['Genres']) else ""
    rating = f"{selected['IMDb Rating']}/10" if not np.isnan(selected['IMDb Rating']) else "N/A"
    year = int(selected['Release Year']) if not np.isnan(selected['Release Year']) else "N/A"
    description = selected['Description'] if pd.notnull(selected['Description']) else "No description available."
    director = selected['Director'] if pd.notnull(selected['Director']) else "N/A"
    cast = selected['Cast'] if pd.notnull(selected['Cast']) else "N/A"

    st.markdown(f"""
    <div class='movie-detail-container'>
        <div class='poster-container'>{poster_html}</div>
        <div class='details-container'>
            <div class='movie-title'>{selected['Title']}</div>
            <div class='genres-row'>{genres_badges}</div>
            <div class='rating-row'><span class='star'>⭐</span> {rating}</div>
            <div class='release-year'>Released: {year}</div>
            <div><span class="detail-label">Director:</span> {director}</div>
            <div><span class="detail-label">Cast:</span> {cast}</div>
            <div class='synopsis'>{description}</div>
            <br>
            <a target="_blank" href="https://www.youtube.com/results?search_query={selected['Title'].replace(' ', '+')}+trailer" style="color:#e50914;font-weight:bold;">▶ Watch Trailer</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Display Recommendations in a Grid Layout
    st.markdown("<h2 style='text-align:center; color:#e50914;'>You Might Also Like:</h2>", unsafe_allow_html=True)
    if recommendations is not None and not recommendations.empty:
        rec_cards = []
        for _, row in recommendations.iterrows():
            rec_poster = (row['Poster URL'] if pd.notnull(row['Poster URL']) and row['Poster URL'].startswith("http")
                          else "https://via.placeholder.com/300x445?text=No+Image")
            rec_rating = f"{row['IMDb Rating']}/10" if not np.isnan(row['IMDb Rating']) else "N/A"
            rec_year = int(row['Release Year']) if not np.isnan(row['Release Year']) else "N/A"
            rec_genres = ", ".join([g.strip() for g in row['Genres'].split(",")]) if pd.notnull(row['Genres']) else ""
            
            card_html = f"""
            <div class="rec-card" style="min-width:250px;">
                <img src="{rec_poster}" alt="Poster">
                <div class="rec-card-content">
                    <div class="rec-title">{row['Title']} ({rec_year})</div>
                    <div class="rec-rating">⭐ {rec_rating}</div>
                    <div class="rec-genres">{rec_genres}</div>
                </div>
            </div>
            """
            rec_cards.append((row, card_html))
        
        # Display cards in a grid using columns
        num_cols = 3
        rows_of_cards = [rec_cards[i:i+num_cols] for i in range(0, len(rec_cards), num_cols)]
        for row_cards in rows_of_cards:
            cols = st.columns(len(row_cards))
            for col, (movie_row, card_html) in zip(cols, row_cards):
                with col:
                    st.markdown(card_html, unsafe_allow_html=True)
                    if st.button("Add to Favorites", key=f"fav_{movie_row.name}"):
                        add_to_favorites(movie_row)
    else:
        st.write("No recommendations found. Please try another movie.")

# ------------------------
# Reset Button (Optional)
# ------------------------
if st.button("Reset Filters and Selections"):
    st.experimental_rerun()