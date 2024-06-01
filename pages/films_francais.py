#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import requests
from sklearn.decomposition import PCA
import random


# Load pickle files
df_fr = pd.read_pickle("./pages/df_fr.pkl.gz")
df_fr_actor = pd.read_pickle("./pages/df_fr_actor.pkl.gz")
tmdb_df_1 = pd.read_pickle("./pages/tmdb_df_1.pkl.gz")
df_fr_genres = pd.read_pickle("./pages/df_fr_genres.pkl.gz")


# Extract the list of actors
actors_list = df_fr_actor["primaryName"].unique().tolist()
actors_list.sort()  # Optional: Sort the list of actors
predefined_genres = [""] + ["Action","Adventure","Animation","Biography","Comedy","Crime","Documentary","Drama","Family","Fantasy",
"History","Horror","Music","Musical","Mystery","Romance","Sci-Fi","Sport","Thriller","War","Western"]

# Streamlit setup
st.set_page_config(
    page_title="Pop Creuse",
    layout="centered",
    page_icon=':popcorn:'
)

# Background
# Background
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://raw.githubusercontent.com/MaximeBarski/streamlit_popcreuse/main/cartoon_sidebar_background.png");
background-size: cover;
background-position: center center;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-image: url('https://raw.githubusercontent.com/MaximeBarski/streamlit_popcreuse/main/cartoon_sidebar_background.png');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    </style>
    """,
    unsafe_allow_html=True
)

list_film_deroulante_films = [""]+list(df_fr["originalTitle"])

# TITRE
st.markdown("<h1 style='text-align: center; color: black;'>Films Français</h1>", unsafe_allow_html=True)

# API settings
url_api = "http://www.omdbapi.com/?i="
key_api = "&apikey=b0402387"
url_imdb = 'https://www.imdb.com/title/'

# Define feature weights
array_size = df_fr.shape[1] - 2  # Assuming first two columns are 'originalTitle' and 'tconst'
feature_weights = np.zeros(array_size, dtype=int)
feature_weights[0:5] = 1
feature_weights[5] = 3
feature_weights[6] = 1
feature_weights[7:31] = 20
feature_weights[31:] = 10

# Extract only the relevant features for training
X = df_fr.iloc[:, 2:]

# Step 1: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Apply the feature weights
X_weighted = X_scaled * feature_weights

# Step 3: Prepare PCA
pca = PCA(n_components=0.7)
X_pca_weighted = pca.fit_transform(X_weighted)

# Step 4: Use NearestNeighbors with the transformed data
nn = NearestNeighbors(n_neighbors=4)
nn.fit(X_pca_weighted)


with st.form("form_1"):
    st.subheader("Choisis le titre d'un film et clique sur 'Go !'")
    films = st.selectbox("Films:", list_film_deroulante_films)
    submit_1 = st.form_submit_button("Go !")

if submit_1:
    df_film_choisi = df_fr[df_fr["originalTitle"] == films]
    if df_film_choisi.empty:
        st.error("Aucun film trouvé pour le titre sélectionné.")
    else:
        film_choisi = df_film_choisi.iloc[:, 2:]
        film_choisi_scaled = scaler.transform(film_choisi)
        film_choisi_weighted = film_choisi_scaled * feature_weights
        film_choisi_weighted_pca = pca.transform(film_choisi_weighted)

        distances, indices = nn.kneighbors(film_choisi_weighted_pca)

        tconst = df_fr.iloc[indices.flatten()[1:], 0].values  # Exclude the first result as it is the same film
        suggestion = df_fr.iloc[indices.flatten()[1:], 1].values

        col1, col2, col3 = st.columns(3)
        for films, code_film, col in zip(suggestion, tconst, [col1, col2, col3]):
            with col:
                url = url_api + str(code_film) + key_api
                url_imdb2 = url_imdb + str(code_film)
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    data = response.json()
                    url_image = data.get('Poster', None)
                    if url_image:
                        st.image(url_image, width=200)
                    else:
                        st.write("Pas d'affiche disponible")
                except requests.exceptions.RequestException as e:
                    st.write(f"Erreur de récupération de données pour {films}: {e}")

                if isinstance(films, str):
                    st.write(f" - [{films}]({url_imdb2})")
                else:
                    st.write(f" - [{films}]({url_imdb2})")

with st.form("form_2"):
    st.subheader("Choisis un acteur / une actrice et clique sur 'Go !'")
    actor = st.selectbox("Acteurs / Actrices :", [""] + actors_list)
    submit_2 = st.form_submit_button("Go !")

if submit_2 and actor != "Choisis un acteur que tu aimes":
    filtered_ids = df_fr_actor["knownForTitles"][df_fr_actor["primaryName"].str.contains(actor, case=False, na=False)]

    list_ids = filtered_ids.tolist()
    unique_ids = set()
    for ids in list_ids:
        for id in ids.split(','):
            unique_ids.add(id.strip())
    unique_ids = list(unique_ids)

    original_titles = []
    for id in unique_ids:
        matching_titles = tmdb_df_1["original_title"][tmdb_df_1["imdb_id"] == id].tolist()
        original_titles.extend(matching_titles)

    original_titles = original_titles[:3]

    if original_titles:
        st.subheader(f"Si tu aimes {actor}, je te suggère fortement les films suivants:")
        col1, col2, col3 = st.columns(3)
        for title, id, col in zip(original_titles, unique_ids[:3], [col1, col2, col3]):
            with col:
                url1 = url_api + str(id) + key_api
                url1_imdb2 = url_imdb + str(id)
                try:
                    response = requests.get(url1)
                    response.raise_for_status()
                    data = response.json()
                    url_image = data.get('Poster')
                    if url_image:
                        st.image(url_image, width=200)
                    else:
                        st.write("Pas d'affiche disponible.")
                except requests.exceptions.RequestException as e:
                    st.write(f"Erreur de récupération de données pour {title}: {e}")

                st.write(f" - [{title}]({url1_imdb2})")
    else:
        st.write(f"Aucun film trouvé pour {actor}.")


with st.form("form_3"):
    st.subheader("Choisis un genre et clique sur 'Go !'")
    selected_genre = st.selectbox("Genre:", predefined_genres)
    submit_3 = st.form_submit_button("Go !")

if submit_3 and selected_genre:  # Check if a genre is selected
    # Create a mask to filter rows containing the selected genre
    mask = df_fr_genres["genres"].str.contains(selected_genre, case=False, na=False)
    filtered_movies = df_fr_genres[mask]

    # Extract the 'tconst' column (which is assumed to be the unique movie identifier)
    unique_ids = filtered_movies["tconst"].tolist()

    original_titles = []
    for id in unique_ids:
        matching_titles = tmdb_df_1["original_title"][tmdb_df_1["imdb_id"] == id].tolist()
        original_titles.extend(matching_titles)

    # Shuffle the original_titles and unique_ids to get random movies
    combined_list = list(zip(original_titles, unique_ids))
    random.shuffle(combined_list)
    shuffled_titles, shuffled_ids = zip(*combined_list)
    selected_titles = shuffled_titles[:3]
    selected_ids = shuffled_ids[:3]

    if selected_titles:
        st.subheader(f"Si tu aimes {selected_genre}, je te recommande :")
        col1, col2, col3 = st.columns(3)
        for title, id, col in zip(selected_titles, selected_ids, [col1, col2, col3]):
            with col:
                url1 = url_api + str(id) + key_api
                url1_imdb2 = url_imdb + str(id)
                try:
                    response = requests.get(url1)
                    response.raise_for_status()
                    data = response.json()
                    url_image = data.get('Poster')
                    if url_image:
                        st.image(url_image, width=200)
                    else:
                        st.write("Pas d'affiche disponible.")
                except requests.exceptions.RequestException as e:
                    st.write(f"Erreur de récupération des données pour {title}: {e}")

                st.write(f" - [{title}]({url1_imdb2})")
    else:
        st.write(f"Aucun film trouvé pour {selected_genre}.")

