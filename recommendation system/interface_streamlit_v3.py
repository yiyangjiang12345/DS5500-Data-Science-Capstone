import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix
import openai
import os

# Load the data
reviews_df = pd.read_csv(r'C:\Users\17191\OneDrive\桌面\cs5500\all_clean.csv')
games_df = pd.read_csv(r'C:\Users\17191\OneDrive\桌面\cs5500\game_score_with_genres.csv')

# For the content-based system
games_df['genre_list'] = games_df['Top 3 Genres'].apply(lambda x: x.split(';'))
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(games_df['game'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Preparing the hybrid recommendation system
reviews_df['user_code'] = reviews_df['author_steamid'].astype('category').cat.codes
reviews_df['game_code'] = reviews_df['appid'].astype('category').cat.codes
interaction_matrix = coo_matrix((reviews_df['sentiment_score'], (reviews_df['user_code'], reviews_df['game_code'])))

model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(interaction_matrix)

def genre_similarity(genre_list1, genre_list2):
    set1 = set(genre_list1)
    set2 = set(genre_list2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0

def find_similar_users(user_code, n_neighbors=6):
    distances, indices = model_knn.kneighbors(interaction_matrix.getrow(user_code), n_neighbors=n_neighbors)
    return indices.flatten()[1:]

def enhanced_find_similar_games(game_id, n_neighbors=5):
    target_genres = set(games_df.loc[games_df['appid'] == game_id, 'genre_list'].iloc[0])
    game_indices = range(len(games_df))
    
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix[games_df.index[games_df['appid'] == game_id].tolist()[0]])
    
    enhanced_scores = []
    for i in game_indices:
        genre_sim = genre_similarity(games_df.iloc[i]['genre_list'], target_genres)
        enhanced_scores.append((i, (cosine_sim[i] + genre_sim) / 2))
    
    sorted_games = sorted(enhanced_scores, key=lambda x: x[1], reverse=True)[:n_neighbors]
    recommended_indices = [i[0] for i in sorted_games]
    
    recommended_games = games_df.iloc[recommended_indices].sort_values('normalized_composite_score', ascending=False)
    
    return recommended_games

# Hybrid recommendation function
def hybrid_recommend(user_id, n_neighbors=5):
    if user_id not in reviews_df['author_steamid'].values:
        return []
    
    user_code = reviews_df.loc[reviews_df['author_steamid'] == user_id, 'user_code'].iloc[0]
    similar_users = find_similar_users(user_code, n_neighbors)
    
    recommended_games = pd.DataFrame()
    for user_code in similar_users:
        user_games = reviews_df[reviews_df['user_code'] == user_code]['appid'].unique()
        for game_id in user_games:
            similar_games = enhanced_find_similar_games(game_id, n_neighbors)
            recommended_games = pd.concat([recommended_games, similar_games])
            
    recommended_games = recommended_games.drop_duplicates(subset=['game']).head(n_neighbors)
    return recommended_games.to_dict('records')

# Content-based recommendation function
def recommend_games(title, cosine_sim):
    idx = games_df.index[games_df['game'] == title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6] # Top 5 similar games
    
    # Enhance recommendations by genre similarity
    target_genres = games_df.iloc[idx]['genre_list']
    enhanced_scores = [(games_df.iloc[i[0]]['game'], genre_similarity(games_df.iloc[i[0]]['genre_list'], target_genres), i[1]) for i in sim_scores]
    
    # Sort by genre similarity and then by normalized_composite_score
    recommended = sorted(enhanced_scores, key=lambda x: (x[1], games_df[games_df['game'] == x[0]]['normalized_composite_score'].values[0]), reverse=True)
    
    return recommended


def generate_reason(game_title):
    openai.api_key = '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'

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Explain why someone should play the game '{game_title}' in two sentences."}
            ]
        )
        return response['choices'][0]['message']['content'].strip()
    except openai.error.OpenAIError as e:
        print(f"An OpenAI error occurred: {e}")
        return "Reasoning not available due to an API error."
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return "Reasoning not available due to an unexpected error."

def streamlit_app():
    st.title('Game Recommendation System')

    # Choose the recommendation system
    rec_system = st.radio("Choose the recommendation system:",
                          ('Hybrid', 'Content-Based'))

    user_input = st.text_input("Enter your user ID or game title:", '')

    if st.button('Get Recommendations'):
        if user_input:
            if rec_system == 'Hybrid':
                try:
                    user_id_int = int(user_input)
                    recommendations = hybrid_recommend(user_id_int)
                    if recommendations:
                        st.write("Recommended games for user ID:")
                        for game in recommendations:
                            reason = generate_reason(game['game'])
                            st.write(f"{game['game']} - Score: {game['normalized_composite_score']}")
                            st.write(f"Reason to play: {reason}")
                    else:
                        st.write("No recommendations available.")
                except ValueError:
                    st.error("Please enter a valid user ID.")
            elif rec_system == 'Content-Based':
                recommendations = recommend_games(user_input, cosine_sim)
                if recommendations:
                    st.write("Recommended games for game title:")
                    for game, _, _ in recommendations:
                        reason = generate_reason(game)
                        st.write(f"{game}")
                        st.write(f"Reason to play: {reason}")
                else:
                    st.write("No recommendations available.")
        else:
            st.error("Please enter a user ID or game title.")

if __name__ == "__main__":
    streamlit_app()