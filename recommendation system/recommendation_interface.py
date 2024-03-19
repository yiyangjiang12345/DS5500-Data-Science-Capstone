import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix
import numpy as np

reviews_df = pd.read_csv('all_clean.csv')
games_df = pd.read_csv('game_score_with_genres.csv')



# Assuming reviews_df and games_df are already defined as per your project's description

# Step 1: Prepare User-User Collaborative Filtering Model
# Convert author_steamid and appid to categorical codes for matrix creation
reviews_df['user_code'] = reviews_df['author_steamid'].astype('category').cat.codes
reviews_df['game_code'] = reviews_df['appid'].astype('category').cat.codes

# Create interaction matrix based on sentiment scores
interaction_matrix = coo_matrix((reviews_df['sentiment_score'], 
                                 (reviews_df['user_code'], reviews_df['game_code'])))

# Fit KNN model for collaborative filtering
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(interaction_matrix)

# Step 2: Prepare Content-Based Filtering Model
# Vectorize game titles for content similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(games_df['game'])


def genre_similarity(genre_list1, genre_list2):
    # Convert genre lists to sets for easy comparison
    set1 = set(genre_list1)
    set2 = set(genre_list2)
    # Calculate intersection and union
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    # Compute Jaccard similarity as the size of intersection divided by the size of union
    similarity = len(intersection) / len(union) if union else 0
    return similarity

# Function to find similar users
def find_similar_users(user_code, n_neighbors=6):
    distances, indices = model_knn.kneighbors(interaction_matrix.getrow(user_code), n_neighbors=n_neighbors)
    return indices.flatten()[1:]

# Function to find similar games based on content
def enhanced_find_similar_games(game_id, n_neighbors=5):
    # Assuming game_id is an int and corresponds to 'appid' in games_df
    target_genres = set(games_df.loc[games_df['appid'] == game_id, 'genre_list'].iloc[0])
    game_indices = range(len(games_df))
    
    # Compute cosine similarity for all games
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix[games_df.index[games_df['appid'] == game_id].tolist()[0]])
    
    # Include genre similarity in scores
    enhanced_scores = []
    for i in game_indices:
        genre_sim = genre_similarity(games_df.iloc[i]['genre_list'], target_genres)
        enhanced_scores.append((i, (cosine_sim[i] + genre_sim) / 2))  # Average content and genre similarities
    
    # Sort games based on enhanced score and normalized_composite_score
    sorted_games = sorted(enhanced_scores, key=lambda x: x[1], reverse=True)[:n_neighbors]
    recommended_indices = [i[0] for i in sorted_games]
    
    # Further sort by normalized_composite_score
    recommended_games = games_df.iloc[recommended_indices].sort_values('normalized_composite_score', ascending=False)
    
    return recommended_games['game'].values

def hybrid_recommend(user_id, n_neighbors=5):
    if user_id not in reviews_df['author_steamid'].values:
        return "User ID not found in the dataset."
    
    user_code = reviews_df.loc[reviews_df['author_steamid'] == user_id, 'user_code'].iloc[0]
    similar_users = find_similar_users(user_code, n_neighbors)
    
    recommended_games = []
    for user_code in similar_users:
        user_games = reviews_df[reviews_df['user_code'] == user_code]['appid'].unique()
        for game_id in user_games:
            similar_games = enhanced_find_similar_games(game_id, n_neighbors)
            recommended_games.extend(similar_games)
            
    recommended_games = list(set(recommended_games))[:n_neighbors]
    
    recommendations_with_scores = []
    for game_title in recommended_games:
        game_info = games_df[games_df['game'] == game_title]
        if not game_info.empty:
            game_id = game_info.iloc[0]['appid']
            normalized_composite_score = game_info.iloc[0]['normalized_composite_score']
            average_sentiment_score = game_info.iloc[0]['average_sentiment']
            positive_rate = game_info.iloc[0]['positive_rate']
        
            recommendations_with_scores.append({
                'game': game_title,
                'normalized_composite_score': normalized_composite_score,
                'average_sentiment_score': average_sentiment_score,
                'positive_rate': positive_rate
            })

    
    return recommendations_with_scores




# Example usage
#user_id = 76561198847956367  # Replace with an actual author_steamid
#recommendations = hybrid_recommend(user_id)
#print(f"Recommended games: {recommendations}")

def main():
    while True:
        user_input = input("Enter a user_id (or 'X' to quit): ")
        if user_input.strip().upper() == 'X':
            break
        try:
            user_id = int(user_input)  # Convert input to integer if your user IDs are integers
            recommendations = hybrid_recommend(user_id)
            print(f"Recommended games: {recommendations}")
        except ValueError:
            print("Please enter a valid user_id or 'X' to quit.")

if __name__ == "__main__":
    main()
