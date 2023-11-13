import pandas as pd
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import sys

def getSubjectivity(text):
    '''
    Getting the Subjectivity using TextBlob
    '''
    return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
    '''
    Getting the Polarity using TextBlob
    '''
    return TextBlob(text).sentiment.polarity

def getAnalysis(score, task="polarity"):
    '''
    Categorizing the Polarity & Subjectivity score
    '''
    if task == "subjectivity":
        if score < 1/3:
            return "low"
        elif score > 1/3:
            return "high"
        else:
            return "medium"
    else:
        if score < 0:
            return 'Negative'
        elif score == 0:
            return 'Neutral'
        else:
            return 'Positive'


def ohe_prep(df, column, new_name): 
    ''' 
    Create One Hot Encoded features of a specific column
    ---
    Input: 
    df (pandas dataframe): Spotify Dataframe
    column (str): Column to be processed
    new_name (str): new column name to be used
        
    Output: 
    tf_df: One-hot encoded features 
    '''
    
    tf_df = pd.get_dummies(df[column])
    feature_names = tf_df.columns
    tf_df.columns = [new_name + "|" + str(i) for i in feature_names]
    tf_df.reset_index(drop = True, inplace = True)    
    return tf_df




def create_feature_set(df, float_cols):
    '''
    Process spotify df to create a final set of features that will be used to generate recommendations
    ---
    Input: 
    df (pandas dataframe): Spotify Dataframe
    float_cols (list(str)): List of float columns that will be scaled
            
    Output: 
    final (pandas dataframe): Final set of features 
    '''
    
    # Tfidf genre lists
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['genres'].apply(lambda x: " ".join(x)))
    genre_df = pd.DataFrame((tfidf_matrix.toarray() + 1) * 0.2)
    genre_df.columns = ['genre' + "|" + i for i in tfidf.get_feature_names_out()]
    genre_df.drop(columns='genre|unknown') # drop unknown genre
    genre_df.reset_index(drop = True, inplace=True)
    
    # Sentiment analysis
    df = sentiment_analysis(df, "name")

    # One-hot Encoding
    subject_ohe = ohe_prep(df, 'subjectivity','subject') * 0.1
    polar_ohe = ohe_prep(df, 'polarity','polar') * 0.1
    key_ohe = ohe_prep(df, 'key','key') * 0.2
    mode_ohe = ohe_prep(df, 'mode','mode') * 0.3

    # Min Max scale
    # Scale popularity columns
    pop = df[["artist_pop","track_pop"]].reset_index(drop = True)
    scaler = MinMaxScaler()
    pop_scaled = pd.DataFrame(scaler.fit_transform(pop), columns = pop.columns) * 0.4 

    # Scale audio columns
    floats = df[float_cols].reset_index(drop = True)
    scaler = MinMaxScaler()
    floats_scaled = pd.DataFrame(scaler.fit_transform(floats), columns = floats.columns) * 0.6

    # Concanenate all features
    final = pd.concat([genre_df, floats_scaled, pop_scaled, subject_ohe, polar_ohe, key_ohe, mode_ohe], axis = 1)
    
    # Add song id
    final['id']=df['id'].values
    
    return final

def generate_artist_feature(complete_feature_set, artist_df):
    '''
    Summarize a user's artist into a single vector
    ---
    Input: 
    complete_feature_set (pandas dataframe): Dataframe which includes all of the features for the spotify songs
    playlist_df (pandas dataframe): playlist dataframe
        
    Output: 
    complete_feature_set_playlist_final (pandas series): single vector feature that summarizes the playlist
    complete_feature_set_nonplaylist (pandas dataframe): 
    '''
    
    # Find song features in the playlist
    complete_feature_set_playlist = complete_feature_set[complete_feature_set['id'].isin(artist_df['id'].values)]
    # Find all non-playlist song features
    complete_feature_set_nonplaylist = complete_feature_set[~complete_feature_set['id'].isin(artist_df['id'].values)]
    complete_feature_set_playlist_final = complete_feature_set_playlist.drop(columns = "id")
    return complete_feature_set_playlist_final.sum(axis = 0), complete_feature_set_nonplaylist




def sentiment_analysis(df, text_col):
    '''
    Perform sentiment analysis on text
    ---
    Input:
    df (pandas dataframe): Dataframe of interest
    text_col (str): column of interest
    '''
    df['subjectivity'] = df[text_col].apply(getSubjectivity).apply(lambda x: getAnalysis(x,"subjectivity"))
    df['polarity'] = df[text_col].apply(getPolarity).apply(getAnalysis)
    return df
    
def generate_playlist_recos(df, features, nonplaylist_features):
    '''
    Generated recommendation based on songs in aspecific playlist.
    ---
    Input: 
    df (pandas dataframe): spotify dataframe
    features (pandas series): summarized playlist feature (single vector)
    nonplaylist_features (pandas dataframe): feature set of songs that are not in the selected playlist
        
    Output: 
    non_playlist_df_top_40: Top 40 recommendations for that playlist
    '''
    
    non_playlist_df = df[df['id'].isin(nonplaylist_features['id'].values)]
    # Find cosine similarity between the playlist and the complete song set
    X = nonplaylist_features.drop('id', axis = 1).values
    y = features.values.reshape(1, -1)
    non_playlist_df['sim'] = cosine_similarity(X, y)[:,0]
    non_playlist_df_top_40 = non_playlist_df.sort_values('sim',ascending = False).head(40)
    
    return non_playlist_df_top_40


if __name__ == "__main__":
    # Grab artist name from args
    artist = " ".join(sys.argv[1:])
    # Read in data
    data = pd.read_csv("songs.csv")
    unfiltered = pd.read_csv("songs.csv")
    #Drop useless info
    data = data.drop(['type', 'uri', 'track_href', 'analysis_url', 'duration_ms', 'time_signature'], axis = 1)

    #Split each row's genres into a list
    data["genres"] = data.genres.str.split(" ")
    data['name'] = data.name.astype(str)
    float_cols = data.dtypes[data.dtypes == 'float64'].index.values

    #Create feature set
    temp = create_feature_set(data, float_cols)

    #Generate artist features and playlist reccomendations
    complete_feature_set_playlist_vector, complete_feature_set_nonplaylist = generate_artist_feature(temp, unfiltered[unfiltered["artist"] == artist])
    recommend = generate_playlist_recos(data, complete_feature_set_playlist_vector, complete_feature_set_nonplaylist)

    #Write recommendations to a csv file
    recommend[["name", "artist"]].to_csv("recommendations.csv", index = False)



