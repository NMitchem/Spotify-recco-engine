import os
from os.path import join, dirname
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import re 
import pandas as pd
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# Load clientID and Secret from your .env file 
load_dotenv()
cid = os.environ.get("CID")
secret = os.environ.get("SECRET")

def playlist_grab(users):
    '''
    Load playlists for a given user and extract name and URI from each playlist
    ---
    Input:
    List of spotify users with public playlists

    Output:
    A list of playlist names and URIs

    '''
    playlist_list = []
    for user in users:
        playlists = sp.user_playlists(user)
        while playlists:
            for playlist in playlists['items']:
                playlist_list.append([playlist["name"], playlist['uri']])
            if playlists['next']:
                playlists = sp.next(playlists)
            else:
                playlists = None
    return playlist_list

def playlist_extract(playlist_name, uri):

    '''
    Extract songs for each playlist name / URI
    ---
    Input:
    List of playlist name and URIs

    Output:
    A list of song data for a given playlist
    '''

   # Define a common list to extract data to
    all_data = []

   # Define an inner function to grab data and manipulate it 
    def tracks_to_features(uris, playlist_name):

        '''
        A helper function that extracts song data given a URI for a song
        ---
        Input:
        Song URIs and playlist_name

        Output:
        A list of song data for a given list of URIs
        '''
        feature_list = sp.audio_features(uris)
        tracks = sp.tracks(uris)
        tracks = tracks["tracks"]
        artists = sp.artists([tracks[i]["artists"][0]["id"] for i in range(len(tracks))])["artists"]
        for i, features in enumerate(feature_list):
            #Artist of the track, for genres and popularity
            track = tracks[i]
            artist = track["artists"][0]["name"]
            artist_data = artists[i]
            artist_pop = artist_data["popularity"]
            artist_genres = artist_data["genres"]

            #Track popularity
            track_pop = track["popularity"]

            #Add in extra features
            try:
                features["artist_pop"] = artist_pop
            except:
                continue
            if artist_genres:
                features["genres"] = " ".join([re.sub(' ','_',i) for i in artist_genres])
            else:
                features["genres"] = "unknown"
            features["track_pop"] = track_pop
            features["playlist_name"] = playlist_name
            features["artist"] = artist
            features["name"] = track["name"]
            all_data.append(features)

    playlist = sp.playlist(uri)
    tracks = []
    for i in playlist["tracks"]["items"]:
        try:
            tracks.append(i["track"]["uri"])
        except:
            continue
    length = len(tracks)
    #Can only process 50 tracks at a time via API
    if length > 50:
        for i in range(0, int(length / 50)):
            subset = tracks[i * 50: (i + 1) * 50]
            try:
                tracks_to_features(subset, playlist_name)
            except:
                # Bad track in the data
                continue
    else:
        try:
            tracks_to_features(tracks, playlist_name)
        except:
            return all_data
    return all_data

def drop_duplicates(df):
    df['artists_song'] = df.apply(lambda row: row['name'] + row['artist'], axis = 1)
    return df.drop_duplicates('artists_song').reset_index(drop = True).drop("artists_song", axis = 1)
    
if __name__ == "__main__":
    #Authentication - without user
    client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
    sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

    #Define users to grab playlists from
    users = ["spotify", "digster.fm", "myplay.com", "topsify", "sanik007"]
    songs = playlist_grab(users)
    df_list = []
    for i in songs:
        df_list = df_list + playlist_extract(i[0], i[1])
    data = pd.DataFrame(df_list)
    data = drop_duplicates(data)
    data.to_csv('songs.csv', index = False)
