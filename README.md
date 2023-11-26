# Spotify Recommendation System

## Introduction
This project provides a basic Spotify recommendation system. It leverages the Spotify APIs to gather data, process it using sentiment analysis and feature engineering, and finally applies machine learning techniques to recommend tracks based on user preferences.

## Features
- **Data Transformation**: Utilizes the `spotipy` library to interface with the Spotify API.
- **Sentiment Analysis**: Analyzes track names for subjectivity and polarity using the `TextBlob` library.
- **Feature Engineering**: One-Hot Encoding, TF-IDF Vectorization, and Min-Max Scaling are used to process the song corpus.
- **Recommendation System**: Generates track recommendations using cosine similarity.

## Prerequisites
- Python 3.x
- Libraries: `spotipy`, `pandas`, `sklearn`, `TextBlob`

## Installation
1. Clone the repository: `git clone [repository-link]`.
2. Install required Python packages: `pip install -r requirements.txt`.

## Usage
To use this recommendation system:
1. Provide your Spotify API credentials in the script via a .env file.
2. Run `python fileGeneration.py` to scrape song data from some of the top playlist producers on Spotify.
3. Run `python dataTransform.py artistname`to and pass in an artist to draw song recommendations from.
4. `Recommendations.csv` is generated with 40 sorted recommendations
## Authors and Acknowledgment
- This code is inspired by [a series on Medium.com](https://medium.com/@enjui.chang/enhance-your-playlists-with-machine-learning-spotify-automatic-playlist-continuation-2aae2c926e77)
- Noah Mitchem - Code implementation
![](http://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExbm54aWxtcDAybDl0cGVpenNuODltc3Q2d3JpdGkwYjMyYWExMDZzbiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/uDa8i4tHogwuppZpQA/giphy.gif)
