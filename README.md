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
2. Run `fileGeneration.py` to scrape song data from some of the top playlist producers on Spotify.
3. Run `dataTransform.py` and pass in an artist to draw recommendations from.

## Contributing
Contributions to improve the functionality or efficiency of the system are welcome. Please adhere to the project's code style and contribute via pull requests.

## Authors and Acknowledgment
- [Your Name] - Initial work
- Acknowledgments to any contributors or inspirations.

## License
This project is licensed under the [LICENSE] - see the file for details.
