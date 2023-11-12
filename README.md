# Spotify Recommendation System

## Introduction
This project provides a sophisticated Spotify recommendation system. It leverages the Spotify API to gather data, processes it using sentiment analysis and feature engineering, and finally applies machine learning techniques to recommend tracks based on user preferences.

## Features
- **Data Transformation**: Utilizes the `spotipy` library to interface with the Spotify API.
- **Sentiment Analysis**: Analyzes track names for subjectivity and polarity using the `TextBlob` library.
- **Feature Engineering**: Employs techniques like One-Hot Encoding, TF-IDF Vectorization, and Min-Max Scaling to process Spotify data.
- **Recommendation System**: Generates track recommendations using cosine similarity and other metrics.

## Prerequisites
- Python 3.x
- Libraries: `spotipy`, `pandas`, `sklearn`, `TextBlob`

## Installation
1. Clone the repository: `git clone [repository-link]`.
2. Install required Python packages: `pip install -r requirements.txt`.

## Usage
To use this recommendation system:
1. Provide your Spotify API credentials in the script.
2. Run `dataTransform.py` to process the data and generate recommendations.

## Contributing
Contributions to improve the functionality or efficiency of the system are welcome. Please adhere to the project's code style and contribute via pull requests.

## Authors and Acknowledgment
- [Your Name] - Initial work
- Acknowledgments to any contributors or inspirations.

## License
This project is licensed under the [LICENSE] - see the file for details.
