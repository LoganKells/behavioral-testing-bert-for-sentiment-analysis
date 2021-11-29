from typing import Union
from pathlib import PurePath
import numpy as np
import pandas as pd


def load_airline_tweets_data(data_path: Union[str, PurePath]) -> tuple:
    # Load the data
    data = pd.read_csv(data_path)
    labels = data['airline_sentiment'].to_list()
    confs = data['airline_sentiment_confidence'].to_list()
    airlines = data['airline'].to_list()
    sentences = data['text'].to_list()
    reasons = data['negativereason'].to_list()

    # Encode the labels
    mapping = {'negative': 0, 'positive': 2, 'neutral': 1}
    labels_encoded = np.array([mapping[x] for x in labels]).astype(int)

    return labels_encoded, confs, airlines, sentences, reasons