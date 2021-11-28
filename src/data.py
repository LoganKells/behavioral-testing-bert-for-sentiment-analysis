from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import gzip
# import simplejson as json
import json
from typing import Tuple, Dict, Optional

TOKENIZER_PATH = 'nlptown/bert-base-multilingual-uncased-sentiment'


def parse(path: str):
    """
    code attribution required:
    Image-based recommendations on styles and substitutes
    J. McAuley, C. Targett, J. Shi, A. van den Hengel
    SIGIR, 2015
    """
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)


def get_df(path: str) -> pd.DataFrame:
    """
    code attribution required:
    Image-based recommendations on styles and substitutes
    J. McAuley, C. Targett, J. Shi, A. van den Hengel
    SIGIR, 2015
    """
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def save_data(df: pd.DataFrame, tokenizer_path: str):
    all_data = {}
    all_labels = {}
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    ex_col = 'reviewText'
    gold_col = 'overall'
    NUM_EXAMPLES = 50000

    for idx in range(NUM_EXAMPLES):
        print(idx)
        example = "ex-" + str(idx)
        review = df[ex_col].iloc[idx]

        if pd.isna(review):
            continue

        token = tokenizer.encode(review, return_tensors='pt')

        if len(token[0]) > 512:
            continue

        label = int(df[gold_col].iloc[idx]) - 1
        data = token[0]
        if len(token[0]) < 512:
            delta = 512 - len(token[0])
            zeros = torch.zeros(delta, dtype=torch.int64)
            data = torch.cat((data, zeros))

        label = torch.tensor(label, dtype=torch.int64)
        all_data[example] = data
        all_labels[example] = label

    torch.save(all_data, 'data_padded.pt')
    torch.save(all_labels, 'labels.pt')


def save_test_data(df: pd.DataFrame, tokenizer_path: str):
    all_data = {}
    all_labels = {}
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    ex_col = 'reviewText'
    gold_col = 'overall'
    NUM_EXAMPLES = 50000
    EXAMPLES_END = 60000

    for idx in range(NUM_EXAMPLES, EXAMPLES_END):
        print(idx)
        example = "ex-" + str(idx)
        review = df[ex_col].iloc[idx]

        if pd.isna(review):
            continue

        token = tokenizer.encode(review, return_tensors='pt')

        if len(token[0]) > 512:
            continue

        label = int(df[gold_col].iloc[idx]) - 1
        data = token[0]
        if len(token[0]) < 512:
            delta = 512 - len(token[0])
            zeros = torch.zeros(delta, dtype=torch.int64)
            data = torch.cat((data, zeros))

        label = torch.tensor(label, dtype=torch.int64)
        all_data[example] = data
        all_labels[example] = label

    torch.save(all_data, 'test-data_padded.pt')
    torch.save(all_labels, 'test_labels.pt')


def load_data(data_path: str, label_path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    data_dict = torch.load(data_path)
    labels_dict = torch.load(label_path)

    return data_dict, labels_dict


class AmazonReviewDataSet(Dataset):
    def __init__(self, data_path: str, label_path: str, max_sequence_length: Optional[int]):
        self.labels = []

        # Load the data
        data_tensors = []
        data_dict, label_dict = load_data(data_path, label_path)
        for data_tensor in data_dict.values():

            if max_sequence_length:  # Trim the length
                data_tensor = data_tensor[:max_sequence_length] if len(data_tensor) > max_sequence_length else data_tensor

            data_tensors.append(data_tensor)
        for label_tensor in label_dict.values():
            self.labels.append(label_tensor)

        # Pad the data
        self.data_padded = torch.nn.utils.rnn.pad_sequence(sequences=data_tensors, batch_first=True,
                                                           padding_value=0.0)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data_padded[idx], self.labels[idx]


def get_data_loader(data_path: str, label_path: str, shuffle=True, num_workers=1, batch_size=1, max_sequence_length=50) -> DataLoader:
    """

    :param data_path:
    :param label_path:
    :param num_workers:
    :param batch_size:
    :param max_sequence_length:
    :return:
    """
    dataset = AmazonReviewDataSet(data_path, label_path, max_sequence_length)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def main():
    path = '/Users/nathancastaneda/Desktop/Video_Games.json.gz'
    tokenizer_path = 'nlptown/bert-base-multilingual-uncased-sentiment'
    df = get_df(path)

    # save_data(df, tokenizer_path)
    # save_test_data(df, tokenizer_path)



