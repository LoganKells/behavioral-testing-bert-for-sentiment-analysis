from pathlib import PurePath, Path
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import gzip
import os
import json
from typing import Tuple, Optional, Union, List
import spacy

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_df(path: Union[str, PurePath]) -> pd.DataFrame:
    """
    This function will return a dataframe from a file.
    :param path: Path to the .json.gz file.
    code attribution required:
    Image-based recommendations on styles and substitutes
    J. McAuley, C. Targett, J. Shi, A. van den Hengel
    SIGIR, 2015
    """
    def parse(path: Union[str, PurePath]):
        """
        This function will load from a .json.gz compressed archive file.
        :param path: Path to the .json.gz file.

        code attribution required:
        Image-based recommendations on styles and substitutes
        J. McAuley, C. Targett, J. Shi, A. van den Hengel
        SIGIR, 2015
        """
        g = gzip.open(path, 'rb')
        for l in g:
            yield json.loads(l)
    i = 0
    data_dict = {}
    for sentence in parse(path):
        data_dict[i] = sentence
        i += 1

    # Create a dataframe
    data_df = pd.DataFrame.from_dict(data_dict, orient='index')

    # Change the labels to be in range(0,4) not (1,5)
    data_df['overall'] = data_df['overall'] - 1
    return data_df


def encode_and_save_data(all_data_df: pd.DataFrame, tokenizer_path: str, file_name: str,
                         save_path: Union[str, PurePath],
                         idx_start: int, example_count: int, padding: bool = False,
                         max_sequence_length: int = 512) -> Tuple[int, List[int]]:
    """
    This function will encode the data sequences using the tokenizer.
    :param all_data_df: Data stored in a df
    :param tokenizer_path: tokenizer for encoding
    :param file_name: File name to save the .pt files
    :param save_path: Path to save the files to.
    :param idx_start: Index of the first example to load
    :param example_count: Total number of examples to encode and save
    :param padding: Choose to pad the sequences if True.
    :param max_sequence_length: Maximum length of sequences to use
    :return: Tuple of final index and a list of indices selected
    """
    all_data = []
    all_labels = []
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    ex_col = 'reviewText'
    gold_col = 'overall'

    example_created_count, idx, idx_list = 0, idx_start, []

    while example_created_count < example_count:
        review = all_data_df[ex_col].iloc[idx]

        # Skip any reviews without review text sequence
        if pd.isna(review):
            continue

        token = tokenizer.encode(review, return_tensors='pt')

        data = token[0]
        if len(data) > max_sequence_length:  # Skip any sequences greater than the max sequence length
            idx += 1
            continue
        elif padding and len(data) < max_sequence_length:
            delta = max_sequence_length - len(data)
            zeros = torch.zeros(delta, dtype=torch.int64)
            data = torch.cat((data, zeros))

        label = int(all_data_df[gold_col].iloc[idx])
        label = torch.tensor(label, dtype=torch.int64)
        all_data.append(data)
        all_labels.append(label)
        idx_list.append(idx)
        idx += 1
        example_created_count += 1

    # Save the data
    save_path = Path(save_path) if isinstance(save_path, str) else save_path
    if padding:
        torch.save(all_data, save_path / f'{file_name}_encoded_padded.pt')
    else:
        torch.save(all_data, save_path / f'{file_name}_encoded.pt')
    torch.save(all_labels, save_path / f'{file_name}_labels.pt')
    return idx, idx_list


def convert_txt_to_encoded_pt(data_path: PurePath, file_name: str):
    """
    This function will convert examples in a .txt to a .pt file that can be loaded with the AmazoneReviewDataSet class.
    :param data_path: Path to the .txt examples
    :param file_name: Name of the .txt file
    :return: None
    """
    # Read the data
    with open(data_path / file_name) as f:
        data = f.readlines()

    # Encode the data
    bert_tokenizer_path = 'nlptown/bert-base-multilingual-uncased-sentiment'
    tokenizer = AutoTokenizer.from_pretrained(bert_tokenizer_path)
    encoded_data = []
    labels = []
    for example in data:
        token = tokenizer.encode(example, return_tensors='pt')
        label = 0  # Use 0, this data is not used for training
        encoded_data.append(token[0])
        labels.append(label)

    # Write the data to a .pt Pytorch file
    torch.save(obj=encoded_data, f=data_path / "data.pt")


class AmazonReviewDataSet(Dataset):
    def __init__(self, data_path: str, label_path: str, max_sequence_length: Optional[int]):
        self.labels = []

        # Load the data
        data_tensors = []
        data_values, label_values = torch.load(data_path), torch.load(label_path)
        for data_tensor in data_values:

            if max_sequence_length:  # Trim the length
                data_tensor = data_tensor[:max_sequence_length] if len(data_tensor) > max_sequence_length else data_tensor

            data_tensors.append(data_tensor)
        for label_tensor in label_values:
            self.labels.append(label_tensor)

        # Pad the data
        self.data_padded = torch.nn.utils.rnn.pad_sequence(sequences=data_tensors, batch_first=True,
                                                           padding_value=0.0)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data_padded[idx], self.labels[idx]


def get_data_loader(data_path: Union[str, PurePath], label_path: Union[str, PurePath],
                    shuffle=True, num_workers=0, batch_size=1, max_sequence_length=50) -> DataLoader:
    """
    This function will return a DataLoader, useful for batching in PyTorch.
    :param data_path:
    :param label_path:
    :param shuffle: True for training, False for other purposes. This will shuffle the data when it is retrieved.
    :param num_workers:
    :param batch_size:
    :param max_sequence_length: Each sequence (sentence) will be truncated and padded to this length.
    :return:
    """
    dataset = AmazonReviewDataSet(data_path, label_path, max_sequence_length)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def load_amazon_review_data(data_path: Union[str, PurePath], label_path: Union[str, PurePath]):
    """
    This function will load the dataset specified in data_path for amazon reviews from a .csv.
    :param data_path: Path to the .csv review data.
    :param label_path: Path to the pytorch .pt label data.
    :return:
    """
    # Load the amazon reviews from CSV
    data = pd.read_csv(data_path)
    sentences = data["reviewText"].to_list()
    labels = data["overall"].to_list()

    # Load the labels from pytorch .pt file
    labels_torch = torch.load(f=label_path)
    # labels_torch = labels_torch.numpy().tolist()

    return sentences, labels, labels_torch


def parse_data(sentences: List[str]):
    nlp = spacy.load('en_core_web_sm')
    return list(nlp.pipe(sentences))


