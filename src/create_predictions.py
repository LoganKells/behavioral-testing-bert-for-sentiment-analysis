import os
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Union, List

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from utils_amazon_reviews import get_data_loader
from fine_tune import get_device

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_predictions(model, data_loader: DataLoader) -> List[str]:
    """
    This function will create a .txt file for the predictions
    :param model: pre-trained model
    :param data_loader: Pytorch data loader
    :return: List[str] of prediction lines. E.g. 4 0.17158113 0.12090003 0.17976387 0.1629968 0.36475816
    """
    prediction_lines = []
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    for data, label in data_loader:
        data, label = data.to(device), label.to(device)

        # Run the data through the model to generate prediction classes and probabilities
        output = model(data)

        # Generate probability distribution over the classes (5 classes)
        logits = output.logits
        probabilities = torch.exp(logits.detach())
        probabilities = torch.softmax(probabilities, dim=1)
        label_prediction = torch.argmax(probabilities, dim=-1)

        # Create a line in format: Prediction Probability-per-class
        # e.g. "3 0.3 0.5 0.1 0.05 0.05"
        softmax_list = []
        probabilities = probabilities.cpu().numpy()
        for p in probabilities[0]:
            softmax_list.append(str(p))
        probability_line = " ".join(softmax_list)
        line = f"{label_prediction.item()} {probability_line}\n"
        prediction_lines.append(line)
    return prediction_lines


def save_predictions(predictions: List[str], file_path: Union[str, PurePath], file_name: str) -> None:
    """
    This function will save the given predictions to a .txt file
    :param predictions: List[str] of prediction lines. E.g. 4 0.17158113 0.12090003 0.17976387 0.1629968 0.36475816
    :param file_path: Path write the predictions file .txt
    :param file_name: File name for the .txt file
    :return: None
    """
    # Convert to a PurePath
    file_path = Path(file_path) if isinstance(file_path, str) else file_path

    # Make the directory
    if not os.path.isdir(file_path):
        os.mkdir(file_path)

    # Write to a file
    file_write_path = file_path / file_name
    with open(file_write_path, 'w') as f:
        f.writelines(predictions)


def save_metadata(model_path: Union[str, PurePath], prediction_path: Union[str, PurePath],
                  data_path: Union[str, PurePath], max_sequence_length: int) -> None:
    """
    This function will save the relevant metadata information to understand what generated the predictions.
    :param model_path:
    :param prediction_path: The metadata.txt will be saved here
    :param data_path:
    :param max_sequence_length:
    :return:
    """
    # Convert to a PurePath
    model_path = Path(model_path) if isinstance(model_path, str) else model_path
    prediction_path = Path(prediction_path) if isinstance(prediction_path, str) else prediction_path
    data_path = Path(data_path) if isinstance(data_path, str) else data_path

    # Make the directory
    if not os.path.isdir(prediction_path):
        os.mkdir(prediction_path)

    # Write to a file
    file_write_path = prediction_path / "metadata.txt"
    with open(file_write_path, 'w') as f:
        f.write(f"Model source: {str(model_path)}\n")
        f.write(f"Data source: {str(data_path)}\n")
        f.write(f"Max sequence length of each sentence: {max_sequence_length}")


@dataclass
class RunningParameters:
    model_path: str = None
    num_works: int = 6
    device: str = "cpu"
    batch_size: int = 1
    max_sequence_length: int = 75
    model_selection: tuple = (1, 2)  # Tuple selection related to get_model_selection(selection)
    data_path = str(PROJECT_ROOT / "data" / "sentiment" / "amazon_reviews" / "test_data_encoded.pt")
    label_path = str(PROJECT_ROOT / "data" / "sentiment" / "amazon_reviews" / "test_data_labels.pt")


def get_model_selection(selection: int) -> dict:
    models_path = PROJECT_ROOT / "models" / "sentiment"
    # Each tuple is: (model_path, prediction_path)
    option_1 = {"model_short_name": "bert_multilingual",
                "model_path": 'nlptown/bert-base-multilingual-uncased-sentiment',
                "prediction_path": "bert_pre_trained"}
    option_2 = {"model_short_name": "bert_multilingual",
                "model_path": str(models_path / "bert_multilingual_amazon_reviews_hugging"),
                "prediction_path": "bert_trained"}
    options = {"1": option_1, "2": option_2}
    return options[str(selection)]


def load_model(model_path: str, device):
    if ".pt" in model_path:
        model = torch.load(model_path, map_location=device)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model = model.to(device)
    model = model.eval()
    return model


if __name__ == '__main__':
    # Parameters for running
    pars = RunningParameters
    pars.device = get_device()
    pars.model_selection = (1, 2)  # Configure which models to run.

    # Get the model to use
    for selection in pars.model_selection:
        model_metadata = get_model_selection(selection=selection)
        model_path, prediction_path = model_metadata["model_path"], model_metadata["prediction_path"]
        model_short_name = model_metadata["model_short_name"]
        model = load_model(model_path, pars.device)

        # Get the data to use with the model
        data_loader = get_data_loader(pars.data_path, pars.label_path,
                                      shuffle=False, num_workers=pars.num_works,
                                      batch_size=pars.batch_size, max_sequence_length=pars.max_sequence_length)

        # Run the model to generate prediction
        print(f"Running model: {model_path}")
        prediction_lines = create_predictions(model, data_loader)

        # Save the predictions to a .txt
        prediction_file_path = PROJECT_ROOT / "predictions" / "sentiment" / "amazon_reviews" / prediction_path
        save_predictions(prediction_lines, prediction_file_path, file_name=f"{model_short_name}.txt")
        save_metadata(model_path, prediction_file_path, pars.data_path, pars.max_sequence_length)

        print(f"Predictions saved to: {prediction_file_path}")
