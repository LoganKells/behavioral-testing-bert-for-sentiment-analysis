import dataclasses
import json
import os
from pathlib import Path, PurePath
from typing import Union

import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_scheduler
from datasets import load_metric
from tqdm.auto import tqdm
from data import AmazonReviewDataSet

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BERT_MODEL_PATH = 'nlptown/bert-base-multilingual-uncased-sentiment'


def get_model(model_path: str):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model


def get_optimizer(model, lr: float):
    optimizer = AdamW(model.parameters(), lr=lr)
    return optimizer


def train_model(model, optimizer, train_data_loader, num_epochs, device):
    """
    This function will re-train a model to optimize the weights on the dataset.
    :param model:
    :param optimizer:
    :param train_data_loader:
    :param num_epochs:
    :param device:
    :return: trained model
    """
    epochs = num_epochs
    num_training_steps = epochs * len(train_data_loader)
    decay = 'linear'
    lr_scheduler = get_scheduler(decay, optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)


    model.to(device)

    progress_bar = tqdm(range(num_training_steps))

    model.train()

    for epoch in range(epochs):
        for data, label in train_data_loader:
            data, label = data.to(device), label.to(device)
            outputs = model(data, labels=label)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    return model


def eval_model(model, test_data_loader):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    metric = load_metric("accuracy")
    model.eval()

    for batch in test_data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    metric.compute()


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


@dataclass
class TrainingParameters:
    learning_rate: float = 1e-5
    epochs: int = 1
    num_works: int = 6
    batch_size: int = 32
    device: str = "cpu"
    max_sequence_length: int = 50
    model_save_path: PurePath = PROJECT_ROOT / "models" / "sentiment" / "bert_multilingual_amazon_reviews_hugging"
    data_path: PurePath = PROJECT_ROOT / "data" / "sentiment" / "amazon_reviews" / "data.pt"
    label_path: PurePath = PROJECT_ROOT / "data" / "sentiment" / "amazon_reviews" / "labels.pt"


def save_training_parameters(file_path: Union[str, PurePath], parameters: TrainingParameters) -> None:
    """
    This function will save the TrainingParameters used when the model was trained.
    :param file_path: file path the model will be saved to
    :param parameters: TrainingParameters used to train the model
    :return: None
    """
    if not os.path.isdir(file_path):
        os.mkdir(file_path)

    parameters.device = str(parameters.device) if not isinstance(parameters.device, str) else parameters.device
    parameters.model_save_path = str(parameters.model_save_path) if not isinstance(parameters.model_save_path, str) \
        else parameters.model_save_path
    parameters.data_path = str(parameters.data_path) if not isinstance(parameters.data_path, str) \
        else parameters.data_path
    parameters.label_path = str(parameters.label_path) if not isinstance(parameters.label_path, str) \
        else parameters.label_path

    json_data = json.dumps(dataclasses.asdict(parameters))
    json_file_path = file_path / "metadata.json"
    with open(json_file_path, 'w') as f:
        json.dump(json_data, f)


if __name__ == "__main__":
    # Training parameters
    pars = TrainingParameters()
    pars.device = get_device()
    save_training_parameters(file_path=pars.model_save_path, parameters=dataclasses.replace(pars))  # Save the parameters to a .json

    # Create a dataset
    training_dataset = AmazonReviewDataSet(data_path=str(pars.data_path),
                                           label_path=str(pars.label_path),
                                           max_sequence_length=pars.max_sequence_length)
    training_dataloader = DataLoader(training_dataset, num_workers=pars.num_works,
                                     batch_size=pars.batch_size, shuffle=True, drop_last=False)

    # model and optimizer
    model = get_model(model_path=BERT_MODEL_PATH)
    optimizer = get_optimizer(model, lr=1e-5)

    # Tune the model
    model = train_model(model, optimizer, training_dataloader, num_epochs=pars.epochs, device=pars.device)

    # Save the model
    try:
        # Save hugging face model
        # see - https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.save_pretrained
        model.save_pretrained(pars.model_save_path)
    except Exception as e:
        print(e)
    try:
        torch_weights_file_path = pars.model_save_path / "pytorch_model.pt"
        torch.save(model.state_dict(), f=torch_weights_file_path)
    except Exception as e:
        print(e)

    # Evaluate accuracy
    # eval_model(model, training_dataloader)