from pathlib import Path
import os
import textattack


from utils_amazon_reviews import load_amazon_review_data

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load the data into format: data = [('Today was....', 1), ('This movie is...', 0)]
data_file_path = PROJECT_ROOT / "data" / "sentiment" / "amazon_reviews" / "test_data.csv"
label_file_path = PROJECT_ROOT / "data" / "sentiment" / "amazon_reviews" / "test_data_labels.pt"
sentences, labels, labels_torch = load_amazon_review_data(data_path=data_file_path, label_path=label_file_path)
data = list(map(tuple, zip(sentences, labels)))

# Create a dataset for textattack
dataset = textattack.datasets.Dataset(data)

# Run CLI with
# textattack attack --model-from-huggingface nlptown/bert-base-multilingual-uncased-sentiment --dataset-from-file amazon_reviews_textattack_dataset.py
