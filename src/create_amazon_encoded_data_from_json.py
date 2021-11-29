from pathlib import Path
import os
from utils_amazon_reviews import encode_and_save_data, get_df

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = PROJECT_ROOT / "data" / "sentiment" / "amazon_reviews"

if __name__ == "__main__":
    # The amazon review data is found here:
    # See "Video Games, 5-core" here: http://jmcauley.ucsd.edu/data/amazon/index_2014.html
    # Direct download: http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Video_Games_5.json.gz
    amazon_review_data_path = PROJECT_ROOT / "data" / "sentiment" / "amazon_reviews" / "reviews_Video_Games_5.json.gz"

    # Convert to a dataframe
    review_data = get_df(amazon_review_data_path)

    # Save the data to a .csv
    csv_file_save_path = DATA_PATH / "all_data.csv"
    review_data.to_csv(str(csv_file_save_path), index=False, encoding="utf-8")

    # Encode the data and save as a .pt file
    bert_tokenizer_path = 'nlptown/bert-base-multilingual-uncased-sentiment'

    # Save training data
    idx_end, idx_list_data = encode_and_save_data(all_data_df=review_data, tokenizer_path=bert_tokenizer_path,
                                                  save_path=DATA_PATH, file_name="data",
                                                  idx_start=0, example_count=50_000, max_sequence_length=512)
    review_data_used = review_data.loc[review_data.index[idx_list_data]]
    review_data_used.to_csv(str(DATA_PATH / "data.csv"), index=True, encoding="utf-8")

    # Save validation data for testing
    idx_end_test, idx_list_test_data = encode_and_save_data(all_data_df=review_data, tokenizer_path=bert_tokenizer_path,
                                                            save_path=DATA_PATH, file_name="test_data",
                                                            idx_start=idx_end, example_count=10_000,
                                                            max_sequence_length=512)
    review_test_data_used = review_data.loc[review_data.index[idx_list_test_data]]
    review_test_data_used.to_csv(str(DATA_PATH / "test_data.csv"), index=True, encoding="utf-8")
