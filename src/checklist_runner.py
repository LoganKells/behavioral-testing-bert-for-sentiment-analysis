import argparse
from dataclasses import dataclass
from typing import List, Union, Tuple
import os
from pathlib import Path, PurePath

from checklist.editor import Editor
from checklist.test_suite import TestSuite
from utils_airline_tweets import load_airline_tweets_data
from utils_amazon_reviews import load_amazon_review_data, convert_txt_to_encoded_pt, get_data_loader
from create_predictions import create_predictions, load_model, get_device, save_predictions, save_metadata
from mft_test_suites import create_mft_test_negation
from invariance_test_suites import create_invariance_test_change_neutral_words
from direction_test_suites import create_directional_expression_test_add_positive_phrases, \
    create_directional_expression_test_add_negative_phrases

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_parameters(args):
    data_selection = args.data_selection

    # Running Parameters
    if data_selection == "airline_tweets":
        parameters = RunningParametersAirlineTweets()
    elif data_selection == "amazon_reviews":
        parameters = RunningParametersAmazonReviews()
    else:
        print("Please select from: \"airline_tweets\" or \"amazon_reviews\"")
        parameters = None
    parameters.device = get_device()

    # Check for errors in parameters
    if ".pkl" not in str(parameters.suite_file_name):
        raise ValueError("suite_save_pkl_path must contain .pkl file extension")
    if ".txt" not in str(parameters.prediction_file_path):
        raise ValueError("prediction_file_path must contain a .txt file extension")

    return parameters


def load_editor(editor: Editor, suite_name: str) -> Tuple[Editor, dict]:
    """
    This function will load predifined lexicon to the Editor. This lexicon is used when creating tests,
    such as in create_invariance_test_change_neutral_words().
    :param editor: Editor to load lexicon to.
    :param suite_name: Define the test suite name, which is used to organize the data, lexicon, etc.
    :return: Editor, used for text generation
    """
    # Text generator
    editor.tg

    pos_adj = ['good', 'great', 'excellent', 'amazing', 'extraordinary', 'beautiful', 'fantastic', 'nice', 'incredible',
               'exceptional', 'awesome', 'perfect', 'fun', 'happy', 'adorable', 'brilliant', 'exciting', 'sweet',
               'wonderful', 'enjoyable', 'exciting', 'amazing', 'engaging']
    neg_adj = ['awful', 'bad', 'horrible', 'weird', 'rough', 'lousy', 'unhappy', 'average', 'difficult', 'poor', 'sad',
               'frustrating', 'hard', 'lame', 'nasty', 'annoying', 'boring', 'creepy', 'dreadful', 'ridiculous',
               'terrible', 'ugly', 'unpleasant']
    neutral_adj = ['American', 'international', 'commercial', 'British', 'private', 'Italian', 'Indian', 'Australian',
                   'Israeli', ]
    pos_verb_present = ['like', 'enjoy', 'appreciate', 'love', 'recommend', 'admire', 'value', 'welcome']
    neg_verb_present = ['hate', 'dislike', 'regret', 'abhor', 'dread', 'despise']
    neutral_verb_present = ['see', 'find']
    pos_verb_past = ['liked', 'enjoyed', 'appreciated', 'loved', 'admired', 'valued', 'welcomed']
    neg_verb_past = ['hated', 'disliked', 'regretted', 'abhorred', 'dreaded', 'despised']
    neutral_verb_past = ['saw', 'found']
    neutral_words = {'.', 'the', 'The', ',', 'a', 'A', 'and', 'of', 'to', 'it', 'that', 'in', 'this', 'for', 'you',
                     'there', 'or', 'an', 'by', 'about', 'flight', 'my', 'in', 'of', 'have', 'with', 'was', 'at', 'it',
                     'get', 'from', 'this', 'Flight', 'plane'}

    lexicon = {'pos_adj': pos_adj, 'neg_adj': neg_adj, 'neutral_adj': neutral_adj,
               'pos_verb_present': pos_verb_present, 'neg_verb_present': neg_verb_present,
               'neutral_verb_present': neutral_verb_present, 'pos_verb_past': pos_verb_past,
               'neg_verb_past': neg_verb_past, 'neutral_verb_past': neutral_verb_past,
               'neutral_words': neutral_words}

    # Add data to the Editor
    editor.add_lexicon('pos_adj', pos_adj, overwrite=True)
    editor.add_lexicon('neg_adj', neg_adj, overwrite=True)
    editor.add_lexicon('neutral_adj', neutral_adj, overwrite=True)
    editor.add_lexicon('pos_verb_present', pos_verb_present, overwrite=True)
    editor.add_lexicon('neg_verb_present', neg_verb_present, overwrite=True)
    editor.add_lexicon('neutral_verb_present', neutral_verb_present, overwrite=True)
    editor.add_lexicon('pos_verb_past', pos_verb_past, overwrite=True)
    editor.add_lexicon('neg_verb_past', neg_verb_past, overwrite=True)
    editor.add_lexicon('neutral_verb_past', neutral_verb_past, overwrite=True)
    editor.add_lexicon('pos_verb', pos_verb_present + pos_verb_past, overwrite=True)
    editor.add_lexicon('neg_verb', neg_verb_present + neg_verb_past, overwrite=True)
    editor.add_lexicon('neutral_verb', neutral_verb_present + neutral_verb_past, overwrite=True)

    # Customize lexicon to the data set used
    if suite_name == "airline_tweets":
        # Pre-defined lexicon
        air_noun = ['flight', 'seat', 'pilot', 'staff', 'service', 'customer service', 'aircraft', 'plane', 'food',
                    'cabin crew', 'company', 'airline', 'crew']
        editor.add_lexicon('air_noun', air_noun)
        lexicon['air_noun'] = air_noun
    elif suite_name == "amazon_reviews":
        review_noun = ['service', 'staff', 'delivery', 'driver', 'food', 'company', 'customer']
        editor.add_lexicon('review_noun', review_noun)
        lexicon['review_noun'] = review_noun

    return editor, lexicon


def save_build_suite(suite: TestSuite, save_path: Union[str, PurePath],
                     file_name: str, samples: int) -> None:
    """
    This wrapper will save the TestSuite while also building for analysis.
    :param suite: TestSuite to build and save
    :param save_path: Path-like object to save the .pkl file locally.
    :param file_name: .pkl file name to save the test suite
    :param samples: Number of samples to draw from for each suite.
    :return: None
    """
    # Make directory if missing
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    # Save the perturbed examples to a .txt and to a .pt file to be used in a model.
    temp_file_path = save_path / "data.txt"
    suite.to_raw_file(str(temp_file_path), n=samples, seed=1)  # Save .txt
    convert_txt_to_encoded_pt(data_path=save_path, file_name="data.txt")  # Save .pt

    # Build the TestSuite
    for test in suite.tests:
        suite.tests[test].name = test
        suite.tests[test].description = suite.info[test]['description]']
        suite.tests[test].capability = suite.info[test]['capability']

    # Save the TestSuite to a local .pkl file
    suite.save(str(save_path / file_name))  # save .pkl

    # Clean up temp file
    # os.remove(temp_file_path)


def run_analysis(suite_path: Union[str, PurePath], pred_path: Union[str, PurePath]) -> None:
    """
    Run the checklist provided in the TestSuite. The TestSuite contains the
    :param suite_path: Path-like object locating the suite.pkl file.
    :param pred_path: Path to the predictions produced by the model.
    Format each sentence prediction as: "Pred, prob-class1, prob-class2, prob-class3, ... prob-classN" where
    the Pred value is the most probable.
    :return: None
    """
    # Load the TestSuite from .pkl file
    suite = TestSuite.from_file(suite_path)

    # Run the analysis
    suite.run_from_file(pred_path, overwrite=True, file_format="pred_and_softmax")
    suite.summary()


def create_suite_file_paths(root_path: PurePath, keys: Tuple[str, ...]) -> dict:
    """
    This function will build a dictionary of PurePath to the test suites
    :return: dictionary of paths, keys are suite names matching the file folder.
    """
    test_suite_paths = {}
    for name in keys:
        test_suite_paths[name] = root_path / name
    return test_suite_paths


def build_all_test_suites(sentences: List[str], lexicon, editor: Editor, save_path: PurePath,
                          test_suite_names: Tuple[str, ...]) -> tuple:
    """
    If this function is run, it will create and build all the suites in test_suite_names.
    :param sentences: List of sentence examples
    :param lexicon: lexicon loaded to the Editor()
    :param editor: Editor() that is loaded with lexicon
    :param test_suite_names: List[str] of all the test suite names to build.
    :return: tuple of (dict of TestSuites, dict of paths to test suite folders)
    """
    """
    In order to guide test ideation, it's useful to think of CheckList as a matrix of Capabilities x Test Types.
    *Capabilities* refers to general-purpose linguistic capabilities, which manifest in one way or another in almost any NLP application.
    We suggest that anyone CheckListing a model go through *at least* the following capabilities, trying to create MFTs, INVs, and DIRs for each if possible.
    1. **Vocabulary + POS:** important words or groups of words (by part-of-speech) for the task
    2. **Taxonomy**: synonyms, antonyms, word categories, etc
    3. **Robustness**: to typos, irrelevant additions, contractions, etc
    4. **Named Entity Recognition (NER)**: person names, locations, numbers, etc
    5. **Fairness**
    6. **Temporal understanding**: understanding order of events and how they impact the task
    7. **Negation**
    8. **Coreference**
    9. **Semantic Role Labeling (SRL)**: understanding roles such as agent, object, passive/active, etc
    10. **Logic**: symmetry, consistency, conjunctions, disjunctions, etc
    """
    tests = {}

    ### TEST 1 ###
    # Test type: Invariance - Invariance test (INV) is when we apply label-preserving perturbations to inputs and
    #                         expect the model prediction to remain the same.
    # Capability: Vocabulary (neutral words changed)
    test_name_invariance_neutral_words = "invariance_neutral_words"
    n = 5
    if test_name_invariance_neutral_words in test_suite_names:
        print(f"Creating test: {test_name_invariance_neutral_words}")
        inv_neutral_suite = create_invariance_test_change_neutral_words(TestSuite(), sentences, lexicon,
                                                                        editor=editor, example_count=n)
        tests[test_name_invariance_neutral_words] = {"suite": inv_neutral_suite, "samples": n}

    ### TEST 2 ###
    test_name_positive_phrases = "directional_positive_phrases"
    n = 5
    if test_name_positive_phrases in test_suite_names:
        print(f"Creating test: {test_name_positive_phrases}")
        pos_suite = create_directional_expression_test_add_positive_phrases(TestSuite(), sentences,
                                                                            editor=editor, example_count=n)
        tests[test_name_positive_phrases] = {"suite": pos_suite, "samples": n}

    ### TEST 3 ###
    test_name_negative_phrases = "directional_negative_phrases"
    n = 5
    if test_name_negative_phrases in test_suite_names:
        print(f"Creating test: {test_name_negative_phrases}")
        neg_suite = create_directional_expression_test_add_negative_phrases(TestSuite(), sentences,
                                                                            editor=editor, example_count=n)
        tests[test_name_negative_phrases] = {"suite": neg_suite, "samples": n}

    ### Test 4 ###
    # TODO finish this test
    # Test type: Minimum Functionality Test (MFT) - used to verify the model has specific capabilities.
    # Capability: Vocabulary, Can the model handle negation?
    test_name_mft_negation = "mft_negation"
    n = 5
    if test_name_mft_negation in test_suite_names:
        mft_negation_suite = create_mft_test_negation(TestSuite(), sentences, lexicon, editor=editor, example_count=n)
        tests[test_name_mft_negation] = {"suite": mft_negation_suite, "samples": n}

    # Make directory if missing
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    # Save and build any suites added to the dictionary.
    for test_name, suite_dict in tests.items():
        suite = suite_dict["suite"]
        n = suite_dict["samples"]
        save_build_suite(suite, save_path=save_path / test_name,
                         file_name="test.pkl", samples=n)

    # Define all the save paths
    paths = create_suite_file_paths(root_path=save_path, keys=test_suite_names)

    return tests, paths


def run_all_test_suites(model_path: PurePath, test_suite_paths: dict, device, max_sequence_length: int) -> None:
    """
    This function will run all the test suites saved within the test_paths dict
    :param model_path: Path to the model
    :param test_suite_paths: dictionary of folder paths for each test suite.
    :param device: Pytorch device, such as cpu or gpu.
    :param max_sequence_length: Maximum sequence length to trim the sentence examples when creating a dataloader.
    :return: None
    """

    # Generate a prediction file for each test
    for test_name, path_to_test_suite in test_suite_paths.items():
        print(f'Running test: {test_name}')
        # Create a dataloader (NOTE: We're passing the data into the labels as well, because these are not used) to
        # make predictions
        path_to_test_suite = Path(path_to_test_suite)
        data_path = path_to_test_suite / "data.pt"
        dataloader = get_data_loader(data_path=data_path, label_path=data_path,
                                     shuffle=False, max_sequence_length=max_sequence_length)

        # Run the new examples through the model
        model = load_model(model_path=str(model_path), device=device)
        prediction_lines = create_predictions(model, data_loader=dataloader)
        save_predictions(prediction_lines, path_to_test_suite, file_name="predictions.txt")
        save_metadata(model_path, path_to_test_suite, data_path, max_sequence_length)

    # Run the analysis
    for test_name, path_to_test_suite in test_suite_paths.items():
        path_to_test_suite = Path(path_to_test_suite)
        prediction_path = path_to_test_suite / "predictions.txt"
        test_suite_path = path_to_test_suite / "test.pkl"
        run_analysis(suite_path=test_suite_path, pred_path=prediction_path)


def _parse_args() -> argparse.Namespace:
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Create TestSuites and run analysis given training data and predictions.")
    parser.add_argument("--data_selection", required=True, type=str, default="airline_tweets",
                        choices=["airline_tweets", "amazon_reviews"],
                        help="This selection will configure all necessary parameters to generate the TestSuite.")
    return parser.parse_args()


@dataclass
class RunningParametersAirlineTweets:
    run_tests_from_pkl: bool = False
    data_file_path: PurePath = PROJECT_ROOT / "data" / "sentiment" / "airline_tweets" / "Tweets.csv"
    label_file_path: PurePath = None
    prediction_file_path: PurePath = PROJECT_ROOT / "predictions" / "sentiment" / "airline_tweets" / "bert.txt"
    suite_save_root: PurePath = PROJECT_ROOT / "test_suites" / "sentiment"
    suite_file_name: str = "test_suite_sentiment_airline_tweets.pkl"


@dataclass
class RunningParametersAmazonReviews:
    run_tests_from_pkl: bool = False
    data_file_path: PurePath = PROJECT_ROOT / "data" / "sentiment" / "amazon_reviews" / "test_data_1exs.csv"
    label_file_path: PurePath = PROJECT_ROOT / "data" / "sentiment" / "amazon_reviews" / "test_data_labels.pt"
    prediction_file_path: PurePath = PROJECT_ROOT / "predictions" / "sentiment" / "amazon_reviews" / "bert_trained" / "bert_multilingual.txt"
    suite_save_root: PurePath = PROJECT_ROOT / "test_suites" / "sentiment_testing"
    suite_file_name: str = "test_suite_sentiment_amazon_reviews.pkl"
    max_sequence_length: int = 75
    model_path: PurePath = PROJECT_ROOT / "models" / "sentiment" / "bert_multilingual_amazon_reviews_hugging"
    device: str = "cpu"
    # Choose the tests to run. This can take a long time, depending on the example count in the TestSuite.
    test_names = ("invariance_neutral_words", "directional_positive_phrases", "directional_negative_phrases") # "mft_negation"
    # Choose to rebuild all of the test_names test suites. This can take a long time, depending on the TestSuite.
    rebuild_test_suites: bool = True


if __name__ == "__main__":
    # Arguments
    args = _parse_args()

    # Check the parameters for valid values
    pars = get_parameters(args)

    # Load the data
    if isinstance(pars, RunningParametersAirlineTweets):
        data = load_airline_tweets_data(pars.data_file_path)
        labels, confs, airlines, sentences, reasons = data  # unpack
        parsed_data = sentences
    elif isinstance(pars, RunningParametersAmazonReviews):
        sentences, labels, labels_torch = load_amazon_review_data(data_path=pars.data_file_path,
                                                                  label_path=pars.label_file_path)
    else:
        raise ValueError(f"Parameters of type {type(pars)} not of correct type.")

    # Build test suites
    if pars.rebuild_test_suites:
        # Create an Editor
        print("Loading Editor")
        editor = Editor()
        editor, lexicon = load_editor(editor, suite_name=pars.suite_file_name)

        # Rebuild the test suites
        print("Rebuilding test suites...")
        test_suites, test_suite_paths = build_all_test_suites(sentences, lexicon, editor, save_path=pars.suite_save_root,
                                                              test_suite_names=pars.test_names)
    else:
        # Create a dictionary of paths to the test suites
        test_suite_paths = create_suite_file_paths(root_path=pars.suite_save_root, keys=pars.test_names)

    # Run the tests
    print("Running all test suites")
    run_all_test_suites(model_path=pars.model_path, test_suite_paths=test_suite_paths, device=pars.device,
                        max_sequence_length=pars.max_sequence_length)

    print("Closing program")
