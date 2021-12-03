import argparse
from dataclasses import dataclass
from typing import List, Union, Tuple, Literal
import os
from pathlib import Path, PurePath
import numpy as np
import spacy
import json

from checklist.editor import Editor
import checklist
import itertools
import checklist.editor
import checklist.text_generation
from checklist.test_types import MFT, INV, DIR
from checklist.expect import Expect
from checklist.test_suite import TestSuite
from checklist.perturb import Perturb
from utils_airline_tweets import load_airline_tweets_data
from utils_amazon_reviews import load_amazon_review_data, convert_txt_to_encoded_pt, get_data_loader
from create_predictions import create_predictions, load_model, get_device, save_predictions, save_metadata

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def parse_data(sentences: List[str]):
    nlp = spacy.load('en_core_web_sm')
    return list(nlp.pipe(sentences))


def create_mft_test_negation(suite: TestSuite, sentences: List[str], lexicon: dict) -> TestSuite:
    """
    This funciton will add a Minimum Functionality Test (MFT) to the TestSuite to check the model's capability
    to handle word negation
    :param suite: TestSuite to add the test to.
    :param sentences: Training data sentences.
    :param lexicon: Lexicon defined for the dataset.
    :return: TestSuite with the new test added.
    """
    # Create a list of positive and negative adjectives

    # Create data with positive and negative negations, where 1=positive, 0=negative
    editor = Editor()
    ret = editor.template('This is not {a:pos} {mask}.', pos=lexicon['pos_adj'], labels=0, save=True, nsamples=100)
    ret += editor.template('This is not {a:neg} {mask}.', neg=lexicon['neg_adj'], labels=1, save=True, nsamples=100)

    test = MFT(ret.data, labels=ret.labels, name='Simple negation', capability='Negation',
               description='Very simple negations.')
    raise NotImplementedError("Need to finish this function")


def create_invariance_test_change_neutral_words(suite: TestSuite, sentences: List[str], lexicon: dict,
                                                example_count: int) -> TestSuite:
    """
    This function will add an Invariance test to the TestSuite where neutral words are randomly replaced.
    INVariance: change neutral words. See https://github.com/marcotcr/checklist/blob/master/notebooks/Sentiment.ipynb
    :param suite: TestSuite to add the test to.
    :param sentences: Training data sentences.
    :param lexicon: Lexicon defined for the dataset.
    :param example_count: Number of examples the test suite will create
    :return: TestSuite with the new test added.
    """
    forbidden = set(['No', 'no', 'Not', 'not', 'Nothing', 'nothing', 'without',
                     'but'] + lexicon['pos_adj'] + lexicon['neg_adj'] + lexicon['pos_verb_present'] +
                    lexicon['pos_verb_past'] + lexicon['neg_verb_present'] + lexicon['neg_verb_past'])

    # Swap out netural words with replacements
    def change_neutral(d):
        #     return d.text
        examples = []
        subs = []
        words_in = [x for x in d.capitalize().split() if x in lexicon['neutral_words']]
        if not words_in:
            return None
        for w in words_in:
            suggestions = [x for x in editor.suggest_replace(d, w, beam_size=5, words_and_sentences=True) if
                           x[0] not in forbidden]
            examples.extend([x[1] for x in suggestions])
            subs.extend(['%s -> %s' % (w, x[0]) for x in suggestions])
        if examples:
            idxs = np.random.choice(len(examples), min(len(examples), 10), replace=False)
            return [examples[i] for i in idxs]  # , [subs[i] for i in idxs])

    # Perturb.perturb(parsed_data[:5], perturb)

    t = Perturb.perturb(sentences, change_neutral, nsamples=example_count)
    test = INV(t.data)
    description = 'Change a set of neutral words with other context-appropriate neutral words (using BERT).'

    # Add to suite
    test_name = 'change neutral words with BERT'
    suite.add(test, name=test_name, capability='Vocabulary', description=description)

    return suite


def create_directional_expression_test_add_positive_phrases(suite: TestSuite, sentences: List[str],
                                                            editor: Editor, example_count: int) -> TestSuite:
    """
    This function will add a DIRectional Expression Test: add strongly positive phrases to end of sentence
    See https://github.com/marcotcr/checklist/blob/master/notebooks/Sentiment.ipynb
    :param suite: TestSuite to add the test to.
    :param sentences: Training data sentences.
    :param lexicon: Lexicon defined for the dataset.
    :param example_count: The number of examples the test suite will create.
    :return: TestSuite with the new test added.
    """
    positive = editor.template('I {pos_verb_present} this game.').data
    positive += editor.template('The game is {pos_adj}.').data
    positive += editor.template('This game is {pos_adj}.').data
    positive += editor.template('This game was {pos_adj}.').data
    positive += ['I want to play this game over and over again.']
    positive += ['I love this game.']

    # Send the sentences through an NLP pipeline
    sentences_nlp = parse_data(sentences)

    def diff_up(orig_pred, pred, orig_conf, conf, labels=None, meta=None):
        tolerance = 0.1
        change = positive_change(orig_conf, conf)
        if change + tolerance >= 0:
            return True
        else:
            return change + tolerance

    goes_up = Expect.pairwise(diff_up)
    t = Perturb.perturb(sentences_nlp, add_phrase_function(positive), nsamples=example_count)
    test = DIR(t.data, goes_up)
    description = 'Add very positive phrases (e.g. I love this game) to the end of sentences, ' \
                  'expect probability of positive to NOT go down (tolerance=0.1)'
    suite.add(test, name='add positive phrases', capability='Vocabulary', description=description)

    return suite


def create_directional_expression_test_add_negative_phrases(suite: TestSuite, sentences: List[str],
                                                            editor: Editor, example_count: int) -> TestSuite:
    """
    This function will add DIRectional Expression Test: add strongly negative phrases to end of sentence
    See https://github.com/marcotcr/checklist/blob/master/notebooks/Sentiment.ipynb
    :param suite: TestSuite to add the test to.
    :param sentences: Training data sentences.
    :param lexicon: Lexicon defined for the dataset.
    :param example_count: The number of examples the test suite will create
    :return: TestSuite with the new test added.
    """

    # Send the sentences through an NLP pipeline
    sentences_nlp = parse_data(sentences)

    negative = editor.template('I {neg_verb_present} this game.').data
    negative += editor.template('The game is {neg_adj}.').data
    negative += ['I would never play this game again.']

    def diff_down(orig_pred, pred, orig_conf, conf, labels=None, meta=None):
        tolerance = 0.1
        change = positive_change(orig_conf, conf)
        if change - tolerance <= 0:
            return True
        else:
            return -(change - tolerance)

    goes_down = Expect.pairwise(diff_down)
    t = Perturb.perturb(sentences_nlp, add_phrase_function(negative), nsamples=example_count)
    test = DIR(t.data, goes_down)
    description = 'Add very negative phrases (e.g. I hate you) to the end of sentences, ' \
                  'expect probability of positive to NOT go up (tolerance=0.1)'
    suite.add(test, name='add negative phrases', capability='Vocabulary', description=description)

    return suite


def add_phrase_function(phrases):
    """
    This funciton will add phrases to original
    examples to creates transformed versions
    """

    def pert(d):
        while d[-1].pos_ == 'PUNCT':
            d = d[:-1]
        d = d.text
        ret = [d + '. ' + x for x in phrases]
        idx = np.random.choice(len(ret), 10, replace=False)
        ret = [ret[i] for i in idx]
        return ret

    return pert


def positive_change(orig_conf, conf):
    """
    This funciton will measure the overall positive change 
    in in an example that is transformed to have a strong positive
    or negative phrase added to the end of a sentence. This metric
    measures the net change in logit values on both ends of the spectrum
    by comparing the transofmred and non-transformed sentence.
    """
    softmax = type(orig_conf) in [np.array, np.ndarray]
    if not softmax or orig_conf.shape[0] != 5:
        raise (Exception('Need prediction function to be softmax with 3 labels (negative, neutral, positive)'))
    return orig_conf[0] - conf[0] + conf[4] - orig_conf[4]


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


def build_suites(sentences, lexicon, editor, save_path: PurePath, test_suite_names: Tuple[str, ...]) -> tuple:
    """
    If this function is run, it will create and build all the suites it contains

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
    :param sentences: List of sentence examples
    :param lexicon: lexicon loaded to the Editor()
    :param editor: Editor() that is loaded with lexicon
    :param test_suite_names: List[str] of all the test suite names.
    :return: tuple of (dict of TestSuites, dict of paths to test suite folders)
    """
    tests = {}

    ### TEST 1 ###
    # Test type: Invariance - Invariance test (INV) is when we apply label-preserving perturbations to inputs and
    #                         expect the model prediction to remain the same.
    # Capability: Vocabulary (neutral words changed)
    test_name_invariance_neutral_words = "invariance_neutral_words"
    if test_name_invariance_neutral_words in test_suite_names:
        print(f"Creating test: {test_name_invariance_neutral_words}")
        inv_neutral_suite = create_invariance_test_change_neutral_words(TestSuite(), sentences, lexicon,
                                                                        example_count=100)
        tests[test_name_invariance_neutral_words] = inv_neutral_suite

    ### TEST 2 ###
    test_name_positive_phrases = "directional_positive_phrases"
    if test_name_positive_phrases in test_suite_names:
        print(f"Creating test: {test_name_positive_phrases}")
        pos_suite = create_directional_expression_test_add_positive_phrases(TestSuite(), sentences, editor,
                                                                            example_count=5_000)
        tests[test_name_positive_phrases] = pos_suite

    ### TEST 3 ###
    test_name_negative_phrases = "directional_negative_phrases"
    if test_name_negative_phrases in test_suite_names:
        print(f"Creating test: {test_name_negative_phrases}")
        neg_suite = create_directional_expression_test_add_negative_phrases(TestSuite(), sentences, editor,
                                                                            example_count=5_000)
        tests[test_name_negative_phrases] = neg_suite

    ### Test 4 ###
    # TODO finish this test
    # Test type: Minimum Functionality Test (MFT) - used to verify the model has specific capabilities.
    # Capability: Can the model handle negation?
    # suite = create_mft_test_negation(suite, sentences, lexicon)

    # Save and build any suites added to the dictionary.
    for test_name, suite in tests.items():
        save_build_suite(suite, save_path=save_path / test_name,
                         file_name="test.pkl", samples=100)

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
class RunningParametersAmazonReviews:
    run_tests_from_pkl: bool = False
    data_file_path: PurePath = PROJECT_ROOT / "data" / "sentiment" / "amazon_reviews" / "test_data.csv"
    label_file_path: PurePath = PROJECT_ROOT / "data" / "sentiment" / "amazon_reviews" / "test_data_labels.pt"
    prediction_file_path: PurePath = PROJECT_ROOT / "predictions" / "sentiment" / "amazon_reviews" / "bert_trained" / "bert_multilingual.txt"
    suite_save_root: PurePath = PROJECT_ROOT / "test_suites" / "sentiment"
    suite_file_name: str = "test_suite_sentiment_amazon_reviews.pkl"
    max_sequence_length: int = 75
    model_path: PurePath = PROJECT_ROOT / "models" / "sentiment" / "bert_multilingual_amazon_reviews_hugging"
    device: str = "cpu"
    test_names = ("directional_positive_phrases", "directional_negative_phrases")  # ("invariance_neutral_words", "directional_positive_phrases", "directional_negative_phrases")
    rebuild_test_suites: bool = False


@dataclass
class RunningParametersAirlineTweets:
    run_tests_from_pkl: bool = False
    data_file_path: PurePath = PROJECT_ROOT / "data" / "sentiment" / "airline_tweets" / "Tweets.csv"
    label_file_path: PurePath = None
    prediction_file_path: PurePath = PROJECT_ROOT / "predictions" / "sentiment" / "airline_tweets" / "bert.txt"
    suite_save_root: PurePath = PROJECT_ROOT / "test_suites" / "sentiment"
    suite_file_name: str = "test_suite_sentiment_airline_tweets.pkl"


if __name__ == "__main__":
    # Arguments
    args = _parse_args()
    data_selection = args.data_selection

    # Running Parameters
    if data_selection == "airline_tweets":
        pars = RunningParametersAirlineTweets()
    elif data_selection == "amazon_reviews":
        pars = RunningParametersAmazonReviews()
    else:
        print("Please select from: \"airline_tweets\" or \"amazon_reviews\"")
        pars = None
    pars.device = get_device()

    # Check for errors in parameters
    if ".pkl" not in str(pars.suite_file_name):
        raise ValueError("suite_save_pkl_path must contain .pkl file extension")
    if ".txt" not in str(pars.prediction_file_path):
        raise ValueError("prediction_file_path must contain a .txt file extension")

    if isinstance(pars, RunningParametersAirlineTweets):
        data = load_airline_tweets_data(pars.data_file_path)
        labels, confs, airlines, sentences, reasons = data  # unpack
        # parsed_data = parse_data(sentences)  # TODO determine if this is required
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
        test_suites, test_suite_paths = build_suites(sentences, lexicon, editor, save_path=pars.suite_save_root,
                                                     test_suite_names=pars.test_names)
    else:
        # Create a dictionary of paths to the test suites
        test_suite_paths = create_suite_file_paths(root_path=pars.suite_save_root, keys=pars.test_names)

    # Run the tests
    print("Running all test suites")
    run_all_test_suites(model_path=pars.model_path, test_suite_paths=test_suite_paths, device=pars.device,
                        max_sequence_length=pars.max_sequence_length)

    print("Closing program")
