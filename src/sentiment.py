import argparse
from typing import List, Union, Tuple, Literal
from checklist.editor import Editor
import checklist
import spacy
import itertools
import csv
import checklist.editor
import checklist.text_generation
from checklist.test_types import MFT, INV, DIR
from checklist.expect import Expect
import numpy as np
import pandas as pd
import spacy
from checklist.test_suite import TestSuite
from checklist.perturb import Perturb
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_data(data_path: Union[str, object]) -> tuple:
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


def load_editor(editor: Editor, suite_name: Literal["airline_tweets", "amazon_reviews"]) -> Tuple[Editor, dict]:
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
    parsed_data = list(nlp.pipe(sentences))
    return parsed_data


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


def create_invariance_test_change_neutral_words(suite: TestSuite, sentences: List[str], lexicon: dict) -> TestSuite:
    """
    This function will add an Invariance test to the TestSuite where neutral words are randomly replaced.
    INVariance: change neutral words. See https://github.com/marcotcr/checklist/blob/master/notebooks/Sentiment.ipynb
    :param suite: TestSuite to add the test to.
    :param sentences: Training data sentences.
    :param lexicon: Lexicon defined for the dataset.
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

    t = Perturb.perturb(sentences, change_neutral, nsamples=500)
    test = INV(t.data)
    description = 'Change a set of neutral words with other context-appropriate neutral words (using BERT).'

    # Add to suite
    test_name = 'change neutral words with BERT'
    suite.add(test, name=test_name, capability='Vocabulary', description=description)

    return suite


def save_build_suite(suite: TestSuite, save_to_pkl_path: Union[str, object], samples: int) -> None:
    """
    This wrapper will save the TestSuite while also building for analysis.
    :param suite: TestSuite to build and save
    :param save_to_pkl_path: Path-like object to save the .pkl file locally.
    :param samples: Number of samples to draw from for each suite.
    :return: None
    """
    # Make directory if missing
    sentiment_path = PROJECT_ROOT / "test_suites" / "sentiment"
    if not os.path.isdir(sentiment_path):
        os.mkdir(sentiment_path)

    # Build the TestSuite
    temp_file_path = sentiment_path / "temp"
    suite.to_raw_file(str(temp_file_path), n=samples, seed=1)
    for test in suite.tests:
        suite.tests[test].name = test
        suite.tests[test].description = suite.info[test]['description]']
        suite.tests[test].capability = suite.info[test]['capability']

    # Save the TestSuite to a local .pkl file
    suite.save(str(save_to_pkl_path))

    # Clean up temp file
    os.remove(temp_file_path)


def run_analysis(suite_path: Union[str, object], pred_path: Union[str, object]) -> None:
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


def _parse_args() -> argparse.Namespace:
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Create TestSuites and run analysis given training data and predictions.")
    parser.add_argument("--training_data_file", required=True, type=str, default="Tweets.csv", choices=["Tweets.csv"])
    parser.add_argument("--prediction_data_file", required=True, type=str, default="bert.txt",
                        choices=["amazon.txt", "bert.txt"])
    parser.add_argument("--suite_name", required=True, type=str, default="airline_tweets", help="Name for suite data.")
    parser.add_argument("--run_from_built_pkl", action='store_true',
                        help="Run the TestSuite loaded with training data and tests from a prebuilt pickle.")
    return parser.parse_args()


if __name__ == "__main__":
    # Arguments
    args = _parse_args()
    suite_name = args.suite_name
    suite_type = "sentiment"
    training_data_file_name, prediction_data_file_name = args.training_data_file, args.prediction_data_file
    run_tests_from_pkl = args.run_from_built_pkl
    suite_save_pkl_path = PROJECT_ROOT / "test_suites" / suite_type / f"test_suite_{suite_type}_{suite_name}.pkl"

    if not run_tests_from_pkl:
        # Data pre-processing
        data = load_data(PROJECT_ROOT / "data" / suite_type / suite_name / training_data_file_name)
        labels, confs, airlines, sentences, reasons = data  # unpack
        # parsed_data = parse_data(sentences)  # TODO determine if this is required
        parsed_data = sentences

        # Create a TestSuite
        print("Creating TestSuite")
        suite = TestSuite()

        # Create an Editor
        print("Loading Editor")
        editor = Editor()
        editor, lexicon = load_editor(editor, suite_name=suite_name)

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

        print("Creating Tests")
        # Test type: Minimum Functionality Test (MFT) - used to verify the model has specific capabilities.
        # Capability: Can the model handle negation?
        suite = create_mft_test_negation(suite, sentences, lexicon)

        # Test type: Invariance - Invariance test (INV) is when we apply label-preserving perturbations to inputs and
        #                         expect the model prediction to remain the same.
        # Capability: Vocabulary (neutral words changed)
        suite = create_invariance_test_change_neutral_words(suite, sentences, lexicon)

        # Save the suite
        print("Building suite")
        save_build_suite(suite, save_to_pkl_path=suite_save_pkl_path,
                         samples=500)

    # Run the analysis
    print("Running analysis")
    run_analysis(suite_path=suite_save_pkl_path,
                 pred_path=str(PROJECT_ROOT / "predictions" / suite_type / suite_name / prediction_data_file_name))

    print("Closing program")
