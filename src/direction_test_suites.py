from typing import List
import numpy as np
from checklist.editor import Editor
from checklist.test_types import DIR
from checklist.expect import Expect
from checklist.test_suite import TestSuite
from checklist.perturb import Perturb
from utils_amazon_reviews import parse_data


def add_phrase_function(phrases):
    """
    This function will add phrases to original
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
