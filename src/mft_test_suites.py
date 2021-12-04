from typing import List
from checklist.editor import Editor
import checklist
import itertools
import checklist.editor
import checklist.text_generation
from checklist.test_types import MFT, INV, DIR
from checklist.expect import Expect
from checklist.test_suite import TestSuite
from checklist.perturb import Perturb


def create_mft_test_negation(suite: TestSuite, sentences: List[str], lexicon: dict,
                             editor: Editor, example_count: int) -> TestSuite:
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