from typing import List
import numpy as np

from checklist.test_types import INV
from checklist.test_suite import TestSuite
from checklist.perturb import Perturb
from checklist.editor import Editor


def create_invariance_test_change_neutral_words(suite: TestSuite, sentences: List[str], lexicon: dict,
                                                editor: Editor, example_count: int) -> TestSuite:
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
