from typing import List
import numpy as np
from checklist.editor import Editor
import checklist
import itertools
import checklist.editor
import checklist.text_generation
from checklist.test_types import MFT, INV, DIR
from checklist.expect import Expect
from checklist.test_suite import TestSuite
from checklist.perturb import Perturb


def create_mft_test_negation(suite: TestSuite, sentences: List[str], labels, lexicon: dict,
                             editor: Editor, example_count: int) -> TestSuite:
    """
    This function will add a Minimum Functionality Test (MFT) to the TestSuite to check the model's capability
    to handle word negation
    :param suite: TestSuite to add the test to.
    :param sentences: Training data sentences.
    :param lexicon: Lexicon defined for the dataset.
    :return: TestSuite with the new test added.
    """
    # Create an initial negative review
    ret = editor.template('This is not {a:pos} {mask}.', pos=lexicon['pos_adj'], labels=1,
                          save=True, nsamples=1)

    # Create new sentence examples using the editor.
    template_example_count = 3
    total_count = 1
    for i, ex in enumerate(sentences):
        # Any sentence examples with { or } will break the Editor.template.
        if '{' in ex or '}' in ex:
            continue

        # Only create a limited number of examples
        if total_count >= example_count:
            break

        # Random path to choose for sentence templates
        left_path = int(np.random.randint(2, size=1)[0])

        # Create new examples using the editor.
        label = labels[i]
        if label > 2:  # Create a negative review from a previously positive review
            # Build either template randomly
            if left_path:
                sentence_template = "I do not like this {pos} {this:object}. " + ex + " Do not buy."
            else:
                sentence_template = "This is not a {pos} {this:object}. " + ex + " Do not buy."

            # Generate a random label of 0 or 1 (corresponding to bad reviews)
            random_label = int(np.random.randint(2, size=1)[0])

            # Build examples using the editor template
            ret += editor.template(sentence_template, pos=lexicon['pos_adj'], object=lexicon['object_words'],
                                   labels=random_label, save=True, nsamples=template_example_count)
            # Increment
            total_count += template_example_count
        elif label < 2:  # Create a positive review from the previously negative review
            # Build either template randomly
            if left_path:
                sentence_template = "I think this is not a {neg} {this:object}. " + ex + " You should buy it."
            else:
                sentence_template = "This is not a {neg} {this:object}. " + ex + " You should buy it."

            # Generate a random label of 3 or 4 (corresponding to good reviews)
            random_label = int(np.random.randint(low=3, high=5, size=1)[0])  # Random label of 3 or 4

            # Build examples using the editor template
            ret += editor.template(sentence_template, neg=lexicon['neg_adj'], object=lexicon['object_words'],
                                   labels=random_label, save=True, nsamples=template_example_count)
            # Increment
            total_count += template_example_count

    test_name = "MFT Vocabulary (word negation)"
    test_capability = "Vocabulary (word negation)"
    test_description = "MFT, capability: Vocabulary (word negation), simple negations."
    test = MFT(ret.data, labels=ret.labels, name=test_name, capability=test_capability, description=test_description)

    # Add to suite
    suite.add(test, name=test_name, capability='Vocabulary', description=test_description)

    return suite