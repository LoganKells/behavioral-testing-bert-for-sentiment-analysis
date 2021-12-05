from typing import List

import numpy as np
from checklist.editor import Editor
from checklist.test_suite import TestSuite
from checklist.test_types import MFT


def create_mft_negated_negative(suite: TestSuite, sentences: List[str], labels: List[str], lexicon: dict,
                                editor: Editor, example_count: int) -> TestSuite:
    """
    MFT_Negation_negated_negative
    This function will add a Minimum Functionality Test (MFT) to the TestSuite to check the model's capability
    to handle word negation
    :param suite: TestSuite to add the test to.
    :param sentences: Sentence examples.
    :param labels: Ground truth labels corresponding to each sentence example.
    :param lexicon: Lexicon defined for the dataset.
    :param editor: Editor() class object.
    :param example_count: Count of examples to create.
    :return: TestSuite with the new test added.
    """
    # Create an initial negative review
    ret = editor.template('This is not {a:pos} {mask}.', pos=lexicon['pos_adj'], labels=1,
                          save=True, nsamples=1)

    # Create new sentence examples using the editor.
    template_example_count = 3
    total_count = 1
    for i, ex in enumerate(sentences):
        # Only create a limited number of examples
        if total_count >= example_count:
            break

        # Any sentence examples with { or } will break the Editor.template.
        if '{' in ex or '}' in ex:
            continue

        # filter for examples that are already positive
        label = int(labels[i])
        if label > 2:
            # Use a shortened sentence
            ex_split = ex.split(". ")
            ex = f"{ex_split[0]}."

            # Random path to choose for sentence templates
            left_path = int(np.random.randint(2, size=1)[0])

            # Build either template randomly
            if left_path:
                sentence_template = "I think this is not a {neg} {this:object}. " + ex
            else:
                sentence_template = "This is not a {neg} {this:object}. " + ex

            # Generate a random label of 3 or 4 (corresponding to good reviews)
            random_label = int(np.random.randint(low=3, high=5, size=1)[0])  # Random label of 3 or 4

            # Build examples using the editor template
            ret += editor.template(sentence_template, neg=lexicon['neg_adj'], object=lexicon['object_words'],
                                   labels=random_label, save=True, nsamples=template_example_count)
            # Increment
            total_count += template_example_count

    test_name = "MFT_Negation_negated_negative"
    test_capability = "Negation"
    test_description = "MFT_Negation_negated_negative"
    test = MFT(ret.data, labels=ret.labels, name=test_name, capability=test_capability, description=test_description)

    # Add to suite
    suite.add(test, name=test_name, capability='Vocabulary', description=test_description)

    return suite


def create_mft_negated_positive(suite: TestSuite, sentences: List[str], labels: List[str], lexicon: dict,
                                editor: Editor, example_count: int) -> TestSuite:
    """
    MFT_Negation_negated_positive
    This function will add a Minimum Functionality Test (MFT) to the TestSuite to check the model's capability
    to handle negating a positive, which should result in a negative review.
    :param suite: TestSuite to add the test to.
    :param sentences: Sentence examples.
    :param labels: Ground truth labels corresponding to each sentence example.
    :param lexicon: Lexicon defined for the dataset.
    :param editor: Editor() class object.
    :param example_count: Count of examples to create.
    :return: TestSuite with the new test added.
    """
    # Create an initial negative review
    ret = editor.template('This is not {a:pos} {mask}.', pos=lexicon['pos_adj'], labels=1,
                          save=True, nsamples=1)

    # Create new sentence examples using the editor.
    template_example_count = 3
    total_count = 1
    for i, ex in enumerate(sentences):
        # Only create a limited number of examples
        if total_count >= example_count:
            break

        # Any sentence examples with { or } will break the Editor.template.
        if '{' in ex or '}' in ex:
            continue

        # filter for examples that are already negative
        label = int(labels[i])
        if label < 2:
            # Use a shortened sentence
            ex_split = ex.split(". ")
            ex = f"{ex_split[0]}."

            # Random path to choose for sentence templates
            left_path = int(np.random.randint(2, size=1)[0])

            # Create new examples using the editor.

            # Build either template randomly
            if left_path:
                sentence_template = "I do not like this {pos} {this:object}. " + ex
            else:
                sentence_template = "This is not a {pos} {this:object}. " + ex

            # Generate a random label of 0 or 1 (corresponding to bad reviews)
            random_label = int(np.random.randint(2, size=1)[0])

            # Build examples using the editor template
            ret += editor.template(sentence_template, pos=lexicon['pos_adj'], object=lexicon['object_words'],
                                   labels=random_label, save=True, nsamples=template_example_count)
            # Increment
            total_count += template_example_count

    test_name = "MFT_Negation_negated_positive"
    test_capability = "Negation"
    test_description = "MFT_Negation_negated_positive"
    test = MFT(ret.data, labels=ret.labels, name=test_name, capability=test_capability, description=test_description)

    # Add to suite
    suite.add(test, name=test_name, capability='Vocabulary', description=test_description)

    return suite
