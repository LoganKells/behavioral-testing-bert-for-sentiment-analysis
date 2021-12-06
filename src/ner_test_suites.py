from typing import List

from checklist.test_types import INV
from checklist.test_suite import TestSuite
from checklist.perturb import Perturb
from utils_amazon_reviews import parse_data


def create_ner_switch_names(suite: TestSuite, sentences: List[str], example_count: int) -> TestSuite:
    """
    NER_Switching_Names
    Named Entity Recognition (NER) where names are swapped with other names in the sentences.
    :param suite: TestSuite to add the test to.
    :param sentences: Sentence examples.
    :param example_count: Count of examples to create.
    :return: TestSuite with the new test added.
    """
    test_name = "NER_Switching_Names"
    test_capability = "NER"

    parsed_data = parse_data(sentences)
    t = Perturb.perturb(parsed_data, Perturb.change_names, nsamples=example_count)
    test = INV(t.data)

    # Add to suite
    suite.add(test, name=test_name, capability=test_capability, description=test_name)

    return suite


def create_ner_switch_locations(suite: TestSuite, sentences: List[str], example_count: int) -> TestSuite:
    """
    NER_Switching_Location
    Named Entity Recognition (NER) where names are swapped with other names in the sentences.
    :param suite: TestSuite to add the test to.
    :param sentences: Sentence examples.
    :param example_count: Count of examples to create.
    :return: TestSuite with the new test added.
    """
    test_name = "NER_Switching_Locations"
    test_capability = "NER"

    parsed_data = parse_data(sentences)
    t = Perturb.perturb(parsed_data, Perturb.change_location, nsamples=example_count)
    test = INV(t.data)

    # Add to suite
    suite.add(test, name=test_name, capability=test_capability, description=test_name)

    return suite
