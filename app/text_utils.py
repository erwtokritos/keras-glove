from typing import List, Union, Tuple
from collections import defaultdict

from keras.preprocessing.text import Tokenizer
import numpy as np


def read_file(filename, num_lines=0) -> List[str]:
    """
    A simple file reader
    :param filename: The name of the file
    :param num_lines: Number of lines to read from the file.
    :return:
    """
    response = []
    for ix, line in enumerate(filename):
        response.append(str(line, errors='ignore'))

        if num_lines > 0 and ix > num_lines:
            break
    return response


def tokenize(lines: List[str], num_words=10000) -> Tuple[List[List], Tokenizer]:
    """
    Using Keras' Tokenizer
    :param lines: A list of strings i.e. the sentences
    :param num_words: The maximum number of words to keep
    :return: A list of word indices per sentence & the tokenizer object itself
    """
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(lines)

    sequences = tokenizer.texts_to_sequences(lines)

    return sequences, tokenizer


def bigram_count(token_list: List[int], window_size: int, cache: defaultdict):
    """
    It computes the actual co-occurrence patterns required by GloVe. The score of a pair
    if weighted by their distance e.g. in a sentence where words 'a' and 'b' are k words apart
    the corresponding cache entry will be updated as
                cache(a, b) += 1.0 / k
    :param token_list: The representation of a sentence as a list of integers (word indices)
    :param window_size: The size of the window around the central word
    :param cache: The cache
    """
    sentence_size = len(token_list)

    for central_index, central_word_id in enumerate(token_list):
        for distance in range(1, window_size + 1):
            if central_index + distance < sentence_size:
                first_id, second_id = sorted([central_word_id, token_list[central_index + distance]])
                cache[first_id][second_id] += 1.0 / distance


def build_cooccurrences(sequences: List[List[int]], cache: defaultdict, window=3):
    """
    It updates a shared cache for by iteratively calling 'bigram_count'
    :param sequences: The input sequences
    :param cache: The current cache
    :param window: The size of window to look around a central word
    """

    for seq in sequences:
        bigram_count(token_list=seq, cache=cache, window_size=window)


def cache_to_pairs(cache: defaultdict) -> Union[np.array, np.array, np.array]:
    """
    Given the computed cache it produces the
    :param cache:
    :return:
    """

    first, second, x_ijs = [], [], []

    for first_id in cache.keys():

        for second_id in cache[first_id].keys():

            x_ij = cache[first_id][second_id]

            # add (main, context) pair
            first.append(first_id)
            second.append(second_id)
            x_ijs.append(x_ij)

            # add (context, main) pair
            first.append(second_id)
            second.append(first_id)
            x_ijs.append(x_ij)

    return np.array(first), np.array(second), np.array(x_ijs)
