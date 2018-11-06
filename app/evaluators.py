from typing import List, Tuple
import pickle

import numpy as np

from app.config import *


def get_most_similar(word: str, k: int) -> List[Tuple]:

    corr_matrix = np.load(f'{OUTPUT_FOLDER}{CORRELATION_MATRIX}.npy')
    word_to_index = pickle.load(open(f'{OUTPUT_FOLDER}{WORD2INDEX}', 'rb'))
    index_to_word = pickle.load(open(f'{OUTPUT_FOLDER}{INDEX2WORD}', 'rb'))

    if word not in word_to_index:
        return []

    word_index = word_to_index[word]
    closest = np.argsort(corr_matrix[word_index, :])[-k:-1]

    return [
        (index_to_word[v], corr_matrix[word_index, v])
        for v in closest
    ]
