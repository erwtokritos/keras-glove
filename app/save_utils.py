import pickle

from keras.models import Model
from keras.preprocessing.text import Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from app.config import *


def save_model(model: Model, tokenizer: Tokenizer):
    """
    Saves the important parts of the model
    :param model: Keras model to save
    :param tokenizer: Keras Tokenizer to save
    """
    for layer in model.layers:
        if '_biases' in layer.name or '_embeddings' in layer.name:
            np.save(file=f'{OUTPUT_FOLDER}{layer.name}', arr=layer.get_weights()[0])

    # save tokenizer
    pickle.dump(obj=tokenizer.index_word, file=open(f'{OUTPUT_FOLDER}{INDEX2WORD}', 'wb'))
    pickle.dump(obj=tokenizer.word_index, file=open(f'{OUTPUT_FOLDER}{WORD2INDEX}', 'wb'))

    # save combined embeddings & correlation matrix
    agg_embeddings = np.load(f'{OUTPUT_FOLDER}{CENTRAL_EMBEDDINGS}.npy') + \
                     np.load(f'{OUTPUT_FOLDER}{CONTEXT_EMBEDDINGS}.npy')

    np.save(file=f'{OUTPUT_FOLDER}{AGGREGATED_EMBEDDINGS}', arr=agg_embeddings)
    np.save(file=f'{OUTPUT_FOLDER}{CORRELATION_MATRIX}', arr=cosine_similarity(cosine_similarity(agg_embeddings)))
