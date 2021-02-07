from tensorflow.python.keras.engine.functional import Functional

import keras_glove.models as m


def test_glove_model():

    test_model = m.glove_model(vector_dim=15, vocab_size=10)

    assert type(test_model) == Functional
    assert len(test_model.layers) == 11
