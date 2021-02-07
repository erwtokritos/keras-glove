import pytest
import numpy as np

import keras_glove.save_utils as su


@pytest.fixture()
def mock_model(mocker):

    mock_layer0 = mocker.Mock()
    mock_layer1 = mocker.Mock()
    mock_layer0.name = 'layer_0_biases'
    mock_layer1.name = 'layer_1_other'

    mock_layer0.get_weights = mocker.Mock()
    mock_layer0.get_weights.return_value = [np.ones((3, 1))]

    mock_model = mocker.Mock()
    mock_model.layers = [mock_layer0, mock_layer1]

    return mock_model


@pytest.fixture()
def mock_tokenizer(mocker):

    mock_tokenizer = mocker.Mock()
    mock_tokenizer.index_word = mocker.Mock()
    mock_tokenizer.word_index = mocker.Mock()

    return mock_tokenizer


def test_save_utils(mock_model, mock_tokenizer, mocker):

    mock_np_save = mocker.patch('keras_glove.save_utils.np.save')
    mock_pickle_dump = mocker.patch('keras_glove.save_utils.pickle.dump')
    mock_open = mocker.patch('builtins.open')
    mock_cos_sim = mocker.patch('keras_glove.save_utils.cosine_similarity')
    mock_cos_sim.return_value = np.ones((3, 3)) * 0.25

    mock_np_load = mocker.patch('keras_glove.save_utils.np.load')
    mock_np_load.side_effect = [
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0, 6.0])
    ]

    su.save_model(model=mock_model, tokenizer=mock_tokenizer)
    assert mock_np_save.call_count == 3
    assert mock_pickle_dump.call_count == 2
    assert mock_open.call_count == 2
