import numpy as np

import keras_glove.evaluators as ev


def test_most_similar(mocker):

    mock_np_load = mocker.patch('keras_glove.evaluators.np.load')
    mock_np_load.return_value = np.array(
        [
            [1.0, 0.5, 0.2],
            [0.5, 1.0, 0.7],
            [0.2, 0.7, 1.0]
        ]
    )

    mock_np_argsort = mocker.patch('keras_glove.evaluators.np.argsort')
    mock_np_argsort.return_value = np.array([0, 2, 1])

    mock_open = mocker.patch('keras_glove.evaluators.open')
    mock_open.return_value = 'test'

    mock_pickle_load = mocker.patch('keras_glove.evaluators.pickle.load')
    mock_pickle_load.side_effect = [
        {
            'bla': 0,
            'blah': 1,
            'blahblah': 2
         },

        {
            0: 'bla',
            1: 'blah',
            2: 'blahblah'
        }
    ]

    res = ev.get_most_similar(word='blah', k=2)
    assert len(res) == 2
    assert res[0] == ('bla', 0.5)
    assert res[1] == ('blahblah', 0.7)


def test_most_uknown_word(mocker):

    mock_np_load = mocker.patch('keras_glove.evaluators.np.load')
    mock_np_load.return_value = np.array(
        [
            [1.0, 0.5, 0.2],
            [0.5, 1.0, 0.7],
            [0.2, 0.7, 1.0]
        ]
    )

    mock_np_argsort = mocker.patch('keras_glove.evaluators.np.argsort')
    mock_np_argsort.return_value = np.array([0, 2, 1])

    mock_open = mocker.patch('keras_glove.evaluators.open')
    mock_open.return_value = 'test'

    mock_pickle_load = mocker.patch('keras_glove.evaluators.pickle.load')
    mock_pickle_load.side_effect = [
        {
            'bla': 0,
            'blah': 1,
            'blahblah': 2
         },

        {
            0: 'bla',
            1: 'blah',
            2: 'blahblah'
        }
    ]

    res = ev.get_most_similar(word='aaaa', k=2)
    assert len(res) == 0
