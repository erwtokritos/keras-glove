from collections import defaultdict

import pytest

import keras_glove.text_utils as tu


@pytest.fixture()
def mock_cache():

    m_cache = defaultdict(lambda: defaultdict(int))
    m_cache['a']['b'] = 2
    m_cache['a']['c'] = 3
    m_cache['b']['d'] = 4

    return m_cache


def test_cache_to_pairs(mock_cache):

    res0, res1, res2 = tu.cache_to_pairs(cache=mock_cache)

    assert len(res0) == 6
    assert len(res1) == 6
    assert len(res2) == 6

    assert res0[0] == 'a'
    assert res1[0] == 'b'
    assert res2[0] == 2.0

    assert res0[-1] == 'd'
    assert res1[-1] == 'b'
    assert res2[-1] == 4.0


def test_bigram_count():

    # input arguments
    _cache = defaultdict(lambda: defaultdict(int))
    _token_list = ['a', 'b', 'c']
    _window = 2

    tu.bigram_count(_token_list, _window, _cache)

    assert len(_cache) == 2 # two keys 'a' & 'b'
    assert _cache['a']['b'] == 1.0 / 1.0
    assert _cache['a']['c'] == 1.0 / 2.0
    assert _cache['b']['c'] == 1.0 / 1.0


def test_build_cooccurrences(mocker):
    # mock bigram count
    mock_bigram_count = mocker.patch('keras_glove.text_utils.bigram_count')

    # input arguments
    _cache = defaultdict(lambda: defaultdict(int))
    _sequences = [[1, 2, 3], [4, 5]]
    _window = 100000000
    tu.build_cooccurrences(_sequences, _cache, _window)

    assert mock_bigram_count.call_count == 2


def test_read_file():

    mock_file = [b'a b c', b'a b c', b'a b c']

    response = tu.read_file(file=mock_file)
    print(response)
    assert len(response) == 3
    assert response[0] == 'a b c'

    response = tu.read_file(file=mock_file, num_lines=1)
    assert len(response) == 1


def test_tokenizer(mocker):

    mock_fit = mocker.patch.object(tu.Tokenizer, 'fit_on_texts')
    mock_t2s = mocker.patch.object(tu.Tokenizer, 'texts_to_sequences')

    tu.tokenize(lines=[])
    assert mock_fit.call_count == 1
    assert mock_t2s.call_count == 1
