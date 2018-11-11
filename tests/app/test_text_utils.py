from collections import defaultdict

import pytest

import app.text_utils as tu


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
