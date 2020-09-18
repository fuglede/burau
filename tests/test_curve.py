import pytest

from buraucracy.curve import calculate_polynomial, Curve


@pytest.fixture
def simple_example():
    return Curve(2, 1, 3, 2)


def test_cap_outer(simple_example):
    assert simple_example.cap_outer == 1


def test_num_strands(simple_example):
    assert simple_example.num_strands == 10


def test_puncture_locations(simple_example):
    assert simple_example.northwest_puncture == 3
    assert simple_example.northeast_puncture == 7


def test_top_pairing(simple_example):
    assert simple_example.north_pairing == {0: 9, 1: 5, 2: 4, 4: 2,
                                            5: 1, 6: 8, 8: 6, 9: 0}


def test_bottom_pairing(simple_example):
    assert simple_example.south_pairing == {0: 5, 1: 4, 2: 3, 3: 2, 4: 1,
                                            5: 0, 6: 9, 7: 8, 8: 7, 9: 6}


def test_is_beta_connected(simple_example):
    with pytest.raises(RuntimeError):
        simple_example.is_beta_connected()
    for _ in simple_example:
        pass
    assert simple_example.is_beta_connected()


def test_norm(simple_example):
    with pytest.warns(UserWarning):
        assert simple_example.norm() == 0
    for _ in simple_example:
        pass
    assert simple_example.norm() == 5


def test_calculate_polynomial():
    polynomial = calculate_polynomial(2, 1, 3, 2)
    assert polynomial == {0: 1, -2: 1, -4: 1, -8: -1, -10: -1}

