import random

import pytest

from burau.curve import calculate_polynomial, Curve


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
    assert simple_example.north_pairing == {
        0: 9,
        1: 5,
        2: 4,
        4: 2,
        5: 1,
        6: 8,
        8: 6,
        9: 0,
    }


def test_bottom_pairing(simple_example):
    assert simple_example.south_pairing == {
        0: 5,
        1: 4,
        2: 3,
        3: 2,
        4: 1,
        5: 0,
        6: 9,
        7: 8,
        8: 7,
        9: 6,
    }


def test_is_beta_connected(simple_example):
    with pytest.raises(RuntimeError):
        simple_example.is_beta_connected()
    simple_example.run_to_end()
    assert simple_example.is_beta_connected()


def test_norm(simple_example):
    with pytest.warns(UserWarning):
        assert simple_example.norm() == 0
    simple_example.run_to_end()
    assert simple_example.norm() == 5


@pytest.mark.parametrize("use_numba", [False, True])
def test_calculate_polynomial(use_numba):
    polynomial, conn, crossings = calculate_polynomial(
        2, 1, 3, 2, use_numba=use_numba
    )
    assert polynomial == {0: 1, -2: 1, -4: 1, -8: -1, -10: -1}
    assert conn
    assert crossings == 5


def test_two_implementations_agree():
    # Run 500 tests both with and without Numba, ensure that the results agree
    # when the input sizes are admissible, and that both implementations raise
    # when they are not
    random.seed(42)
    for _ in range(500):
        cap_west, cap_east, cup_west, cup_east = [
            random.randint(0, 100) for _ in range(4)
        ]
        numba_raises = no_numba_raises = False
        try:
            pol_numba, conn_numba, crossings_numba = calculate_polynomial(
                cap_west, cap_east, cup_west, cup_east
            )
        except ValueError:
            numba_raises = True
        try:
            (
                pol_no_numba,
                conn_no_numba,
                crossings_no_numba,
            ) = calculate_polynomial(
                cap_west, cap_east, cup_west, cup_east, use_numba=False
            )
        except ValueError:
            no_numba_raises = True
        assert numba_raises == no_numba_raises
        if not numba_raises:
            assert pol_numba == pol_no_numba
            assert conn_numba == conn_no_numba
            assert crossings_numba == crossings_no_numba
