r"""Algorithm for finding kernel elements of the B_4 Burau representation.

Background
----------
Here, we follow the lead from [0] and consider a particular family of pairs
of curves in the four-punctured disc. We follow more or less the same
prescription as in Bigelow's C implementation currently available at

    https://github.com/freshbugs/burau4/blob/master/iv.c

and in particular, we thank Bigelow for useful ideas on how to represent the
curves: We place the four punctures in a square, identifing them in our
explanation via compass directions. In the notation of [0], we let
:math:`\alpha` denote the curve connecting the two south punctures, ordered so
that the code below does not have any sign mistakes::

    ╳  ╳

    ╳──╳

The curve :math:`\beta` is more complicated. The two north punctures are placed
at the middle of two caps, each consisting of a given number of parallel
curves, the two south punctures at the middle of two such multi-cups, two
vertical strands are pulled south from each of the two north punctures, and a
multi-cap extending above the two north caps ensures that everything can be
tied together.

As an example, suppose `cap_west = 2`, `cap_east = 1`, `cup_west = 3`,
and `cup_east = 2`. Then, joining the north and south halves of the picture,
we have::

    num_strands = 2 * (cup_west + cup_east) = 10

strands, and to tie things together, we need an outer cap containing::

   cap_outer = cup_west + cup_east - (cap_west + cap_east + 1) = 1

strand. At the end of the day, we get a picture that looks as follows (not
including the drawing of :math:`\alpha`)::

    ┌─────────────────┐
    │ ┌───────┐       │
    │ │ ┌───┐ │ ┌───┐ │
    │ │ │ ╳ │ │ │ ╳ │ │
    │ │ │ │ │ │ │ │ │ │
    │ │ │╳│ │ │ │ │╳│ │
    │ │ └─┘ │ │ │ └─┘ │
    │ └─────┘ │ └─────┘
    └─────────┘

Note that for this picture to make sense, we must require that::

    cup_west + cup_east - (cap_west + cap_east) > 0.

Our goal is to calculate, in the notation of [0], :math:`\int_\beta \alpha`. As
such, we follow :math:`\beta` from the northwest puncture to the northeast
puncture, along the way keeping track of intersections with :math:`\alpha`.
Each intersection contributes a summand :math:`\pm t^k:, in which the sign of
the coefficient is determined by whether we intersect from the north or the
south, and whose power is determined by the current "level". To determine this
power, we picture our four-punctured disk as the vertical projection of a
parking garage that extends infinitely up and down (see [1] for an
illustration), with a copy of :math:`\alpha` living in each level. We start at
level 0 of the garage, and as we move along, we may encounter four "ramps" that
take us between different levels::

    down             up
   ──────╳        ╳──────
     up             down

    down             up
   ──────╳        ╳──────
     up             down

With the example :math:`\beta` above, we first encounter an :math:`\alpha` at
level 0 from the north, giving us a summand of :math:`t^0`. Then, we loop
around, encounter two ramps, both of which take us down a level, before we
encounter :math:`\alpha` from the north again, now at level -2, giving us a
summand of :math:`t^{-2}`. A bit later, we get a :math:`t^{-4}` before we loop
all the way around, encounter four down-ramps, then intersect :math:`\alpha`
from the south, this time changing the sign, so we get a :math:`-t^{-8}` and a
bit later a :math:`-t^{-10}`. Adding all of these up, we find that

.. math::

    \int_\beta \alpha = 1 + t^{-2} + t^{-4} - t^{-8} - t^{-10}

To show that the Burau representation of :math:`B_4` is not faithful amounts to
finding a non-trivial :math:`\beta` so that the above polynomial is 0.

Usage
-----

The above example can be reproduced using the functionality of this Python
module as follows:

>>> calculate_polynomial(cap_west=2, cap_east=1, cup_west=3, cup_east=2)
(DictType[int64,int64]<iv=None>({0: 1, -2: 1, -4: 1, -8: -1, -10: -1}),
 True,
 5)

Here, the first output is a dictionary mapping a power of the polynomial to the
coefficient of that power. The second output indicates that the curve beta is
a connected curve (an example for which this is not the case is the
input (1, 1, 3, 3)). The third output is the number of crossings with
:math:`\alpha` encountered along the way.

A kernel element thus corresponds to the empty dictionary. The implementation
uses Numba under the hood to improve the speed of the calculation. We also
provide a more vanilla Python implementation which is about 100x slower than
the Numba-friendly one, but is easier to read and can be used if system
restrictions make it impossible to run Numba.

References
----------
.. [0] Bigelow, Stephen (1999). "The Burau representation is not faithful
       for n = 5". Geometry & Topology. 3: 397–404. arXiv:math/9904100.
       doi:10.2140/gt.1999.3.397.
.. [1] https://en.wikipedia.org/wiki/Burau_representation
"""

from collections import defaultdict
from dataclasses import dataclass
import logging
import warnings

import numpy as np
from numba import njit
from numba.core import types
from numba.typed import Dict


@dataclass
class Curve:
    r"""Represents a particular curve :math:`\beta`."""
    cap_west: int
    cap_east: int
    cup_west: int
    cup_east: int

    def __post_init__(self):
        self.cap_outer = self.cup_west + self.cup_east - \
                         (self.cap_west + self.cap_east + 1)
        self.num_strands = 2 * (self.cup_west + self.cup_east)
        if self.cap_outer < 0:
            raise ValueError('inadmissible cap/cup widths')
        self.northwest_puncture = self.cap_outer + self.cap_west
        self.northeast_puncture = self.cap_outer + 2 * self.cap_west + 1 \
            + self.cap_east

        # Create the two pairings between the strands
        self.north_pairing = {}
        for i in range(self.cap_outer):
            self.north_pairing[i] = self.num_strands - (i + 1)
        for i in range(self.cap_west):
            self.north_pairing[self.northwest_puncture - (i + 1)] = \
                self.northwest_puncture + i + 1
        for i in range(self.cap_east):
            self.north_pairing[self.northeast_puncture - (i + 1)] = \
                self.northeast_puncture + i + 1
        self.north_pairing.update(
            [(v, k) for k, v in self.north_pairing.items()])

        self.south_pairing = {}
        for i in range(self.cup_west):
            self.south_pairing[self.cup_west - (i + 1)] = self.cup_west + i
        for i in range(self.cup_east):
            self.south_pairing[2 * self.cup_west + self.cup_east - (
                        i + 1)] = 2 * self.cup_west + self.cup_east + i
        self.south_pairing.update(
            [(v, k) for k, v in self.south_pairing.items()])

        # We start at level 0 at our northwest puncture moving
        # towards the south.
        self.current_strand = self.northwest_puncture
        self.at_north = True
        self.southbound = True
        self.polynomial = defaultdict(int)
        self.level = 0
        self.steps = 0
        self.done = False
        self.num_crossings = 0

    def _intersecting_alpha(self):
        return self.cup_west <= self.current_strand\
               < 2 * self.cup_west + self.cup_east

    def _crossing_northwest_ramp(self):
        return self.current_strand < self.northwest_puncture

    def _crossing_northeast_ramp(self):
        return self.current_strand > self.northeast_puncture

    def _crossing_southwest_ramp(self):
        return self.current_strand < self.cup_west

    def _crossing_southeast_ramp(self):
        return self.current_strand >= 2 * self.cup_west + self.cup_east

    def _intersect_alpha(self):
        self.polynomial[self.level] += 1 if self.southbound else -1
        self.num_crossings += 1
        # Get rid of terms with zero coefficients
        if self.polynomial[self.level] == 0:
            del self.polynomial[self.level]

    def is_beta_connected(self):
        if not self.done:
            raise RuntimeError('connectedness can only be checked once beta '
                               'has been traversed')
        return self.steps == 2 * self.num_strands - 1

    def norm(self):
        if not self.done:
            warnings.warn('beta has not yet been traversed')
        return sum(abs(v) for v in self.polynomial.values())

    def run_to_end(self):
        for _ in self:
            pass

    def __next__(self):
        r"""Move along math:`\beta`."""
        logging.debug(
            f'At {"north" if self.at_north else "south"} end of'
            f'strand {self.current_strand}, '
            f'facing {"south" if self.southbound else "north"}')
        if self.at_north and self.current_strand == self.northeast_puncture:
            self.done = True
            raise StopIteration

        self.steps += 1
        # As we move along the path, we update our polynomial
        if self.at_north:
            if self.southbound:
                # Move south, stay at same strand, but keep track of crossings
                # with ramps and alpha
                if self._crossing_northwest_ramp():
                    self.level += 1
                    logging.debug(
                        f'Going up northwest ramp. '
                        f'Now at level {self.level}.')
                if self._crossing_southwest_ramp():
                    self.level += 1
                    logging.debug(
                        f'Going up southwest ramp. '
                        f'Now at level {self.level}.')
                if self._crossing_northeast_ramp():
                    self.level -= 1
                    logging.debug(
                        f'Going down northeast ramp. '
                        f'Now at level {self.level}.')
                if self._crossing_southeast_ramp():
                    self.level -= 1
                    logging.debug(
                        f'Going down southeast ramp. '
                        f'Now at level {self.level}.')
                if self._intersecting_alpha():
                    self._intersect_alpha()
                    logging.debug(
                        f'Intersecting alpha from above '
                        f'at level {self.level}.')
                self.at_north = False
            else:
                # Currently facing north, so follow pairing, stay north,
                # but face south
                self.current_strand = self.north_pairing[self.current_strand]
                self.southbound = True
        else:
            if self.southbound:
                # Currently facing south, so follow pairing, stay south,
                # but face north
                self.current_strand = self.south_pairing[self.current_strand]
                self.southbound = False
            else:
                # Move north, stay at same strand, but keep track of crossings
                # with ramps and alpha
                if self._crossing_southwest_ramp():
                    self.level -= 1
                    logging.debug(
                        f'Going down southwest ramp. '
                        f'Now at level {self.level}.')
                if self._crossing_northwest_ramp():
                    self.level -= 1
                    logging.debug(
                        f'Going down northwest ramp. '
                        f'Now at level {self.level}.')
                if self._crossing_southeast_ramp():
                    self.level += 1
                    logging.debug(
                        f'Going up southeast ramp. '
                        f'Now at level {self.level}.')
                if self._crossing_northeast_ramp():
                    self.level += 1
                    logging.debug(
                        f'Going up northeast ramp. '
                        f'Now at level {self.level}.')
                if self._intersecting_alpha():
                    self._intersect_alpha()
                    logging.debug(
                        f'Intersecting alpha from below '
                        f'at level {self.level}.')
                self.at_north = True

    def __iter__(self):
        return self


@njit
def calculate_polynomial_impl(cap_west, cap_east, cup_west, cup_east):
    cap_outer = cup_west + cup_east - (cap_west + cap_east + 1)
    """Numba-friendly implementation of polynomial calculation
    
    This is more or less a one-to-one port of the pure-Python implementation
    above, with all of the gory duplication that comes with that. The only real
    differences are using arrays to represent pairings, and using Numba's typed
    dictionary to represent the polynomial.
    """
    if cap_outer < 0:
        raise ValueError('inadmissible cap/cup widths')
    num_strands = 2 * (cup_west + cup_east)
    northwest_puncture = cap_outer + cap_west
    northeast_puncture = cap_outer + 2 * cap_west + 1 + cap_east

    # Compared to our pure-Python implementation above, we now use arrays
    # instead of dictionaries to simplify access.
    north_pairing = np.empty(num_strands, dtype=np.int_)
    for i in range(cap_outer):
        j = num_strands - (i + 1)
        north_pairing[i] = j
        north_pairing[num_strands - (i + 1)] = i
    for i in range(cap_west):
        north_pairing[northwest_puncture - (i + 1)] =\
            northwest_puncture + i + 1
        north_pairing[northwest_puncture + i + 1] =\
            northwest_puncture - (i + 1)
    for i in range(cap_east):
        north_pairing[northeast_puncture - (i + 1)] = \
            northeast_puncture + i + 1
        north_pairing[northeast_puncture + i + 1] = \
            northeast_puncture - (i + 1)

    south_pairing = np.empty(num_strands, dtype=np.int_)
    for i in range(cup_west):
        south_pairing[cup_west - (i + 1)] = cup_west + i
        south_pairing[cup_west + i] = cup_west - (i + 1)
    for i in range(cup_east):
        south_pairing[2 * cup_west + cup_east - (i + 1)]\
            = 2 * cup_west + cup_east + i
        south_pairing[2 * cup_west + cup_east + i]\
            = 2 * cup_west + cup_east - (i + 1)

    current_strand = northwest_puncture
    at_north = True
    southbound = True
    polynomial = Dict.empty(key_type=types.int64, value_type=types.int64)
    level = 0
    steps = 0
    num_crossings = 0

    def intersecting_alpha(strand):
        return cup_west <= strand < 2 * cup_west + cup_east

    def crossing_northwest_ramp(strand):
        return strand < northwest_puncture

    def crossing_northeast_ramp(strand):
        return strand > northeast_puncture

    def crossing_southwest_ramp(strand):
        return strand < cup_west

    def crossing_southeast_ramp(strand):
        return strand >= 2 * cup_west + cup_east

    while not at_north or current_strand != northeast_puncture:
        steps += 1
        # As we move along the path, we update our polynomial
        if at_north:
            if southbound:
                # Move south, stay at same strand, but keep track of crossings
                # with ramps and alpha
                if crossing_northwest_ramp(current_strand):
                    level += 1
                if crossing_southwest_ramp(current_strand):
                    level += 1
                if crossing_northeast_ramp(current_strand):
                    level -= 1
                if crossing_southeast_ramp(current_strand):
                    level -= 1
                if intersecting_alpha(current_strand):
                    polynomial[level] = polynomial.get(level, 0) + 1
                    num_crossings += 1
                    if polynomial[level] == 0:
                        del polynomial[level]
                at_north = False
            else:
                current_strand = north_pairing[current_strand]
                southbound = True
        else:
            if southbound:
                # Currently facing south, so follow pairing, stay south,
                # but face north
                current_strand = south_pairing[current_strand]
                southbound = False
            else:
                # Move north, stay at same strand, but keep track of crossings
                # with ramps and alpha
                if crossing_southwest_ramp(current_strand):
                    level -= 1
                if crossing_northwest_ramp(current_strand):
                    level -= 1
                if crossing_southeast_ramp(current_strand):
                    level += 1
                if crossing_northeast_ramp(current_strand):
                    level += 1
                if intersecting_alpha(current_strand):
                    polynomial[level] = polynomial.get(level, 0) - 1
                    num_crossings += 1
                    # Get rid of terms with zero coefficients
                    if polynomial[level] == 0:
                        del polynomial[level]
                at_north = True
    is_beta_connected = steps == 2 * num_strands - 1
    return polynomial, is_beta_connected, num_crossings


def calculate_polynomial(cap_west: int, cap_east: int,
                         cup_west: int, cup_east: int,
                         use_numba=True):
    if use_numba:
        return calculate_polynomial_impl(cap_west, cap_east,
                                         cup_west, cup_east)

    curve = Curve(cap_west, cap_east, cup_west, cup_east)
    curve.run_to_end()
    return curve.polynomial, curve.is_beta_connected(), curve.num_crossings
