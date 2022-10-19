Burau
=====

Methods for finding kernel elements of the :math:`B_4` Burau representation,
or helping to show that no non-trivial elements of the kernel exist.

Background
----------
Here, we follow the lead from Bigelow [0]_ and consider a particular family of
pairs of curves in the four-punctured disc. We follow more or less the same
prescription as in Bigelow's C implementation currently available at::

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

As an example, suppose ``cap_west = 2``, ``cap_east = 1``, ``cup_west = 3``,
and ``cup_east = 2``. Then, joining the north and south halves of the picture,
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

Our goal is to calculate, in the notation of Bigelow [0]_,
:math:`\int_\beta \alpha`. As such, we follow :math:`\beta` from the northwest
puncture to the northeast puncture, along the way keeping track of
intersections with :math:`\alpha`. Each intersection contributes a summand
:math:`\pm t^k`, in which the sign of the coefficient is determined by whether
we intersect from the north or the south, and whose power is determined by the
current "level". To determine this power, we picture our four-punctured disk as
the vertical projection of a parking garage that extends infinitely up and down
(see Wikipedia [1]_ for an illustration), with a copy of :math:`\alpha` living
in each level. We start at level 0 of the garage, and as we move along, we may
encounter four "ramps" that take us between different levels::

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

The package can be installed from PyPI::

    pip install burau

or it can be obtained from `conda-forge <https://anaconda.org/conda-forge/burau>`_::

    mamba install -c conda-forge burau

The above example can be reproduced using the functionality of this Python
module as follows:

>>> from burau.curve import calculate_polynomial
>>> calculate_polynomial(cap_west=2, cap_east=1, cup_west=3, cup_east=2)
(DictType[int64,int64]<iv=None>({0: 1, -2: 1, -4: 1, -8: -1, -10: -1}), True, 5)

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
.. [1] https://en.wikipedia.org/wiki/Burau_representation..
