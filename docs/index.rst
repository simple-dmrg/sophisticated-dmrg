==================
sophisticated-dmrg
==================

Source code: https://github.com/simple-dmrg/sophisticated-dmrg/

Documentation: http://sophisticated-dmrg.readthedocs.org/

This code is an expanded `density-matrix renormalization group
<http://en.wikipedia.org/wiki/Density_matrix_renormalization_group>`_
(DMRG) program, based on code written for a `tutorial
<http://simple-dmrg.readthedocs.org/>`_ given originally at the `2013
summer school on quantum spin liquids
<http://www.democritos.it/qsl2013/>`_, in Trieste, Italy.  It
implements DMRG its traditional formulation (i.e. without using matrix
product states).  DMRG is a numerical method that allows for the
efficient simulation of quantum model Hamiltonians.  Since it is a
low-entanglement approximation, it often works quite well for
one-dimensional systems, giving results that are nearly exact.

Typical implementations of DMRG in C++ or Fortran can be tens of
thousands of lines long.  Here, we have attempted to strike a balance
between clear, simple code, and including many features and
optimizations that would exist in a production code.  One thing that
helps with this is the use of `Python <http://www.python.org/>`_.  We
have tried to write the code in a very explicit style, hoping that it
will be (mostly) understandable to somebody new to Python.

Features
========

Beyond the features already existing in `simple-dmrg
<http://simple-dmrg.readthedocs.org/>`_ (infinite and finite system
algorithms, conserved abelian quantum numbers, and eigenstate
prediction), sophisticated-dmrg offers the following improvements:

- pluggable models

  - `Heisenberg XXZ
    <http://en.wikipedia.org/wiki/Heisenberg_model_(quantum)>`_
  - `Bose-Hubbard
    <http://en.wikipedia.org/wiki/Bose%E2%80%93Hubbard_model>`_

- choice between open or periodic boundary conditions

- measurements (assumes operators on different sites commute)

- site-dependent potential (e.g. to implement disorder)

Future features
===============

Planned and potential features
------------------------------

- use disk (not RAM) for persistent storage
- efficient representation of the Hamiltonian (if easily possible in python)
- time-dependent DMRG
- custom Lanczos
- fermions and fermionic Hubbard models
- models for ladder systems
- site-dependent hopping terms (e.g. to implement "hopping disorder")

Highly unlikely future features
-------------------------------

- rewrite in terms of matrix product states
- non-abelian symmetries (e.g. SU(2))

Authors
=======

- James R. Garrison (UCSB)
- Ryan V. Mishmash (UCSB)

Licensed under the MIT license.  If you plan to publish work based on
this code, please contact us to find out how to cite us.

Contents
========

.. toctree::
   :maxdepth: 2

   using
