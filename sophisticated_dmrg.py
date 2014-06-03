#!/usr/bin/env python
#
# "Sophisticated DMRG," originally based on the simple-dmrg tutorial.
#
# Copyright 2013 James R. Garrison and Ryan V. Mishmash.
# Open source under the MIT license.  Source code at
# <https://github.com/simple-dmrg/sophisticated-dmrg/>

# This code will run under any version of Python >= 2.6.  The following line
# provides consistency between python2 and python3.
from __future__ import print_function, division  # requires Python >= 2.6

import sys
from collections import namedtuple, Callable
from itertools import chain

import numpy as np
from scipy.sparse import kron, identity, lil_matrix
from scipy.sparse.linalg import eigsh  # Lanczos routine from ARPACK

open_bc = 0
periodic_bc = 1

# We will use python's "namedtuple" to represent the Block and EnlargedBlock
# objects
Block = namedtuple("Block", ["length", "basis_size", "operator_dict", "basis_sector_array"])
EnlargedBlock = namedtuple("EnlargedBlock", ["length", "basis_size", "operator_dict", "basis_sector_array"])

def is_valid_block(block):
    if len(block.basis_sector_array) != block.basis_size:
        return False
    for op in block.operator_dict.values():
        if op.shape[0] != block.basis_size or op.shape[1] != block.basis_size:
            return False
    return True

# This function should test the same exact things, so there is no need to
# repeat its definition.
is_valid_enlarged_block = is_valid_block

# Model-specific code for the Heisenberg XXZ chain
class HeisenbergXXZChain(object):
    dtype = 'd'  # double-precision floating point
    d = 2  # single-site basis size

    Sz1 = np.array([[0.5, 0], [0, -0.5]], dtype)  # single-site S^z
    Sp1 = np.array([[0, 1], [0, 0]], dtype)  # single-site S^+

    sso = {"Sz": Sz1, "Sp": Sp1, "Sm": Sp1.transpose()}  # single-site operators

    # S^z sectors corresponding to the single site basis elements
    single_site_sectors = np.array([0.5, -0.5])

    def __init__(self, J=1., Jz=None, hz=0., boundary_condition=open_bc):
        """
        `hz` can be either a number (for a constant magnetic field) or a
        callable (which is called with the site index and returns the
        magnetic field on that site).
        """
        if Jz is None:
            Jz = J
        self.J = J
        self.Jz = Jz
        self.boundary_condition = boundary_condition
        if isinstance(hz, Callable):
            self.hz = hz
        else:
            self.hz = lambda site_index: hz

    def H1(self, site_index):
        half_hz = .5 * self.hz(site_index)
        return np.array([[half_hz, 0], [0, -half_hz]], self.dtype)

    def H2(self, Sz1, Sp1, Sz2, Sp2):  # two-site part of H
        """Given the operators S^z and S^+ on two sites in different Hilbert spaces
        (e.g. two blocks), returns a Kronecker product representing the
        corresponding two-site term in the Hamiltonian that joins the two sites.
        """
        return (
            (self.J / 2) * (kron(Sp1, Sp2.conjugate().transpose()) +
                            kron(Sp1.conjugate().transpose(), Sp2)) +
            self.Jz * kron(Sz1, Sz2)
        )

    def initial_block(self, site_index):
        if self.boundary_condition == open_bc:
            # conn refers to the connection operator, that is, the operator on the
            # site that was most recently added to the block.  We need to be able
            # to represent S^z and S^+ on that site in the current basis in order
            # to grow the chain.
            operator_dict = {
                "H": self.H1(site_index),
                "conn_Sz": self.Sz1,
                "conn_Sp": self.Sp1,
            }
        else:
            # Since the PBC block needs to be able to grow in both directions,
            # we must be able to represent the relevant operators on both the
            # left and right sites of the chain.
            operator_dict = {
                "H": self.H1(site_index),
                "l_Sz": self.Sz1,
                "l_Sp": self.Sp1,
                "r_Sz": self.Sz1,
                "r_Sp": self.Sp1,
            }
        return Block(length=1, basis_size=self.d, operator_dict=operator_dict,
                     basis_sector_array=self.single_site_sectors)

    def enlarge_block(self, block, direction, bare_site_index):
        """This function enlarges the provided Block by a single site, returning an
        EnlargedBlock.
        """
        mblock = block.basis_size
        o = block.operator_dict

        # Create the new operators for the enlarged block.  Our basis becomes a
        # Kronecker product of the Block basis and the single-site basis.  NOTE:
        # `kron` uses the tensor product convention making blocks of the second
        # array scaled by the first.  As such, we adopt this convention for
        # Kronecker products throughout the code.
        if self.boundary_condition == open_bc:
            enlarged_operator_dict = {
                "H": kron(o["H"], identity(self.d)) +
                     kron(identity(mblock), self.H1(bare_site_index)) +
                     self.H2(o["conn_Sz"], o["conn_Sp"], self.Sz1, self.Sp1),
                "conn_Sz": kron(identity(mblock), self.Sz1),
                "conn_Sp": kron(identity(mblock), self.Sp1),
            }
        else:
            assert direction in ("l", "r")
            if direction == "l":
                enlarged_operator_dict = {
                    "H": kron(o["H"], identity(self.d)) +
                         kron(identity(mblock), self.H1(bare_site_index)) +
                         self.H2(o["l_Sz"], o["l_Sp"], self.Sz1, self.Sp1),
                    "l_Sz": kron(identity(mblock), self.Sz1),
                    "l_Sp": kron(identity(mblock), self.Sp1),
                    "r_Sz": kron(o["r_Sz"], identity(self.d)),
                    "r_Sp": kron(o["r_Sp"], identity(self.d)),
                }
            else:
                enlarged_operator_dict = {
                    "H": kron(o["H"], identity(self.d)) +
                         kron(identity(mblock), self.H1(bare_site_index)) +
                         self.H2(o["r_Sz"], o["r_Sp"], self.Sz1, self.Sp1),
                    "l_Sz": kron(o["l_Sz"], identity(self.d)),
                    "l_Sp": kron(o["l_Sp"], identity(self.d)),
                    "r_Sz": kron(identity(mblock), self.Sz1),
                    "r_Sp": kron(identity(mblock), self.Sp1),
                }

        # This array keeps track of which sector each element of the new basis is
        # in.  `np.add.outer()` creates a matrix that adds each element of the
        # first vector with each element of the second, which when flattened
        # contains the sector of each basis element in the above Kronecker product.
        enlarged_basis_sector_array = np.add.outer(block.basis_sector_array, self.single_site_sectors).flatten()

        return EnlargedBlock(length=(block.length + 1),
                             basis_size=(block.basis_size * self.d),
                             operator_dict=enlarged_operator_dict,
                             basis_sector_array=enlarged_basis_sector_array)

    def construct_superblock_hamiltonian(self, sys_enl, env_enl):
        sys_enl_op = sys_enl.operator_dict
        env_enl_op = env_enl.operator_dict
        if self.boundary_condition == open_bc:
            # L**R
            H_int = self.H2(sys_enl_op["conn_Sz"], sys_enl_op["conn_Sp"], env_enl_op["conn_Sz"], env_enl_op["conn_Sp"])
        else:
            assert self.boundary_condition == periodic_bc
            # L*R*
            H_int = (self.H2(sys_enl_op["r_Sz"], sys_enl_op["r_Sp"], env_enl_op["l_Sz"], env_enl_op["l_Sp"]) +
                     self.H2(sys_enl_op["l_Sz"], sys_enl_op["l_Sp"], env_enl_op["r_Sz"], env_enl_op["r_Sp"]))
        return (kron(sys_enl_op["H"], identity(env_enl.basis_size)) +
                kron(identity(sys_enl.basis_size), env_enl_op["H"]) +
                H_int)

# Model-specific code for the Bose-Hubbard chain
class BoseHubbardChain(object):
    dtype = 'd'  # double-precision floating point

    def __init__(self, d, U=0., mu=0., t=1., boundary_condition=open_bc):
        self.t = t  # hopping
        self.U = U  # on-site interaction
        self.boundary_condition = boundary_condition
        assert d >= 2
        self.d = d  # single-site basis size (enforces a maximum of d-1 particles on a site)
        ndiag = np.array(range(d), self.dtype)
        self.single_site_sectors = ndiag
        self.n_op = np.diag(ndiag)
        self.b_op = np.diag(np.sqrt(ndiag[1:]), k=1)  # k=1 => upper diagonal
        assert np.sum(np.abs(self.n_op - self.b_op.transpose().dot(self.b_op))) < 1e-4
        self.H1 = np.diag(.5 * U * ndiag * (ndiag - 1) - mu * ndiag)  # single-site term of H
        self.sso = {"n": self.n_op, "b": self.b_op}  # single-site operators

    def initial_block(self, site_index):
        if self.boundary_condition == open_bc:
            # conn refers to the connection operator, that is, the operator on the
            # site that was most recently added to the block.  We need to be able
            # to represent S^z and S^+ on that site in the current basis in order
            # to grow the chain.
            operator_dict = {
                "H": self.H1,
                "conn_n": self.n_op,
                "conn_b": self.b_op,
            }
        else:
            # Since the PBC block needs to be able to grow in both directions,
            # we must be able to represent the relevant operators on both the
            # left and right sites of the chain.
            operator_dict = {
                "H": self.H1,
                "l_n": self.n_op,
                "l_b": self.b_op,
                "r_n": self.n_op,
                "r_b": self.b_op,
             }
        return Block(length=1, basis_size=self.d, operator_dict=operator_dict,
                     basis_sector_array=self.single_site_sectors)

    def H2(self, b1, b2):  # two-site part of H
        """Given the operator b on two sites in different Hilbert spaces
        (e.g. two blocks), returns a Kronecker product representing the
        corresponding two-site term in the Hamiltonian that joins the two
        sites.
        """
        return -self.t * (kron(b1, b2.conjugate().transpose()) +
                          kron(b1.conjugate().transpose(), b2))

    def enlarge_block(self, block, direction, bare_site_index):
        """This function enlarges the provided Block by a single site, returning an
        EnlargedBlock.
        """
        mblock = block.basis_size
        o = block.operator_dict

        # Create the new operators for the enlarged block.  Our basis becomes a
        # Kronecker product of the Block basis and the single-site basis.  NOTE:
        # `kron` uses the tensor product convention making blocks of the second
        # array scaled by the first.  As such, we adopt this convention for
        # Kronecker products throughout the code.
        if self.boundary_condition == open_bc:
            enlarged_operator_dict = {
                "H": kron(o["H"], identity(self.d)) +
                     kron(identity(mblock), self.H1) +
                     self.H2(o["conn_b"], self.b_op),
                "conn_n": kron(identity(mblock), self.n_op),
                "conn_b": kron(identity(mblock), self.b_op),
            }
        else:
            assert direction in ("l", "r")
            if direction == "l":
                enlarged_operator_dict = {
                    "H": kron(o["H"], identity(self.d)) +
                         kron(identity(mblock), self.H1) +
                         self.H2(o["l_b"], self.b_op),
                    "l_n": kron(identity(mblock), self.n_op),
                    "l_b": kron(identity(mblock), self.b_op),
                    "r_n": kron(o["r_n"], identity(self.d)),
                    "r_b": kron(o["r_b"], identity(self.d)),
                }
            else:
                enlarged_operator_dict = {
                    "H": kron(o["H"], identity(self.d)) +
                         kron(identity(mblock), self.H1) +
                         self.H2(o["r_b"], self.b_op),
                    "l_n": kron(o["l_n"], identity(self.d)),
                    "l_b": kron(o["l_b"], identity(self.d)),
                    "r_n": kron(identity(mblock), self.n_op),
                    "r_b": kron(identity(mblock), self.b_op),
                }

        # This array keeps track of which sector each element of the new basis is
        # in.  `np.add.outer()` creates a matrix that adds each element of the
        # first vector with each element of the second, which when flattened
        # contains the sector of each basis element in the above Kronecker product.
        enlarged_basis_sector_array = np.add.outer(block.basis_sector_array, self.single_site_sectors).flatten()

        return EnlargedBlock(length=(block.length + 1),
                             basis_size=(block.basis_size * self.d),
                             operator_dict=enlarged_operator_dict,
                             basis_sector_array=enlarged_basis_sector_array)

    def construct_superblock_hamiltonian(self, sys_enl, env_enl):
        sys_enl_op = sys_enl.operator_dict
        env_enl_op = env_enl.operator_dict
        if self.boundary_condition == open_bc:
            # L**R
            H_int = self.H2(sys_enl_op["conn_b"], env_enl_op["conn_b"])
        else:
            assert self.boundary_condition == periodic_bc
            # L*R*
            H_int = (self.H2(sys_enl_op["r_b"], env_enl_op["l_b"]) +
                     self.H2(sys_enl_op["l_b"], env_enl_op["r_b"]))
        return (kron(sys_enl_op["H"], identity(env_enl.basis_size)) +
                kron(identity(sys_enl.basis_size), env_enl_op["H"]) +
                H_int)

def rotate_and_truncate(operator, transformation_matrix):
    """Transforms the operator to the new (possibly truncated) basis given by
    `transformation_matrix`.
    """
    return transformation_matrix.conjugate().transpose().dot(operator.dot(transformation_matrix))

def index_map(array):
    """Given an array, returns a dictionary that allows quick access to the
    indices at which a given value occurs.

    Example usage:

    >>> by_index = index_map([3, 5, 5, 7, 3])
    >>> by_index[3]
    [0, 4]
    >>> by_index[5]
    [1, 2]
    >>> by_index[7]
    [3]
    """
    d = {}
    for index, value in enumerate(array):
        d.setdefault(value, []).append(index)
    return d

def graphic(boundary_condition, sys_block, env_block, direction="r"):
    """Returns a graphical representation of the DMRG step we are about to
    perform, using '=' to represent the system sites, '-' to represent the
    environment sites, and '**' to represent the two intermediate sites.
    """
    order = {"r": 1, "l": -1}[direction]
    l_symbol, r_symbol = ("=", "-")[::order]
    l_length, r_length = (sys_block.length, env_block.length)[::order]

    if boundary_condition == open_bc:
        return (l_symbol * l_length) + "+*"[::order] + (r_symbol * r_length)
    else:
        return (l_symbol * l_length) + "+" + (r_symbol * r_length) + "*"

def bare_site_indices(boundary_condition, sys_block, env_block, direction):
    """Returns the site indices of the two bare sites: first the one in the
    enlarged system block, then the one in the enlarged environment block.
    """
    order = {"r": 1, "l": -1}[direction]
    l_block, r_block = (sys_block, env_block)[::order]
    if boundary_condition == open_bc:
        l_site_index = l_block.length
        r_site_index = l_site_index + 1
        sys_site_index, env_site_index = (l_site_index, r_site_index)[::order]
    else:
        sys_site_index = l_block.length
        env_site_index = l_block.length + r_block.length + 1

    g = graphic(boundary_condition, sys_block, env_block, direction)
    assert sys_site_index == g.index("+")
    assert env_site_index == g.index("*")

    return sys_site_index, env_site_index

def single_dmrg_step(model, sys, env, m, direction, target_sector=None, psi0_guess=None):
    """Perform a single DMRG step using `sys` as the system and `env` as the
    environment, keeping a maximum of `m` states in the new basis.  If
    `psi0_guess` is provided, it will be used as a starting vector for the
    Lanczos algorithm.
    """
    assert is_valid_block(sys)
    assert is_valid_block(env)

    # Determine the site indices of the two bare sites
    sys_site_index, env_site_index = bare_site_indices(model.boundary_condition, sys, env, direction)

    # Enlarge each block by a single site.
    sys_enl = model.enlarge_block(sys, direction, sys_site_index)
    sys_enl_basis_by_sector = index_map(sys_enl.basis_sector_array)
    env_enl = model.enlarge_block(env, direction, env_site_index)
    env_enl_basis_by_sector = index_map(env_enl.basis_sector_array)

    assert is_valid_enlarged_block(sys_enl)
    assert is_valid_enlarged_block(env_enl)

    m_sys_enl = sys_enl.basis_size
    m_env_enl = env_enl.basis_size

    # Construct the full superblock Hamiltonian.
    superblock_hamiltonian = model.construct_superblock_hamiltonian(sys_enl, env_enl)

    if target_sector is not None:
        # Build up a "restricted" basis of states in the target sector and
        # reconstruct the superblock Hamiltonian in that sector.
        sector_indices = {} # will contain indices of the new (restricted) basis
                            # for which the enlarged system is in a given sector
        restricted_basis_indices = []  # will contain indices of the old (full) basis, which we are mapping to
        for sys_enl_sector, sys_enl_basis_states in sys_enl_basis_by_sector.items():
            sector_indices[sys_enl_sector] = []
            env_enl_sector = target_sector - sys_enl_sector
            if env_enl_sector in env_enl_basis_by_sector:
                for i in sys_enl_basis_states:
                    i_offset = m_env_enl * i  # considers the tensor product structure of the superblock basis
                    for j in env_enl_basis_by_sector[env_enl_sector]:
                        current_index = len(restricted_basis_indices)  # about-to-be-added index of restricted_basis_indices
                        sector_indices[sys_enl_sector].append(current_index)
                        restricted_basis_indices.append(i_offset + j)

        if not restricted_basis_indices:
            raise RuntimeError("There are zero states in the restricted basis.")

        restricted_superblock_hamiltonian = superblock_hamiltonian[:, restricted_basis_indices][restricted_basis_indices, :]
        if psi0_guess is not None:
            restricted_psi0_guess = psi0_guess[restricted_basis_indices]
        else:
            restricted_psi0_guess = None

    else:
        # Our "restricted" basis is really just the original basis.  The only
        # thing to do is to build the `sector_indices` dictionary, which tells
        # which elements of our superblock basis correspond to a given sector
        # in the enlarged system.
        sector_indices = {}
        restricted_basis_indices = range(m_sys_enl * m_env_enl)
        for sys_enl_sector, sys_enl_basis_states in sys_enl_basis_by_sector.items():
            sector_indices[sys_enl_sector] = [] # m_env_enl
            for i in sys_enl_basis_states:
                sector_indices[sys_enl_sector].extend(range(m_env_enl * i, m_env_enl * (i + 1)))

        restricted_superblock_hamiltonian = superblock_hamiltonian
        restricted_psi0_guess = psi0_guess

    if restricted_superblock_hamiltonian.shape == (1, 1):
        restricted_psi0 = np.array([[1.]], dtype=model.dtype)
        energy = restricted_superblock_hamiltonian[0, 0]
    else:
        # Call ARPACK to find the superblock ground state.  ("SA" means find the
        # "smallest in amplitude" eigenvalue.)
        (energy,), restricted_psi0 = eigsh(restricted_superblock_hamiltonian, k=1, which="SA", v0=restricted_psi0_guess)

    # Construct each block of the reduced density matrix of the system by
    # tracing out the environment
    rho_block_dict = {}
    for sys_enl_sector, indices in sector_indices.items():
        if indices: # if indices is nonempty
            psi0_sector = restricted_psi0[indices, :]
            # We want to make the (sys, env) indices correspond to (row,
            # column) of a matrix, respectively.  Since the environment
            # (column) index updates most quickly in our Kronecker product
            # structure, psi0_sector is thus row-major ("C style").
            psi0_sector = psi0_sector.reshape([len(sys_enl_basis_by_sector[sys_enl_sector]), -1], order="C")
            rho_block_dict[sys_enl_sector] = np.dot(psi0_sector, psi0_sector.conjugate().transpose())

    # Diagonalize each block of the reduced density matrix and sort the
    # eigenvectors by eigenvalue.
    possible_eigenstates = []
    for sector, rho_block in rho_block_dict.items():
        evals, evecs = np.linalg.eigh(rho_block)
        current_sector_basis = sys_enl_basis_by_sector[sector]
        for eval, evec in zip(evals, evecs.transpose()):
            possible_eigenstates.append((eval, evec, sector, current_sector_basis))
    possible_eigenstates.sort(reverse=True, key=lambda x: x[0])  # largest eigenvalue first

    # Build the transformation matrix from the `m` overall most significant
    # eigenvectors.  It will have sparse structure due to the conserved quantum
    # number.
    my_m = min(len(possible_eigenstates), m)
    transformation_matrix = lil_matrix((sys_enl.basis_size, my_m), dtype=model.dtype)
    new_sector_array = np.zeros((my_m,), model.dtype)  # lists the sector of each
                                                       # element of the new/truncated basis
    for i, (eval, evec, sector, current_sector_basis) in enumerate(possible_eigenstates[:my_m]):
        for j, v in zip(current_sector_basis, evec):
            transformation_matrix[j, i] = v
        new_sector_array[i] = sector
    # Convert the transformation matrix to a more efficient internal
    # representation.  `lil_matrix` is good for constructing a sparse matrix
    # efficiently, but `csr_matrix` is better for performing quick
    # multiplications.
    transformation_matrix = transformation_matrix.tocsr()

    truncation_error = 1 - sum([x[0] for x in possible_eigenstates[:my_m]])
    print("truncation error", truncation_error)

    # Rotate and truncate each operator.
    new_operator_dict = {}
    for name, op in sys_enl.operator_dict.items():
        new_operator_dict[name] = rotate_and_truncate(op, transformation_matrix)

    newblock = Block(length=sys_enl.length,
                     basis_size=my_m,
                     operator_dict=new_operator_dict,
                     basis_sector_array=new_sector_array)

    # Construct psi0 (that is, in the full superblock basis) so we can use it
    # later for eigenstate prediction.
    psi0 = np.zeros([m_sys_enl * m_env_enl, 1], model.dtype)
    for i, z in enumerate(restricted_basis_indices):
        psi0[z, 0] = restricted_psi0[i, 0]
    assert np.all(psi0[restricted_basis_indices] == restricted_psi0)
    if psi0_guess is not None:
        overlap = np.absolute(np.dot(psi0_guess.conjugate().transpose(), psi0).item())
        overlap /= np.linalg.norm(psi0_guess) * np.linalg.norm(psi0)  # normalize it
        print("overlap |<psi0_guess|psi0>| =", overlap)

    return newblock, energy, transformation_matrix, psi0, restricted_basis_indices

def infinite_system_algorithm(model, L, m, target_sector=None):
    block = model.initial_block(0)
    # Repeatedly enlarge the system by performing a single DMRG step, using a
    # reflection of the current block as the environment.
    while 2 * block.length < L:
        current_L = 2 * block.length + 2  # current superblock length
        if target_sector is not None:
            # assumes the value is extensive
            current_target_sector = int(target_sector) * current_L // L
        else:
            current_target_sector = None
        print("L =", current_L)
        block, energy, transformation_matrix, psi0, rbi = single_dmrg_step(model, block, block, m=m, direction="r", target_sector=current_target_sector)
        print("E/L =", energy / current_L)

def finite_system_algorithm(model, L, m_warmup, m_sweep_list, target_sector=None, measurements=None):
    if not (L > 0 and L % 2 == 0):
        raise ValueError("System length `L` must be an even, positive number.")

    # To keep things simple, these dictionaries are not actually saved to disk,
    # but they are used to represent persistent storage.
    block_disk = {}  # "disk" storage for Block objects
    trmat_disk = {}  # "disk" storage for transformation matrices

    # Use the infinite system algorithm to build up to desired size.  Each time
    # we construct a block, we save it for future reference as both a left
    # ("l") and right ("r") block, as the infinite system algorithm assumes the
    # environment is a mirror image of the system.
    block = model.initial_block(0)
    assert block.length == 1
    block_disk["l", 1] = block
    while 2 * block.length < L:
        # Perform a single DMRG step and save the new Block to "disk"
        print(graphic(model.boundary_condition, block, block))
        current_L = 2 * block.length + 2  # current superblock length
        if target_sector is not None:
            current_target_sector = int(target_sector) * current_L // L
        else:
            current_target_sector = None
        block, energy, transformation_matrix, psi0, rbi = single_dmrg_step(model, block, block, m=m_warmup, direction="r", target_sector=current_target_sector)
        print("E/L =", energy / current_L)
        block_disk["l", block.length] = block
        block_disk["r", block.length] = block

    # Assuming a site-dependent Hamiltonian, the infinite system algorithm
    # above actually used the wrong superblock Hamiltonian, since the left
    # block was mirrored and used as the environment.  This mistake will be
    # fixed during the finite system algorithm sweeps below as long as we begin
    # with the correct initial block of the right-hand system.
    if model.boundary_condition == open_bc:
        right_initial_block_site_index = L - 1  # right-most site
    else:
        right_initial_block_site_index = L - 2  # second site from right
    block_disk["r", 1] = model.initial_block(right_initial_block_site_index)

    # Now that the system is built up to its full size, we perform sweeps using
    # the finite system algorithm.  At first the left block will act as the
    # system, growing at the expense of the right block (the environment), but
    # once we come to the end of the chain these roles will be reversed.
    sys_label, env_label = "l", "r"
    sys_block = block; del block  # rename the variable
    sys_trmat = None
    for m in m_sweep_list:
        print("Performing sweep with m =", m)
        while True:
            # Load the appropriate environment from "disk"
            env_block = block_disk[env_label, L - sys_block.length - 2]
            env_trmat = trmat_disk.get((env_label, L - sys_block.length - 1))

            # If possible, predict an estimate of the ground state wavefunction
            # from the previous step's psi0 and known transformation matrices.
            if psi0 is None or sys_trmat is None or env_trmat is None:
                psi0g = None
            else:
                # psi0 currently looks e.g. like ===**--- but we need to
                # transform it to look like ====**-- using the relevant
                # transformation matrices and paying careful attention to the
                # tensor product structure.
                #
                # Keep in mind that the tensor product of the superblock is
                # (sys_enl_block, env_enl_block), which is equal to
                # (sys_block, sys_extra_site, env_block, env_extra_site).
                # Note that this does *not* correspond to left-to-right order
                # on the chain.
                #
                # Also keep in mind that `trmat.shape` corresponds to
                # (old extended block size, new truncated block size).
                #
                # First we reshape the psi0 vector into a matrix with rows
                # corresponding to the enlarged system basis and columns
                # corresponding to the enlarged environment basis.
                psi0g = psi0.reshape((-1, env_trmat.shape[1] * model.d), order="C")
                # Now we transform the enlarged system block into a system
                # block, so that psi0g looks like ====*-- (with only one
                # intermediate site).
                psi0g = sys_trmat.conjugate().transpose().dot(psi0g)
                # At the moment, the tensor product goes as (sys_block,
                # env_enl_block) == (sys_block, env_block, extra_site), but we
                # need it to look like (sys_enl_block, env_block) ==
                # (sys_block, extra_site, env_block).  In other words, the
                # single intermediate site should now be part of a new enlarged
                # system, not part of the enlarged environment.
                psi0g = psi0g.reshape((-1, env_trmat.shape[1], model.d), order="C").transpose(0, 2, 1)
                # Now we reshape the psi0g vector into a matrix with rows
                # corresponding to the enlarged system and columns
                # corresponding to the environment block.
                psi0g = psi0g.reshape((-1, env_trmat.shape[1]), order="C")
                # Finally, we transform the environment block into the basis of
                # an enlarged block the so that psi0g has the tensor
                # product structure of ====**--.
                psi0g = env_trmat.dot(psi0g.transpose()).transpose()
                if model.boundary_condition != open_bc:
                    # All of the above logic still holds, but the bare sites
                    # are mixed up with each other, so we need to swap their
                    # positions in the tensor product space.
                    psi0g = psi0g.reshape((sys_trmat.shape[1], model.d, env_trmat.shape[0] // model.d, model.d), order="C").transpose(0, 3, 2, 1)

            if env_block.length == 1:
                # We've come to the end of the chain, so we reverse course.
                sys_block, env_block = env_block, sys_block
                sys_label, env_label = env_label, sys_label
                if psi0g is not None:
                    # Re-order the psi0_guess based on the new sys, env labels.
                    psi0g = psi0g.reshape((sys_trmat.shape[1] * model.d, env_trmat.shape[0]), order="C").transpose()

            if psi0g is not None:
                # Reshape into a column vector
                psi0g = psi0g.reshape((-1, 1), order="C")

            # Note that the direction of the DMRG step will always be the same
            # as the environment label.  If the system block is being enlarged
            # to the right, the environment block will be on the right.  If the
            # system block is being enlarged to the left, the environment block
            # must be on the right.  This is true for both open and periodic
            # boundary conditions:
            #
            # Open BC:
            #    ======+*------   (direction == "r")
            #    ------*+======   (direction == "l")
            # Periodic BC:
            #    ======+------*   (direction == "r")
            #    ------+======*   (direction == "l")
            direction = env_label

            # Perform a single DMRG step.
            print(graphic(model.boundary_condition, sys_block, env_block, direction=direction))
            sys_block, energy, sys_trmat, psi0, rbi = single_dmrg_step(model, sys_block, env_block, m=m, direction=direction, target_sector=target_sector, psi0_guess=psi0g)

            print("E/L =", energy / L)
            print("E   =", energy)
            sys.stdout.flush()

            # Save the block and transformation matrix from this step to disk.
            block_disk[sys_label, sys_block.length] = sys_block
            trmat_disk[sys_label, sys_block.length] = sys_trmat

            # Check whether we just completed a full sweep.
            if sys_label == "l" and 2 * sys_block.length == L:
                break  # escape from the "while True" loop

    if measurements is None:
        return

    if not m_sweep_list:
        raise RuntimeError("You must perform some sweeps in the finite system algorithm if you wish to make any measurements.")

    # figure out which sites are where
    LEFT_BLOCK, LEFT_SITE, RIGHT_BLOCK, RIGHT_SITE = 0, 1, 2, 3
    if model.boundary_condition == open_bc:
        sites_by_area = (
            range(0, L // 2 - 1),  # left block
            [L // 2 - 1],  # left bare site
            range(L // 2 + 1, L)[::-1],  # right block
            [L // 2],  # right bare site
        )
    else:
        # PBC algorithm
        sites_by_area = (
            range(0, L // 2 - 1),  # left block
            [L // 2 - 1],  # left bare site
            range(L // 2, L - 1)[::-1],  # right block
            [L - 1],  # right bare site
        )
    assert sum([len(z) for z in sites_by_area]) == L
    assert set(chain.from_iterable(sites_by_area)) == set(range(L))

    # from the above info, make a lookup table of what class each site is in so
    # we can look it up easily.
    site_class = [None] * L
    for i, heh in enumerate(sites_by_area):
        for site_index in heh:
            site_class[site_index] = i
    assert None not in site_class

    def canonicalize(meas_desc):
        # This performs a stable sort on operators by site (so we are assuming
        # that operators on different sites commute).  By doing this, we will
        # be able to combine operators ASAP as a block grows
        return sorted(meas_desc, key=lambda x: x[0])

    InnerObject = namedtuple("InnerObject", ["site_indices", "operator_names", "area"])

    class OuterObject(object):
        def __init__(self, site_indices, operator_names, area, built=False):
            self.obj = InnerObject(site_indices, operator_names, area)
            self.built = built

    # first make a list of which operators we need to consider on each site.
    # This way we won't have to look at every single operator each time we add
    # a site.
    #
    # we are doing at least a few others things here, too...
    #
    # We are also checking that each operator and site index is valid.  And we
    # are copying the measurements so that we can modify them.
    #
    # We also make note of which of the four blocks each site is in
    measurements_by_site = {}
    processed_measurements = []
    for meas_desc in measurements:
        site_indices = set()
        for site_index, operator_name in meas_desc:
            if operator_name not in model.sso:
                raise RuntimeError("Unknown operator: %s" % operator_name)
            if site_index not in range(L):
                raise RuntimeError("Unknown site index: %r (L = %r)" % (site_index, L))
            site_indices.add(site_index)
        measurement = [OuterObject((site_index,), (operator_name,),
                                   site_class[site_index], False)
                       for site_index, operator_name in canonicalize(meas_desc)]
        processed_measurements.append(measurement)
        for site_index in site_indices:
            measurements_by_site.setdefault(site_index, []).append(measurement)
    assert len(measurements) == len(processed_measurements)

    class Yay(object):
        def __init__(self, m):
            self.m = m
            self.refcnt = 1

    def handle_operators_on_site(site_index, area, some_dict, msize):
        for measurement in measurements_by_site[site_index]:
            # first replace each operator on this site by an actual operator
            for i, obj in enumerate(measurement):
                if obj.obj.site_indices[0] != site_index:
                    continue
                assert obj.built is False
                try:
                    some_dict[obj.obj].refcnt += 1
                except KeyError:
                    assert len(obj.obj.operator_names) == 1
                    sso = model.sso[obj.obj.operator_names[0]]
                    mat = kron(identity(msize), sso)
                    some_dict[obj.obj] = Yay(mat)
                obj.built = True

            # second, combine all operators that are possible to combine (and decref them in the process)
            for i in range(len(measurement) - 1)[::-1]:
                # attempt to combine i and i+1
                if measurement[i].built and measurement[i + 1].built and measurement[i].obj.area == measurement[i + 1].obj.area == area:
                    c1, c2 = measurement[i], measurement[i + 1]

                    o1 = some_dict[c1.obj]
                    o1.refcnt -= 1
                    if o1.refcnt == 0:
                        del some_dict[c1.obj]

                    o2 = some_dict[c2.obj]
                    o2.refcnt -= 1
                    if o2.refcnt == 0:
                        del some_dict[c2.obj]

                    c3 = OuterObject(tuple(chain(c1.obj[0], c2.obj[0])),
                                     tuple(chain(c1.obj[1], c2.obj[1])),
                                     area, True)
                    try:
                        some_dict[c3.obj].refcnt += 1
                    except KeyError:
                        some_dict[c3.obj] = Yay(o1.m.dot(o2.m))

                    measurement.pop(i + 1)
                    measurement[i] = c3

    g_some_dict = ({}, {}, {}, {})

    def build_block(area, block_label):
        some_dict = g_some_dict[area]
        assert not some_dict
        msize = 1

        for i, site_index in enumerate(sites_by_area[area]):
            # kronecker all the operators on the block
            for k, v in some_dict.items():
                v.m = kron(v.m, identity(model.d))

            handle_operators_on_site(site_index, area, some_dict, msize)

            if i == 0:
                msize = model.d
            else:
                # transform all the operators on the block
                trmat = trmat_disk[block_label, i + 1]
                for k, v in some_dict.items():
                    assert v.refcnt > 0
                    assert k.area == area
                    v.m = rotate_and_truncate(v.m, trmat)
                msize = trmat.shape[1]

        return msize

    # build up the left and right blocks
    lb_msize = build_block(LEFT_BLOCK, "l")
    rb_msize = build_block(RIGHT_BLOCK, "r")

    # build up the two bare sites
    handle_operators_on_site(sites_by_area[LEFT_SITE][0], LEFT_SITE, g_some_dict[LEFT_SITE], 1)
    handle_operators_on_site(sites_by_area[RIGHT_SITE][0], RIGHT_SITE, g_some_dict[RIGHT_SITE], 1)

    # loop through each operator, put everything together, and take the expectation value.
    rpsi0 = psi0[rbi]
    mm_orig = [
        identity(lb_msize),
        identity(model.d),
        identity(rb_msize),
        identity(model.d),
    ]
    returned_measurements = []
    for pm in processed_measurements:
        # Because we used the "canonicalize" function, there should be at most
        # one operator from each of the four areas.
        assert all([obj.built for obj in pm])
        st = set([obj.obj.area for obj in pm])
        assert len(st) == len(pm)

        # kron the operators together.
        mm = list(mm_orig) # initialize it with identity matrices
        for obj in pm:
            mm[obj.obj.area] = g_some_dict[obj.obj.area][obj.obj].m
        big_m = kron(kron(mm[0], mm[1]), kron(mm[2], mm[3]))

        # rewrite the big operator in the restricted basis of psi0
        if target_sector is not None:
            big_m = big_m.tocsr()[:, rbi][rbi, :] # FIXME: why not have it be CSR before?

        # take the expectation value wrt psi0
        ev = rpsi0.conjugate().transpose().dot(big_m.dot(rpsi0)).item()

        returned_measurements.append(ev)

    for measurement, ev in zip(measurements, returned_measurements):
        for site_index, operator in measurement:
            print("%s_{%d}" % (operator, site_index), end=" ")
        print("=", ev)

    return returned_measurements

if __name__ == "__main__":
    np.set_printoptions(precision=10, suppress=True, threshold=10000, linewidth=300)

    def run_sample_spin_chain(boundary_condition, L=20):
        model = HeisenbergXXZChain(J=1., Jz=1., boundary_condition=boundary_condition)
        measurements = ([[(i, "Sz")] for i in range(L)] +
                        [[(i, "Sz"), (j, "Sz")] for i in range(L) for j in range(L)] +
                        [[(i, "Sp"), (j, "Sm")] for i in range(L) for j in range(L)])
        finite_system_algorithm(model, L=L, m_warmup=10, m_sweep_list=[10, 20, 30, 40, 40], target_sector=0, measurements=measurements)

    def run_sample_bose_hubbard_chain(boundary_condition, L=20):
        model = BoseHubbardChain(d=4, U=0.5, mu=0.69, boundary_condition=boundary_condition)
        measurements = ([[(i, "n")] for i in range(L)] +
                        [[(i, "n"), (j, "n")] for i in range(L) for j in range(L)])
        finite_system_algorithm(model, L=L, m_warmup=10, m_sweep_list=[10, 20, 30, 40, 40], target_sector=L, measurements=measurements)

    run_sample_spin_chain(open_bc)
    #run_sample_spin_chain(periodic_bc)
    run_sample_bose_hubbard_chain(open_bc)
    #run_sample_bose_hubbard_chain(periodic_bc)
