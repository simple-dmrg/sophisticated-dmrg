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

from collections import namedtuple

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

    # S^z sectors corresponding to the single site basis elements
    single_site_sectors = np.array([0.5, -0.5])

    H1 = np.array([[0, 0], [0, 0]], dtype)  # single-site portion of H is zero

    def __init__(self, J=1., Jz=None, boundary_condition=open_bc):
        if Jz is None:
            Jz = J
        self.J = J
        self.Jz = Jz
        self.boundary_condition = boundary_condition

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

    def initial_block(self):
        if self.boundary_condition == open_bc:
            # conn refers to the connection operator, that is, the operator on the
            # site that was most recently added to the block.  We need to be able
            # to represent S^z and S^+ on that site in the current basis in order
            # to grow the chain.
            operator_dict = {
                "H": self.H1,
                "conn_Sz": self.Sz1,
                "conn_Sp": self.Sp1,
            }
        else:
            # Since the PBC block needs to be able to grow in both directions,
            # we must be able to represent the relevant operators on both the
            # left and right sites of the chain.
            operator_dict = {
                "H": self.H1,
                "l_Sz": self.Sz1,
                "l_Sp": self.Sp1,
                "r_Sz": self.Sz1,
                "r_Sp": self.Sp1,
            }
        return Block(length=1, basis_size=self.d, operator_dict=operator_dict,
                     basis_sector_array=self.single_site_sectors)

    def enlarge_block(self, block, direction=None):
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
                     self.H2(o["conn_Sz"], o["conn_Sp"], self.Sz1, self.Sp1),
                "conn_Sz": kron(identity(mblock), self.Sz1),
                "conn_Sp": kron(identity(mblock), self.Sp1),
            }
        else:
            assert direction in ("l", "r")
            if direction == "l":
                enlarged_operator_dict = {
                    "H": kron(o["H"], identity(self.d)) +
                         kron(identity(mblock), self.H1) +
                         self.H2(o["l_Sz"], o["l_Sp"], self.Sz1, self.Sp1),
                    "l_Sz": kron(identity(mblock), self.Sz1),
                    "l_Sp": kron(identity(mblock), self.Sp1),
                    "r_Sz": kron(o["r_Sz"], identity(self.d)),
                    "r_Sp": kron(o["r_Sp"], identity(self.d)),
                }
            else:
                enlarged_operator_dict = {
                    "H": kron(o["H"], identity(self.d)) +
                         kron(identity(mblock), self.H1) +
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

    def initial_block(self):
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

    def enlarge_block(self, block, direction=None):
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

def single_dmrg_step(model, sys, env, m, direction=None, target_sector=None, psi0_guess=None):
    """Perform a single DMRG step using `sys` as the system and `env` as the
    environment, keeping a maximum of `m` states in the new basis.  If
    `psi0_guess` is provided, it will be used as a starting vector for the
    Lanczos algorithm.
    """
    assert is_valid_block(sys)
    assert is_valid_block(env)

    # Enlarge each block by a single site.
    sys_enl = model.enlarge_block(sys, direction)
    sys_enl_basis_by_sector = index_map(sys_enl.basis_sector_array)
    if sys is env:  # no need to recalculate a second time
        env_enl = sys_enl
        env_enl_basis_by_sector = sys_enl_basis_by_sector
    else:
        env_enl = model.enlarge_block(env, direction)
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
    if psi0_guess is not None:
        overlap = np.absolute(np.dot(psi0_guess.conjugate().transpose(), psi0).item())
        overlap /= np.linalg.norm(psi0_guess) * np.linalg.norm(psi0)  # normalize it
        print("overlap |<psi0_guess|psi0>| =", overlap)

    return newblock, energy, transformation_matrix, psi0

def graphic(boundary_condition, sys_block, env_block, sys_label="l"):
    """Returns a graphical representation of the DMRG step we are about to
    perform, using '=' to represent the system sites, '-' to represent the
    environment sites, and '**' to represent the two intermediate sites.
    """
    order = {"l": 1, "r": -1}[sys_label]
    l_symbol, r_symbol = ("=", "-")[::order]
    l_length, r_length = (sys_block.length, env_block.length)[::order]

    if boundary_condition == open_bc:
        return (l_symbol * l_length) + "**" + (r_symbol * r_length)
    else:
        return (l_symbol * l_length) + "*" + (r_symbol * r_length) + "*"

def infinite_system_algorithm(model, L, m, target_sector=None):
    block = model.initial_block()
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
        block, energy, transformation_matrix, psi0 = single_dmrg_step(model, block, block, m=m, direction="r", target_sector=current_target_sector)
        print("E/L =", energy / current_L)

def finite_system_algorithm(model, L, m_warmup, m_sweep_list, target_sector=None):
    assert L % 2 == 0  # require that L is an even number

    # To keep things simple, these dictionaries are not actually saved to disk,
    # but they are used to represent persistent storage.
    block_disk = {}  # "disk" storage for Block objects
    trmat_disk = {}  # "disk" storage for transformation matrices

    # Use the infinite system algorithm to build up to desired size.  Each time
    # we construct a block, we save it for future reference as both a left
    # ("l") and right ("r") block, as the infinite system algorithm assumes the
    # environment is a mirror image of the system.
    block = model.initial_block()
    block_disk["l", block.length] = block
    block_disk["r", block.length] = block
    while 2 * block.length < L:
        # Perform a single DMRG step and save the new Block to "disk"
        print(graphic(model.boundary_condition, block, block))
        current_L = 2 * block.length + 2  # current superblock length
        if target_sector is not None:
            current_target_sector = int(target_sector) * current_L // L
        else:
            current_target_sector = None
        block, energy, transformation_matrix, psi0 = single_dmrg_step(model, block, block, m=m_warmup, direction="r", target_sector=current_target_sector)
        print("E/L =", energy / current_L)
        block_disk["l", block.length] = block
        block_disk["r", block.length] = block

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

            # Perform a single DMRG step.
            print(graphic(model.boundary_condition, sys_block, env_block, sys_label))
            sys_block, energy, sys_trmat, psi0 = single_dmrg_step(model, sys_block, env_block, m=m, direction=env_label, target_sector=target_sector, psi0_guess=psi0g)

            print("E/L =", energy / L)
            print("E   =", energy)
            sys.stdout.flush()

            # Save the block and transformation matrix from this step to disk.
            block_disk[sys_label, sys_block.length] = sys_block
            trmat_disk[sys_label, sys_block.length] = sys_trmat

            # Check whether we just completed a full sweep.
            if sys_label == "l" and 2 * sys_block.length == L:
                break  # escape from the "while True" loop

if __name__ == "__main__":
    np.set_printoptions(precision=10, suppress=True, threshold=10000, linewidth=300)

    #model = BoseHubbardChain(d=5, U=3., boundary_condition=periodic_bc)
    model = HeisenbergXXZChain(J=1., Jz=1., boundary_condition=open_bc)

    #infinite_system_algorithm(model, L=100, m=20, target_sector=0)
    finite_system_algorithm(model, L=20, m_warmup=10, m_sweep_list=[10, 20, 30, 40, 40], target_sector=None)
