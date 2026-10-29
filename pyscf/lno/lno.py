#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Hong-Zhou Ye <hzyechem@gmail.com>
#

''' Generic framework for spin-restricted local natural orbital (LNO)-based methods.
    This code can be used to implement LNO-based local correlation approximation to
    many correlated wavefunction methods. See `lnoccsd.py` for the implementation of
    LNO-CCSD as an example.

    - Original publication by Kállay and co-workers:
        Rolik and Kállay, J. Chem. Phys. 135, 104111 (2011)

    - Publication for this implementation by Ye and Berkelbach:
        Ye and Berkelbach, J. Chem. Theory Comput. 2024, 20, 20, 8948–8959
'''

import os
import sys
import numbers
from collections.abc import Iterable
from functools import reduce

import numpy as np
import h5py

from pyscf import mp
from pyscf.lib import logger
from pyscf import lib
from pyscf import __config__

from pyscf.lno.make_lno_rdm1 import make_lo_rdm1_occ, make_lo_rdm1_vir
from pyscf.lno import rpa
from pyscf.lno import bws2
from pyscf.lno import domain
from pyscf.lno.tools import df_loop

einsum = lib.einsum

DEBUG_BLKSIZE = getattr(__config__, 'lnocc_DEBUG_BLKSIZE', False)


MUTE_CHKFILE = getattr(__config__, 'lno_LNO_mute_chkfile', False)


def kernel(mlno, lo_coeff, frag_lolist, lno_type, lno_thresh=None, lno_pct_occ=None,
           lno_norb=None, eris=None):
    r''' Kernel function for LNO-based methods.

    The LNO active spaces of all fragments are built first, then all impurity problems
    are solved. The full-system integrals used for the LNOs (`eris`) are released in
    between unless the impurity problems need them (see `LNO._keep_lno_eris`).

    Fragments whose LNOs (`mlno.frag_lno`) or energies (`mlno.frag_eners`) are already
    known, e.g. from `mlno.load_chk`, are not recomputed: a fragment with energies is
    skipped, and a fragment with LNOs only has its impurity problem solved. Both are
    saved to `mlno.chkfile` as each fragment finishes.

    Args:
        lo_coeff (np.ndarray):
            Column vectors are the AO coefficients for a set of local(ized) orbitals.
            These LOs must span at least the occupied space but can span none, part, or
            full of the virtual space. Thus, lo_coeff.shape[1] >= nmo.
        frag_lolist (list of list):
            Fragment definition in terms of the LOs specified by 'lo_coeff'. E.g.,
                [[0,1,2],[3,5],[4],[6,7,8,9]...]
            means
                fragment 1 consists of LO 0, 1, and 2,
                fragment 2 consists of LO 3 and 5,
                fragment 3 consists of LO 4,
                fragment 4 consists of LO 6, 7, 8, and 9,
                ...
    '''
    nfrag = len(frag_lolist)
    if lno_pct_occ is None:
        lno_pct_occ = [None, None]
    if lno_norb is None:
        lno_norb = [[None,None]] * nfrag

    mf = mlno._scf

    log = logger.new_logger(mlno)

    cput0 = (logger.process_clock(), logger.perf_counter())

    # restart data
    frag_lno = [None] * nfrag if mlno.frag_lno is None else list(mlno.frag_lno)
    frag_eners = [None] * nfrag if mlno.frag_eners is None else list(mlno.frag_eners)
    if len(frag_lno) != nfrag or len(frag_eners) != nfrag:
        log.error('frag_lno and frag_eners must have one entry per fragment (%d).', nfrag)
        raise ValueError
    frag_res = [None] * nfrag
    for ifrag,dic in enumerate(frag_eners):
        if dic is not None:
            frag_res[ifrag] = mlno._frag_res_from_dict(dic)
            if frag_res[ifrag] is None:
                log.info('Fragment %d/%d  energies incomplete; the fragment is recomputed',
                         ifrag+1, nfrag)
    mlno._chk_init(lo_coeff, frag_lolist, frag_lno, frag_eners)

    # full-system low-level amplitudes; the LNOs below solve these themselves if needed
    frag_todo_las = [ifrag for ifrag in range(nfrag)
                     if frag_res[ifrag] is None and frag_lno[ifrag] is None]
    need_rpa = ((mlno._lno_uses_rpa or mlno._pt2_uses_rpa) and
                (mlno.e_corr_rpa_full is None or
                 (mlno._pt2_amp == 'sosex' and mlno.e_corr_sosex_full is None)))
    need_bws2 = ('bws2' in (mlno._lno_amp, mlno._pt2_amp) and mlno.e_corr_bws2_full is None)
    if frag_todo_las or need_rpa or need_bws2:
        if eris is None: eris = mlno.ao2mo()
        if need_rpa: mlno.solve_rpa(eris)
        if need_bws2: mlno.solve_bws2(eris)
        mlno._chk_dump_full()

    cput2 = cput1 = (logger.process_clock(), logger.perf_counter())

    # Step 1: LNO active spaces of all fragments
    for ifrag in frag_todo_las:
        orbloc, lno_param = _frag_orbloc(lo_coeff, frag_lolist[ifrag], ifrag, lno_thresh,
                                         lno_pct_occ, lno_norb)

        lnofrag, frozen, uocc_loc, frag_msg = mlno.make_las(eris, orbloc, lno_type, lno_param)
        frag_lno[ifrag] = mlno._pack_las(lnofrag, frozen)
        mlno._chk_dump_frag(ifrag, 'las', dict(orbloc=orbloc, **frag_lno[ifrag]))
        cput2 = log.timer('Fragment %d make las'%(ifrag+1), *cput2)
        log.info('Fragment %d/%d  LAS: %s', ifrag+1, nfrag, frag_msg)
    lnofrag = frozen = uocc_loc = None

    # the integrals for the LNOs, and the dressed ones of BW-s2, are not needed anymore
    mlno._bws2 = None
    if not mlno._keep_lno_eris:
        if eris is not None:
            log.info('Releasing the full-system integrals used for the LNOs')
        eris = None

    # Step 2: impurity problems of all fragments
    for ifrag,loidx in enumerate(frag_lolist):
        if frag_res[ifrag] is not None:
            log.info('Fragment %d/%d  Sol: restored', ifrag+1, nfrag)
            continue

        orbloc = _frag_orbloc(lo_coeff, loidx, ifrag)[0]
        lnofrag, frozen, uocc_loc = mlno._unpack_las(frag_lno[ifrag], orbloc)

        # the RPA/SOSEX fragment energies project the full-system integrals
        if eris is None and mlno._pt2_uses_rpa: eris = mlno.ao2mo()

        # calculate domains/prune basis
        if mlno.prune_lno_basis:
            # compute domain
            mf_frag, lnofrag, uocc_loc, eris, frozen = \
                domain.prune_lno_basis(mf, lnofrag, orbloc, uocc_loc, eris,
                                   frozen, bp_thr=mlno.lno_basis_thresh)

            log.info('Domain: %d/%d basis functions', mf_frag.mol.nao,
                     mf.mol.nao)

            # set properties
            mlno._scf = mf_frag
            mlno._mo_occ = mf_frag.mo_occ
            mlno._s1e = mlno._scf.get_ovlp()
            mlno._h1e = None
            mlno._vhf = None
            mlno._precompute()
            mlno.lnofrag=lnofrag.copy()

        else:
            mf_frag = mf

        # solve impurity problem
        frag_res[ifrag], frag_msg = mlno.impurity_solve(mf_frag, lnofrag, uocc_loc, eris, frozen=frozen)
        frag_eners[ifrag] = mlno._frag_res_to_dict(frag_res[ifrag])
        mlno._chk_dump_frag(ifrag, 'energy', frag_eners[ifrag])
        cput2 = log.timer('Fragment %d imp sol '%(ifrag+1), *cput2)
        log.info('Fragment %d/%d  Sol: %s', ifrag+1, nfrag, frag_msg)
        cput1 = log.timer('Fragment %d'%(ifrag+1)+' '*(8-len(str(ifrag+1))), *cput1)

        # reinitialize basis for active-space calc
        if mlno.prune_lno_basis:
            # reset properties
            mlno._scf = mf
            mlno._mo_occ = None
            mlno._s1e = None
            mlno._h1e = None
            mlno._vhf = None
            mlno._precompute()

    classname = mlno.__class__.__name__
    cput0 = log.timer(classname+' '*(17-len(classname)), *cput0)

    return frag_res

def _frag_orbloc(lo_coeff, loidx, ifrag, lno_thresh=None, lno_pct_occ=None, lno_norb=None):
    r''' LOs of a fragment and, if `lno_thresh` is given, its LNO truncation parameters. '''
    if len(loidx) == 2 and isinstance(loidx[0], Iterable): # Unrestricted
        orbloc = [lo_coeff[0][:,loidx[0]], lo_coeff[1][:,loidx[1]]]
        if lno_thresh is None:
            return orbloc, None
        lno_param = [
            [
                {
                    'thresh': (
                        lno_thresh[i][s] if isinstance(lno_thresh[i], Iterable)
                        else lno_thresh[i]
                    ),
                    'pct_occ': (
                        lno_pct_occ[i][s] if isinstance(lno_pct_occ[i], Iterable)
                        else lno_pct_occ[i]
                    ),
                    'norb': (
                        lno_norb[ifrag][i][s] if isinstance(lno_norb[ifrag][i], Iterable)
                        else lno_norb[ifrag][i]
                    ),
                } for i in [0, 1]
            ] for s in range(2)
        ]

    else:
        orbloc = lo_coeff[:,loidx]
        if lno_thresh is None:
            return orbloc, None
        lno_param = [{'thresh': lno_thresh[i], 'pct_occ': lno_pct_occ[i],
                      'norb': lno_norb[ifrag][i]} for i in [0,1]]
    return orbloc, lno_param

def split_las(orbfrag, frozen, nocc):
    r''' Active occupied and virtual LNOs of an LAS returned by `make_las`, whose first
    `nocc` columns are the (frozen and active) occupied orbitals.
    '''
    nmo = orbfrag.shape[1]
    maskact = np.ones(nmo, dtype=bool)
    if isinstance(frozen, numbers.Integral):
        maskact[:frozen] = False
    else:
        maskact[np.asarray(frozen, dtype=int)] = False
    maskocc = np.arange(nmo) < nocc
    return orbfrag[:,maskact&maskocc], orbfrag[:,maskact&~maskocc]

def join_las(lno_occ, lno_vir, mo_coeff, mo_occ, s1e, orbloc):
    r''' Inverse of `split_las`: the LAS (orbfrag, frozen, uocc_loc) as returned by
    `make_las` from the active occupied and virtual LNOs. The frozen orbitals are the
    complements of the LNOs in the occupied and virtual MO spaces; any rotation among
    them gives the same impurity problem.
    '''
    maskocc = mo_occ > 1e-10
    orboccfrz = _orth_complement(mo_coeff[:,maskocc], lno_occ, s1e)
    orbvirfrz = _orth_complement(mo_coeff[:,~maskocc], lno_vir, s1e)
    orbfragall = [orboccfrz, lno_occ, lno_vir, orbvirfrz]
    orbfrag = np.hstack(orbfragall)
    locfragall = np.cumsum([0] + [x.shape[1] for x in orbfragall]).astype(int)
    frzfrag = np.concatenate((
        np.arange(locfragall[0], locfragall[1]),
        np.arange(locfragall[3], locfragall[4]))).astype(int)
    if len(frzfrag) == 0:
        frzfrag = 0
    uoccact_loc = np.linalg.multi_dot((lno_occ.T.conj(), s1e, orbloc))
    return orbfrag, frzfrag, uoccact_loc

def _orth_complement(orb, orbact, s1e):
    r''' Orbitals spanning the complement of `orbact` within the space of `orb`. '''
    nact = orbact.shape[1]
    if nact == 0:
        return orb
    u = np.linalg.multi_dot((orb.T.conj(), s1e, orbact))
    return np.dot(orb, np.linalg.svd(u)[0][:,nact:])

def _same_orbloc(orbloc1, orbloc2):
    if isinstance(orbloc2, (list, tuple)):    # Unrestricted
        return all(_same_orbloc(x1, x2) for x1,x2 in zip(orbloc1, orbloc2))
    return np.shape(orbloc1) == np.shape(orbloc2) and np.allclose(orbloc1, orbloc2, atol=1e-8)


def make_las(mlno, eris, orbloc, lno_type, lno_param):
    log = logger.new_logger(mlno)
    cput1 = (logger.process_clock(), logger.perf_counter())

    s1e = mlno.s1e

    orboccfrz_core, orbocc, orbvir, orbvirfrz_core = mlno.split_mo_coeff()
    moeocc, moevir = mlno.split_mo_energy()[1:3]

    ''' Projection of LO onto occ and vir
    '''
    uocc_loc = reduce(np.dot, (orbloc.T.conj(), s1e, orbocc))
    uocc_loc, uocc_std, uocc_orth = \
            projection_construction(uocc_loc, mlno.lo_proj_thresh, mlno.lo_proj_thresh_active)
    if uocc_loc.shape[1] == 0:
        log.error('LOs do not overlap with occupied space. This could be caused '
                  'by either a bad fragment choice or too high of `lo_proj_thresh_active` '
                  '(current value: %s).', mlno.lo_proj_thresh_active)
        raise RuntimeError
    log.info('LO occ proj: %d active | %d standby | %d orthogonal',
             *[u.shape[1] for u in [uocc_loc,uocc_std,uocc_orth]])

    uvir_loc = reduce(np.dot, (orbloc.T.conj(), s1e, orbvir))
    uvir_loc, uvir_std, uvir_orth = \
            projection_construction(uvir_loc, mlno.lo_proj_thresh, mlno.lo_proj_thresh_active)
    log.info('LO vir proj: %d active | %d standby | %d orthogonal',
             *[u.shape[1] for u in [uvir_loc,uvir_std,uvir_orth]])
    if uvir_loc.shape[1] == 0:
        uvir_loc = uvir_std = uvir_orth = None

    ''' LNO construction
    '''
    dmoo = mlno.make_lo_rdm1_occ(eris, moeocc, moevir, uocc_loc, uvir_loc, lno_type[0])
    if mlno._match_oldcode: dmoo *= 0.5 # TO MATCH OLD LNO CODE
    dmoo = reduce(np.dot, (uocc_orth.T.conj(), dmoo, uocc_orth))
    if lno_param[0]['norb'] is not None:
        lno_param[0]['norb'] -= uocc_loc.shape[1] + uocc_std.shape[1]
    uoccact_orth, uoccfrz_orth = natorb_select(dmoo, uocc_orth, **lno_param[0])
    orboccfrz = np.hstack((orboccfrz_core, np.dot(orbocc, uoccfrz_orth)))
    uoccact = subspace_eigh(np.diag(moeocc), np.hstack((uoccact_orth, uocc_std, uocc_loc)))[1]
    orboccact = np.dot(orbocc, uoccact)
    uoccact_loc = np.linalg.multi_dot((orboccact.T.conj(), s1e, orbloc))
    cput1 = log.timer_debug1('make_lo_rdm1_occ', *cput1)

    dmvv = mlno.make_lo_rdm1_vir(eris, moeocc, moevir, uocc_loc, uvir_loc, lno_type[1])
    if mlno._match_oldcode: dmvv *= 0.5 # TO MATCH OLD LNO CODE
    if uvir_orth is not None:
        dmvv = reduce(np.dot, (uvir_orth.T.conj(), dmvv, uvir_orth))
        if lno_param[1]['norb'] is not None:
            lno_param[1]['norb'] -= uvir_loc.shape[1] + uvir_std.shape[1]
        uviract_orth, uvirfrz_orth = natorb_select(dmvv, uvir_orth, **lno_param[1])
        orbvirfrz = np.hstack((np.dot(orbvir, uvirfrz_orth), orbvirfrz_core))
        uviract = subspace_eigh(np.diag(moevir), np.hstack((uviract_orth, uvir_std, uvir_loc)))[1]
        orbviract = np.dot(orbvir, uviract)
    else:
        orbviract, orbvirfrz = natorb_select(dmvv, orbvir, **lno_param[1])
        orbvirfrz = np.hstack((orbvirfrz, orbvirfrz_core))
        uviract = reduce(np.dot, (orbvir.T.conj(), s1e, orbviract))
        uviract = subspace_eigh(np.diag(moevir), uviract)[1]
        orbviract = np.dot(orbvir, uviract)
    cput1 = log.timer_debug1('make_lo_rdm1_vir', *cput1)

    ''' LAS construction
    '''
    orbfragall = [orboccfrz, orboccact, orbviract, orbvirfrz]
    orbfrag = np.hstack(orbfragall)
    norbfragall = np.asarray([x.shape[1] for x in orbfragall])
    locfragall = np.cumsum([0] + norbfragall.tolist()).astype(int)
    frzfrag = np.concatenate((
        np.arange(locfragall[0], locfragall[1]),
        np.arange(locfragall[3], locfragall[4]))).astype(int)
    frag_msg = '%d/%d Occ | %d/%d Vir | %d/%d MOs' % (
                    norbfragall[1], sum(norbfragall[:2]),
                    norbfragall[2], sum(norbfragall[2:4]),
                    sum(norbfragall[1:3]), sum(norbfragall)
                )
    if len(frzfrag) == 0:
        frzfrag = 0

    return orbfrag, frzfrag, uoccact_loc, frag_msg

def projection_construction(M, thresh, thresh_act=None):
    r''' Given M_{mu,i} = <mu | i> the ovlp between two orthonormal basis, find
    the unitary rotation |j'> = u_ij |i> so that {|j'>} significantly ovlp with
    {|mu>}.

    Three subsets will be returned:
        active  : singular value >  thresh_act
        standby : singular value <= thresh_act but > thresh
        frozen  : singular value <= thresh
    '''
    n, m = M.shape
    e, u = np.linalg.eigh(np.dot(M.T.conj(), M))
    if thresh_act is None: thresh_act = thresh
    assert( thresh_act >= thresh )
    mask_act = abs(e) > thresh_act
    mask_std = np.logical_and(abs(e) > thresh, ~mask_act)
    mask_frz = abs(e) <= thresh
    return u[:,mask_act], u[:,mask_std], u[:,mask_frz]

def subspace_eigh(fock, orb):
    f = reduce(np.dot, (orb.T.conj(), fock, orb))
    if orb.shape[1] == 1:
        moe = np.array([f[0,0]])
    else:
        moe, u = np.linalg.eigh(f)
        orb = np.dot(orb, u)
    return moe, orb

def natorb_select(dm, orb, thresh, pct_occ=None, norb=None):
    e, u = np.linalg.eigh(dm)
    e = abs(e)
    order = np.argsort(e)[::-1]
    e = e[order]
    u = u[:,order]
    if norb is None:
        if pct_occ is None:
            nkeep = np.count_nonzero(e > thresh)
        else:
            nkeep = np.count_nonzero(np.cumsum(e)/np.sum(e) <= pct_occ)
    else:
        nkeep = min(max(norb, 0), e.size)

    idx  = np.arange(0,     nkeep,  dtype=int)
    idxc = np.arange(nkeep, e.size, dtype=int)
    orbx = np.dot(orb, u)
    orb1x = sub_colspace(orbx, idx)
    orb0x = sub_colspace(orbx, idxc)
    return orb1x, orb0x

def sub_colspace(A, idx):
    if idx.size == 0:
        return np.zeros([A.shape[0],0])
    else:
        return A[:,idx]

def get_fragment_energy(oovv, t2, uloc):
    m = np.dot(uloc, uloc.T.conj())
    ed =  einsum('ijab,kjab,ik->', t2, oovv, m) * 2
    ex = -einsum('ijab,kjba,ik->', t2, oovv, m)
    ed = ed.real
    ex = ex.real
    ess = ed*0.5 + ex
    eos = ed*0.5
    return lib.tag_array(ess+eos, spin_comp=np.array((ess, eos)))


class LNO(lib.StreamObject):

    r''' Base class for LNO-based methods

    This base class provides common functions for constructing LNO subspace.
    Specific LNO-based methods (e.g., LNO-CCSD, LNO-CCSD(T)) can be implemented as
    derived classes from this base class with appropriately defined method
    `impurity_solve`.

    Input:
        mf (PySCF SCF object):
            Mean-field object.
        lo_coeff (np.ndarray):
            AO coefficient matrix of LOs. LOs must span the occupied space.
        frag_lolist (nested list):
            Fragment assignment in terms of LO index. E.g., [[0,2], [1], ...] means
            frag 1 consists of LO 0 and 2, frag 2 consists of LO 1, etc.
        lno_type (len-2 list):
            lno_type = [occ_lno_type, vir_lno_type], where 'occ_lno_type' can be
            '1h', '1p', or '2p' and 'vir_lno_type' can be '1p', '1h', '2h'.
            Default is ['1h','1h'].
        lno_thresh (float of len-2 list):
            Thresholds for LNO truncation. Use a len-2 list to specify thresh for
            occ and vir separately. Default is [1e-5,1e-6].
        frozen (int or list):
            Same as the `frozen` attr in MP2/CCSD etc. modules.
    '''

    def __init__(self, mf, lo_coeff, frag_lolist, lno_type=None, lno_thresh=None, frozen=None):

        self.mol = mf.mol
        self._scf = mf
        if hasattr(self._scf, 'with_df'):
            self.with_df = self._scf.with_df
        else:
            self.with_df = None
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.lo_coeff = lo_coeff
        self.frag_lolist = frag_lolist
        self.frozen = frozen

        # for LNO construction
        self.lno_type = ['1h','1h'] if lno_type is None else lno_type
        self.lno_thresh = [1e-5, 1e-6] if lno_thresh is None else lno_thresh
        self.lno_pct_occ = None
        self.lno_norb = None
        self.lo_proj_thresh = 1e-10
        self.lo_proj_thresh_active = 0.1

        # extra parameters
        self.frag_wghtlist = None
        self.verbose_imp = 0 # allow separate verbose level for `impurity_solve`

        # domain parameters
        self.prune_lno_basis = False  # whether or not to use domains
        self.lno_basis_thresh = 0.02  # default Boughton-Pulay parameter

        # low-level amplitudes
        #   lno_amp : amplitudes for the LNO density matrices
        #             'mp2', 'kappa-mp2', 'bws2', 'rpa' or 'sosex' ('rpa' and 'sosex'
        #             both use the drCCD amplitudes; they differ only in the energy)
        #   pt2_amp : amplitudes for the fragment PT2 energy and the impurity CC guess;
        #             None uses lno_amp, otherwise one of the lno_amp choices
        self.lno_amp = 'mp2'
        self.pt2_amp = None
        self.kappa = 1.1        # kappa-MP2 regularization strength, arXiv:2508.15744
        self.alpha = 1.0        # BW-s2(alpha) regularizer strength, arXiv:2309.01376
        # BW-s2 solver; full-system energy (per unit cell for KLNO) set when solved
        self.bws2_max_cycle = 50
        self.bws2_conv_tol = 1e-8
        self.e_corr_bws2_full = None
        self._bws2 = None
        # drCCD solver, arXiv:2601.00131
        self.rpa_cd_thresh = 1e-6   # Cholesky threshold for the drCCD energy denominator
        self.rpa_max_cycle = 100
        self.rpa_conv_tol = 1e-8
        self.rpa_conv_tol_normu = 1e-6
        self.rpa_diis_space = 8
        self.rpa_damp = 0.
        # full-system drCCD energies (per unit cell for KLNO), set when drCCD is solved
        self.e_corr_rpa_full = None
        self.e_corr_sosex_full = None   # only if pt2_amp resolves to 'sosex'
        self._rpa_amp = None

        # df eri
        self._ovL = None
        self._ovL_to_save = None
        self.force_outcore_ao2mo = False

        # checkpoint and restart
        #   chkfile    : the LOs and LNOs of each fragment are saved as they are built,
        #                and its energies as its impurity problem is solved. As for SCF,
        #                the default file is removed with the object; assign a filename
        #                to keep it.
        #   frag_lno   : restart input, per fragment None or its active LNOs
        #                {'lno_occ', 'lno_vir'} (lists over spin for ULNO); given LNOs
        #                are not rebuilt
        #   frag_eners : restart input, per fragment None or its energies (see
        #                `_frag_res_to_dict`); fragments with complete energies are skipped
        #   `load_chk` sets both from a chkfile.
        if MUTE_CHKFILE:
            self.chkfile = None
        else:
            self._chkfile = lib.NamedTemporaryFile(dir=lib.param.TMPDIR)
            self.chkfile = self._chkfile.name
        self.frag_lno = None
        self.frag_eners = None
        self._chk_loaded = None

        # reverse compatibility
        self._match_oldcode = False # if True, MP2 dm for LNO generation is multiplied by 0.5

        # Not input options
        self._nmo = None
        self._nocc = None
        self._s1e = None

        self._mo_occ = None
        self._mo_coeff = None
        self._mo_energy = None

    @property
    def nfrag(self):
        return len(self.frag_lolist)

    @property
    def s1e(self):
        if self._s1e is None:
            self._s1e = self._scf.get_ovlp()
        return self._s1e

    @property
    def mo_occ(self):
        if self._mo_occ is None:
            return self._scf.mo_occ
        else:
            return self._mo_occ
    @mo_occ.setter
    def mo_occ(self, x):
        self._mo_occ = x

    @property
    def mo_coeff(self):
        if self._mo_coeff is None:
            return self._scf.mo_coeff
        else:
            return self._mo_coeff
    @mo_coeff.setter
    def mo_coeff(self, x):
        self._mo_coeff = x

    @property
    def mo_energy(self):
        if self._mo_energy is None:
            return self._scf.mo_energy
        else:
            return self._mo_energy
    @mo_energy.setter
    def mo_energy(self, x):
        self._mo_energy = x

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('nocc = %s, nmo = %s', self.nocc, self.nmo)
        if self.frozen is not None:
            log.info('frozen orbitals %s', self.frozen)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        try:
            nlo = self.lo_coeff.shape[1]
        except AttributeError:
            nlo = [self.lo_coeff[0].shape[1], self.lo_coeff[1].shape[1]]
        log.info('nfrag = %d  nlo = %s', self.nfrag, nlo)
        log.info('frag_lolist = %s', self.frag_lolist)
        log.info('frag_wghtlist = %s', self.frag_wghtlist)
        log.info('lno_type = %s', self.lno_type)
        log.info('lno_thresh = %s', self.lno_thresh)
        log.info('lno_pct_occ = %s', self.lno_pct_occ)
        log.info('lno_norb = %s', self.lno_norb)
        log.info('lo_proj_thresh = %s', self.lo_proj_thresh)
        log.info('lo_proj_thresh_active = %s', self.lo_proj_thresh_active)
        log.info('verbose_imp = %s', self.verbose_imp)
        log.info('_ovL = %s', self._ovL)
        log.info('_ovL_to_save = %s', self._ovL_to_save)
        log.info('force_outcore_ao2mo = %s', self.force_outcore_ao2mo)
        log.info('chkfile = %s', self.chkfile)
        if self.frag_lno is not None or self.frag_eners is not None:
            log.info('restart: LNOs of %d and energies of %d fragments given',
                     sum(x is not None for x in (self.frag_lno or [])),
                     sum(x is not None for x in (self.frag_eners or [])))
        log.info('_match_oldcode = %s', self._match_oldcode)
        log.info('lno_amp = %s  pt2_amp = %s', self._lno_amp, self._pt2_amp)
        if 'kappa-mp2' in (self._lno_amp, self._pt2_amp):
            log.info('kappa = %s', self.kappa)
        if 'bws2' in (self._lno_amp, self._pt2_amp):
            log.info('alpha = %s  bws2_max_cycle = %d  bws2_conv_tol = %g',
                     self.alpha, self.bws2_max_cycle, self.bws2_conv_tol)
        if self._lno_uses_rpa or self._pt2_uses_rpa:
            log.info('rpa_cd_thresh = %g  rpa_max_cycle = %d  rpa_conv_tol = %g  '
                     'rpa_diis_space = %d  rpa_damp = %g', self.rpa_cd_thresh,
                     self.rpa_max_cycle, self.rpa_conv_tol, self.rpa_diis_space, self.rpa_damp)
        return self

    def kernel(self, eris=None):
        '''The LNO calculation driver.
        '''
        self.dump_flags()

        log = logger.new_logger(self)
        cput0 = (logger.process_clock(), logger.perf_counter())

        frag_wghtlist = self.frag_wghtlist
        nfrag = self.nfrag

        # frag weights
        if frag_wghtlist is None:
            frag_wghtlist = np.ones(nfrag)
        elif isinstance(frag_wghtlist, numbers.Number):
            frag_wghtlist = np.ones(nfrag) * frag_wghtlist
        elif isinstance(frag_wghtlist, Iterable):
            try:
                frag_wghtlist = np.asarray(frag_wghtlist).ravel()
                if len(frag_wghtlist) != nfrag:
                    log.error('Input frag_wghtlist has wrong length (expecting %d; '
                              'got %d).', nfrag, len(frag_wghtlist))
                    raise ValueError
            except Exception:
                raise ValueError
        else:
            log.error('Input frag_wghtlist has wrong data type (expecting '
                      'array-like; got %s)', type(frag_wghtlist))
            raise ValueError

        # dump info
        log.info('Regularized frag_wghtlist = %s', frag_wghtlist)

        log.timer('LO and fragment  ', *cput0)

        self._precompute()

        # the full-system RPA and BW-s2 problems are solved in `kernel`

        frag_res = kernel(self, self.lo_coeff, self.frag_lolist,
                          self.lno_type, self.lno_thresh,
                          self.lno_pct_occ, self.lno_norb, eris=eris)

        self._post_proc(frag_res, frag_wghtlist)

        self._finalize()

        return self.e_corr

    def ao2mo(self):
        log = logger.new_logger(self)

        if self.with_df is None:
            log.error('DF is not found. Rerun SCF with DF.')
            raise NotImplementedError
        else:
            cput0 = (logger.process_clock(), logger.perf_counter())
            orbocc, orbvir = self.split_mo_coeff()[1:3]
            dsize = orbocc.itemsize
            nocc = orbocc.shape[1]
            nvir = orbvir.shape[1]
            # FIXME: more accurate mem estimate
            mem_now = self.max_memory - lib.current_memory()[0]
            naux = self.with_df.get_naoaux()
            mem_df = nocc*nvir*naux*dsize/1024**2.
            log.debug('ao2mo est mem= %.2f MB  avail mem= %.2f MB', mem_df, mem_now)
            if ( (self._ovL_to_save is not None) or (self._ovL is not None) or
                 self.force_outcore_ao2mo or (mem_df > min(mem_now*0.75, mem_now-2000)) ):
                eris = _LNODFOUTCOREERIS(self.with_df, orbocc, orbvir, self.max_memory,
                                         ovL=self._ovL, ovL_to_save=self._ovL_to_save,
                                         verbose=self.verbose, stdout=self.stdout)
            else:
                eris = _LNODFINCOREERIS(self.with_df, orbocc, orbvir, self.max_memory,
                                        verbose=self.verbose, stdout=self.stdout)
            eris.build()
            log.timer('Integral xform   ', *cput0)

            return eris

    def make_lo_rdm1_occ(self, eris, moeocc, moevir, uocc_loc, uvir_loc, occ_lno_type):
        if self._lno_uses_rpa:
            return make_lo_rdm1_occ(self.get_rpa_amp(eris), moeocc, moevir, uocc_loc,
                                    uvir_loc, occ_lno_type)
        if self._lno_amp == 'bws2':
            # MP2 dm in the dressed occupied basis, rotated back
            moeoccW, U, erisW = self.get_bws2(eris)
            dm = make_lo_rdm1_occ(erisW, moeoccW, moevir, np.dot(U.T, uocc_loc), uvir_loc,
                                  occ_lno_type)
            return np.linalg.multi_dot((U, dm, U.T))
        kappa = self.kappa if self._lno_amp == 'kappa-mp2' else None
        return make_lo_rdm1_occ(eris, moeocc, moevir, uocc_loc, uvir_loc, occ_lno_type,
                                kappa=kappa)

    def make_lo_rdm1_vir(self, eris, moeocc, moevir, uocc_loc, uvir_loc, vir_lno_type):
        if self._lno_uses_rpa:
            return make_lo_rdm1_vir(self.get_rpa_amp(eris), moeocc, moevir, uocc_loc,
                                    uvir_loc, vir_lno_type)
        if self._lno_amp == 'bws2':
            moeoccW, U, erisW = self.get_bws2(eris)
            return make_lo_rdm1_vir(erisW, moeoccW, moevir, np.dot(U.T, uocc_loc),
                                    uvir_loc, vir_lno_type)
        kappa = self.kappa if self._lno_amp == 'kappa-mp2' else None
        return make_lo_rdm1_vir(eris, moeocc, moevir, uocc_loc, uvir_loc, vir_lno_type,
                                kappa=kappa)

    _AMP_TYPES = ('mp2', 'kappa-mp2', 'bws2', 'rpa', 'sosex')

    @property
    def _lno_amp(self):
        amp = str(self.lno_amp).lower()
        if amp not in self._AMP_TYPES:
            raise ValueError('lno_amp must be one of %s (got %s)' % (self._AMP_TYPES, amp))
        return amp

    @property
    def _pt2_amp(self):
        if self.pt2_amp is None:
            return self._lno_amp
        amp = str(self.pt2_amp).lower()
        if amp not in self._AMP_TYPES:
            raise ValueError('pt2_amp must be None or one of %s (got %s)' %
                             (self._AMP_TYPES, amp))
        return amp

    @property
    def _lno_uses_rpa(self):
        return self._lno_amp in ('rpa', 'sosex')

    @property
    def _pt2_uses_rpa(self):
        return self._pt2_amp in ('rpa', 'sosex')

    @property
    def _pt2_label(self):
        return {'mp2': 'MP2', 'kappa-mp2': 'kappa-MP2', 'bws2': 'BW-s2', 'rpa': 'RPA',
                'sosex': 'SOSEX'}[self._pt2_amp]

    @property
    def _e_full_scale(self):
        r''' Factor converting a full-system energy to the units of the fragment sum. '''
        return 1.

    @property
    def _rpa_kwargs(self):
        return dict(cd_thresh=self.rpa_cd_thresh, max_cycle=self.rpa_max_cycle,
                    conv_tol=self.rpa_conv_tol, conv_tol_normu=self.rpa_conv_tol_normu,
                    diis_space=self.rpa_diis_space, damp=self.rpa_damp)

    def solve_rpa(self, eris):
        r''' Full-system factorized drCCD. Sets `e_corr_rpa_full` (and `e_corr_sosex_full`
        if pt2_amp is 'sosex') and caches the amplitudes used for LNO construction.
        '''
        log = logger.new_logger(self)
        if self.prune_lno_basis:
            raise NotImplementedError('RPA-based LNO does not support prune_lno_basis.')
        if np.ndim(self.mo_occ) != 1:
            raise NotImplementedError('RPA-based LNO is implemented for RHF only.')
        moeocc, moevir = self.split_mo_energy()[1:3]
        nocc, nvir = moeocc.size, moevir.size
        J = rpa.get_ovJ(eris).reshape(nocc*nvir,-1)
        eia = (moevir[None,:] - moeocc[:,None]).ravel()
        conv, e_rpa, U, tau = rpa.kernel(J, eia, log=log, **self._rpa_kwargs)
        self.e_corr_rpa_full = e_rpa * self._e_full_scale
        log.note('Full-system drCCD:  converged = %s  E_corr(RPA) = %.15g',
                 conv, self.e_corr_rpa_full)
        if self._pt2_amp == 'sosex':
            e_sosex = rpa.energy_sosex(U, J, tau, nocc, nvir, max_memory=self.max_memory)
            self.e_corr_sosex_full = e_sosex * self._e_full_scale
            log.note('Full-system drCCD:  E_corr(SOSEX) = %.15g', self.e_corr_sosex_full)
        J = None
        if self._lno_uses_rpa:
            self._rpa_amp = (eris, rpa._RPAAmpERIS(U.reshape(nocc,nvir,-1), tau,
                                                   self.max_memory, verbose=self.verbose,
                                                   stdout=self.stdout))
        return e_rpa

    def get_rpa_amp(self, eris):
        if self._rpa_amp is None or self._rpa_amp[0] is not eris:
            self.solve_rpa(eris)
        return self._rpa_amp[1]

    def solve_bws2(self, eris):
        r''' Full-system BW-s2(alpha). Sets `e_corr_bws2_full` and caches the dressed
        occupied energies, orbitals and DF integrals used for LNO construction.
        '''
        log = logger.new_logger(self)
        if self.prune_lno_basis:
            raise NotImplementedError('BW-s2-based LNO does not support prune_lno_basis.')
        if np.ndim(self.mo_occ) != 1:
            raise NotImplementedError('BW-s2-based LNO is implemented for RHF only.')
        moeocc, moevir = self.split_mo_energy()[1:3]
        J = rpa.get_ovJ(eris)
        conv, e_corr, moeoccW, U = bws2.kernel(J, moeocc, moevir, alpha=self.alpha,
                                               max_cycle=self.bws2_max_cycle,
                                               conv_tol=self.bws2_conv_tol,
                                               max_memory=self.max_memory, log=log)
        self.e_corr_bws2_full = e_corr * self._e_full_scale
        log.note('Full-system BW-s2(alpha=%g):  converged = %s  E_corr = %.15g',
                 self.alpha, conv, self.e_corr_bws2_full)
        erisW = None
        if self._lno_amp == 'bws2':
            erisW = bws2._DressedERIS(einsum('iax,iI->Iax', J, U), self.max_memory,
                                      verbose=self.verbose, stdout=self.stdout)
        self._bws2 = (eris, moeoccW, U, erisW)
        return e_corr

    def get_bws2(self, eris):
        if self._bws2 is None or self._bws2[0] is not eris or self._bws2[3] is None:
            self.solve_bws2(eris)
        return self._bws2[1:]

    def _precompute(self, *args, **kwargs):
        pass

    get_frozen_mask = mp.mp2.get_frozen_mask
    get_nocc = mp.mp2.get_nocc
    get_nmo = mp.mp2.get_nmo
    split_mo_coeff = mp.dfmp2.DFMP2.split_mo_coeff
    split_mo_energy = mp.dfmp2.DFMP2.split_mo_energy
    split_mo_occ = mp.dfmp2.DFMP2.split_mo_occ
    make_las = make_las

    @property
    def nocc(self):
        return self.get_nocc()

    @property
    def nmo(self):
        return self.get_nmo()

    @property
    def _keep_lno_eris(self):
        r''' Whether the full-system integrals used for the LNOs are kept for the impurity
        problems: when they are read from or saved to a file (`_ovL`, `_ovL_to_save`), with
        domains (`prune_lno_basis`), or with the RPA/SOSEX amplitudes.
        '''
        return (self._ovL is not None or self._ovL_to_save is not None or
                self.prune_lno_basis or self._lno_uses_rpa or self._pt2_uses_rpa)

    def _pack_las(self, lnofrag, frozen):
        r''' Active LNOs {'lno_occ', 'lno_vir'} of an LAS returned by `make_las`. '''
        mo_occ = self.mo_occ
        if np.ndim(mo_occ) == 1:
            lno_occ, lno_vir = split_las(lnofrag, frozen, np.count_nonzero(mo_occ > 1e-10))
        else:
            lno_occ, lno_vir = [list(x) for x in zip(*[
                split_las(lnofrag[s], frozen[s], np.count_nonzero(mo_occ[s] > 1e-10))
                for s in range(2)])]
        return {'lno_occ': lno_occ, 'lno_vir': lno_vir}

    def _unpack_las(self, las, orbloc):
        r''' The LAS (orbfrag, frozen, uocc_loc) as returned by `make_las` from the active
        LNOs `las` of a fragment with LOs `orbloc`.
        '''
        orbloc_las = las.get('orbloc')
        if orbloc_las is not None and not _same_orbloc(orbloc_las, orbloc):
            logger.warn(self, 'The LOs saved with the LNOs differ from those of the fragment.')
        mo_coeff, mo_occ, s1e = self.mo_coeff, self.mo_occ, self.s1e
        if np.ndim(mo_occ) == 1:
            return join_las(las['lno_occ'], las['lno_vir'], mo_coeff, mo_occ, s1e, orbloc)
        return tuple(list(x) for x in zip(*[
            join_las(las['lno_occ'][s], las['lno_vir'][s], mo_coeff[s], mo_occ[s], s1e,
                     orbloc[s]) for s in range(2)]))

    def _chk_init(self, lo_coeff, frag_lolist, frag_lno, frag_eners):
        r''' Start the chkfile of a `kernel` run with the restart data it is given. '''
        chkfile = self.chkfile
        if not chkfile:
            return
        if self._chk_loaded != os.path.abspath(chkfile):
            # a chkfile of another calculation: replace its LNO data
            if h5py.is_hdf5(chkfile):
                with h5py.File(chkfile, 'r+') as f:
                    if 'lno' in f:
                        del f['lno']
            self._chk_dump_full()
            for ifrag,loidx in enumerate(frag_lolist):
                if frag_lno[ifrag] is not None:
                    orbloc = _frag_orbloc(lo_coeff, loidx, ifrag)[0]
                    self._chk_dump_frag(ifrag, 'las',
                                        dict(orbloc=orbloc, lno_occ=frag_lno[ifrag]['lno_occ'],
                                             lno_vir=frag_lno[ifrag]['lno_vir']))
                self._chk_dump_frag(ifrag, 'energy', frag_eners[ifrag])
        lib.chkfile.dump(chkfile, 'lno/nfrag', len(frag_lolist))

    def _chk_dump_frag(self, ifrag, key, value):
        if self.chkfile and value is not None:
            lib.chkfile.dump(self.chkfile, 'lno/frag%d/%s' % (ifrag, key), value)

    def _chk_dump_full(self):
        if self.chkfile:
            for key in ('e_corr_bws2_full', 'e_corr_rpa_full', 'e_corr_sosex_full'):
                if getattr(self, key) is not None:
                    lib.chkfile.dump(self.chkfile, 'lno/'+key, getattr(self, key))

    def load_chk(self, chkfile=None):
        r''' Restart from a chkfile written by `kernel` (default: `self.chkfile`).

        Sets `frag_lno` and `frag_eners` from the LNOs and energies of the fragments that
        finished, and the full-system energies (`e_corr_bws2_full`, `e_corr_rpa_full`,
        `e_corr_sosex_full`) that were computed, so that the next `kernel` call only
        computes what is missing. If `self.chkfile` is the same file, the run continues
        writing to it.
        '''
        if chkfile is None: chkfile = self.chkfile
        dat = lib.chkfile.load(chkfile, 'lno')
        if dat is None:
            raise KeyError('No LNO data in chkfile %s' % chkfile)
        nfrag = int(dat['nfrag'])
        if nfrag != self.nfrag:
            logger.error(self, 'chkfile %s has %d fragments (expecting %d).',
                         chkfile, nfrag, self.nfrag)
            raise ValueError
        self.frag_lno = [None] * nfrag
        self.frag_eners = [None] * nfrag
        for ifrag in range(nfrag):
            frag = dat.get('frag%d' % ifrag)
            if frag is not None:
                self.frag_lno[ifrag] = frag.get('las')
                self.frag_eners[ifrag] = frag.get('energy')
        for key in ('e_corr_bws2_full', 'e_corr_rpa_full', 'e_corr_sosex_full'):
            if key in dat:
                setattr(self, key, float(dat[key]))
        self._chk_loaded = os.path.abspath(chkfile)
        return self

    ''' The following methods need to be implemented for derived LNO classes.
    '''
    def impurity_solve(self, mf, mo_coeff, uocc_loc, eris=None, frozen=None, log=None):
        log = logger.new_logger(self)
        log.error('You are calling the base LNO class! Please call the method-specific '
                  'LNO classes.')
        raise NotImplementedError

    def _post_proc(self, frag_res, frag_wghtlist):
        pass

    def _finalize(self):
        pass

    def _frag_res_to_dict(self, frag_res):
        r''' A fragment result of `impurity_solve` as a dict of numbers/arrays for the
        chkfile and `frag_eners`, or None to not save it.
        '''
        return None

    def _frag_res_from_dict(self, dic):
        r''' Inverse of `_frag_res_to_dict`; None if `dic` is incomplete. '''
        return None


class _LNODFINCOREERIS:
    def __init__(self, with_df, orbocc, orbvir, max_memory, verbose=None, stdout=None):
        self.with_df = with_df
        self.orbocc = orbocc
        self.orbvir = orbvir

        self.max_memory = max_memory
        self.verbose = verbose
        self.stdout = stdout

        self.dtype = self.orbocc.dtype
        self.dsize = self.orbocc.itemsize

        self.ovL = None

    @property
    def nocc(self):
        return self.orbocc.shape[1]
    @property
    def nvir(self):
        return self.orbvir.shape[1]
    @property
    def naux(self):
        return self.with_df.get_naoaux()

    def build(self):
        log = logger.new_logger(self)
        self.ovL = _init_mp_df_eris(self.with_df, self.orbocc, self.orbvir,
                                    self.max_memory, ovL=self.ovL, log=log)

    def get_occ_blk(self, i0,i1):
        return np.asarray(self.ovL[i0:i1], order='C')

    def get_vir_blk(self, a0,a1):
        return np.asarray(self.ovL[:,a0:a1], order='C')

    def xform_occ(self, u):
        # return lib.einsum('iax,iI->Iax', self.ovL, u.conj())
        nocc, nvir, naux = self.nocc, self.nvir, self.naux
        nOcc = u.shape[1]
        M = (self.max_memory - lib.current_memory()[0])*1e6 / self.dsize
        occblksize = min(nocc, max(1, int(np.floor(M*0.5/(nvir*naux) - nOcc))))
        if DEBUG_BLKSIZE: occblksize = max(1,nocc//2)
        OvL = np.empty((nOcc,nvir,naux), dtype=self.dtype)
        for iblk,(i0,i1) in enumerate(lib.prange(0,nocc,occblksize)):
            if iblk == 0:
                OvL[:]  = lib.einsum('iax,iI->Iax', self.get_occ_blk(i0,i1), u[i0:i1].conj())
            else:
                OvL[:] += lib.einsum('iax,iI->Iax', self.get_occ_blk(i0,i1), u[i0:i1].conj())
        return OvL

    def xform_vir(self, u):
        # return lib.einsum('iax,aA->iAx', self.ovL, u)
        nocc, nvir, naux = self.nocc, self.nvir, self.naux
        nVir = u.shape[1]
        M = (self.max_memory - lib.current_memory()[0])*1e6 / self.dsize
        occblksize = min(nocc, max(1, int(np.floor(M*0.5/(nvir*naux) - nocc*nVir/float(nvir)))))
        if DEBUG_BLKSIZE: occblksize = max(1,nocc//2)
        oVL = np.empty((nocc,nVir,naux), dtype=self.dtype)
        for i0,i1 in lib.prange(0,nocc,occblksize):
            oVL[i0:i1] = lib.einsum('iax,aA->iAx', self.get_occ_blk(i0,i1), u)
        return oVL


class _LNODFOUTCOREERIS(_LNODFINCOREERIS):
    def __init__(self, with_df, orbocc, orbvir, max_memory, ovL=None, ovL_to_save=None,
                 verbose=None, stdout=None):
        _LNODFINCOREERIS.__init__(self, with_df, orbocc, orbvir, max_memory, verbose, stdout)

        self._ovL = ovL
        self._ovL_to_save = ovL_to_save

    def build(self):
        log = logger.new_logger(self)
        nocc,nvir,naux = self.nocc,self.nvir,self.naux
        ovL_shape = (nocc,nvir,naux)
        if self._ovL is None:
            if isinstance(self._ovL_to_save, str):
                self.feri = h5py.File(self._ovL_to_save, 'w')
            else:
                self.feri = lib.H5TmpFile()
            log.info('ovL is saved to %s', self.feri.filename)
            self.ovL = self.feri.create_dataset('ovL', ovL_shape, dtype=self.dtype,
                                                chunks=(1,*ovL_shape[1:]))
            _init_mp_df_eris(self.with_df, self.orbocc, self.orbvir, self.max_memory,
                             ovL=self.ovL, log=log)
        elif isinstance(self._ovL, str):
            self.feri = h5py.File(self._ovL, 'r')
            log.info('ovL is read from %s', self.feri.filename)
            assert( 'ovL' in self.feri )
            assert( self.feri['ovL'].shape == ovL_shape )
            self.ovL = self.feri['ovL']
        else:
            raise RuntimeError

def _init_mp_df_eris(with_df, occ_coeff, vir_coeff, max_memory, ovL=None, log=None):
    from pyscf.ao2mo import _ao2mo

    if log is None: log = logger.Logger(sys.stdout, 3)

    nao,nocc = occ_coeff.shape
    nvir = vir_coeff.shape[1]
    nmo = nocc + nvir
    nao_pair = nao**2
    naux = with_df.get_naoaux()

    dtype = occ_coeff.dtype
    dsize = occ_coeff.itemsize

    mo = np.asarray(np.hstack((occ_coeff,vir_coeff)), order='F')
    ijslice = (0, nocc, nocc, nmo)

    if ovL is None:
        ovL = np.empty((nocc,nvir,naux), dtype=dtype)

    mem_avail = max_memory - lib.current_memory()[0]

    if dtype == np.float64:
        def loop_df(blksize):
            for Lpq in df_loop(with_df, blksize=blksize):
                yield Lpq
                Lpq = None
        def ao2mo_df(Lpq, mo, ijslice, out):
            return _ao2mo.nr_e2(Lpq, mo, ijslice, aosym='s2', out=out)
    else:
        def loop_df(blksize):
            kpti_kptj = [with_df.kpts[0]]*2
            for LpqR, LpqI, sign in with_df.sr_loop(blksize=aux_blksize,
                                                    kpti_kptj=kpti_kptj):
                Lpq = LpqR + LpqI*1j
                LpqR = LpqI = None
                if Lpq.shape[1] != nao_pair:
                    Lpq = lib.unpack_tril(Lpq).astype(dtype)
                yield Lpq
                Lpq = None
        def ao2mo_df(Lpq, mo, ijslice, out):
            return _ao2mo.r_e2(Lpq, mo, ijslice, [], None, aosym='s1', out=out)

    if isinstance(ovL, np.ndarray):
        # incore: batching aux (OV + Nao_pair) * [X] = M
        mem_auxblk = (nao_pair+nocc*nvir) * dsize/1e6
        aux_blksize = min(naux, max(1, int(np.floor(mem_avail*0.5 / mem_auxblk))))
        if DEBUG_BLKSIZE: aux_blksize = max(1,naux//2)
        log.debug('aux blksize for incore ao2mo: %d/%d', aux_blksize, naux)
        buf = np.empty(aux_blksize*nocc*nvir, dtype=dtype)
        ijslice = (0,nocc,nocc,nmo)

        p1 = 0
        for Lpq in loop_df(aux_blksize):
            p0, p1 = p1, p1+Lpq.shape[0]
            out = ao2mo_df(Lpq, mo, ijslice, buf)
            ovL[:,:,p0:p1] = out.reshape(-1,nocc,nvir).transpose(1,2,0)
            Lpq = out = None
        buf = None
    else:
        # outcore: batching occ [O]XV and aux ([O]V + Nao_pair)*[X]
        mem_occblk = naux*nvir * dsize/1e6
        occ_blksize = min(nocc, max(1, int(np.floor(mem_avail*0.6 / mem_occblk))))
        if DEBUG_BLKSIZE: occ_blksize = max(1,nocc//2)
        mem_auxblk = (occ_blksize*nvir+nao_pair) * dsize/1e6
        aux_blksize = min(naux, max(1, int(np.floor(mem_avail*0.3 / mem_auxblk))))
        if DEBUG_BLKSIZE: aux_blksize = max(1,naux//2)
        log.debug('occ blksize for outcore ao2mo: %d/%d', occ_blksize, nocc)
        log.debug('aux blksize for outcore ao2mo: %d/%d', aux_blksize, naux)
        buf = np.empty(naux*occ_blksize*nvir, dtype=dtype)
        buf2 = np.empty(aux_blksize*occ_blksize*nvir, dtype=dtype)

        for i0,i1 in lib.prange(0,nocc,occ_blksize):
            nocci = i1-i0
            ijslice = (i0,i1,nocc,nmo)
            p1 = 0
            OvL = np.ndarray((nocci,nvir,naux), dtype=dtype, buffer=buf)
            for Lpq in loop_df(aux_blksize):
                p0, p1 = p1, p1+Lpq.shape[0]
                out = ao2mo_df(Lpq, mo, ijslice, buf2)
                OvL[:,:,p0:p1] = out.reshape(-1,nocci,nvir).transpose(1,2,0)
                Lpq = out = None
            ovL[i0:i1] = OvL    # this avoids slow operations like ovL[i0:i1,:,p0:p1] = ...
            OvL = None
        buf = buf2 = None

    return ovL
