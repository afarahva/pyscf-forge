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
import json
import numbers
from collections.abc import Iterable
from functools import reduce

import numpy as np
import h5py

from pyscf import mp
from pyscf.lib import logger
from pyscf import lib
from pyscf import __config__

from pyscf.lno.make_lno_rdm1 import make_lo_rdm1
from pyscf.lno import drpa
from pyscf.lno.tools import df_loop

einsum = lib.einsum

DEBUG_BLKSIZE = getattr(__config__, 'lnocc_DEBUG_BLKSIZE', False)


MUTE_CHKFILE = getattr(__config__, 'lno_LNO_mute_chkfile', False)

# low-level amplitudes (lno_amp / pt2_amp) and their energy labels
PT2_LABELS = {'mp2': 'MP2', 'kappa-mp2': 'kappa-MP2', 'rpa': 'RPA', 'sosex': 'SOSEX'}


def kernel(mlno, lo_coeff, frag_lolist, lno_type, lno_thresh=None, lno_pct_occ=None,
           lno_norb=None, eris=None):
    r''' Kernel function for LNO-based methods.

    Fragments are treated one at a time: the LNO active space of a fragment is built from
    the full-system integrals (`eris`) and its impurity problem is solved before moving on
    to the next fragment. With `mlno.delete_full_eris`, the full-system integrals are
    released before each impurity problem, leaving the memory to the impurity solver, and
    rebuilt for the next fragment that needs LNOs.

    Fragments whose energies (`mlno.frag_eners`) are already known, e.g. from
    `mlno.load_chk`, are skipped, and fragments whose LNOs (`mlno.frag_lno`) are known
    reuse them instead of building them. As each fragment finishes, its energies and (with
    `mlno.save_lno`) its LNOs are saved to `mlno.chkfile`.

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
        eris (optional):
            Full-system integrals from `mlno.ao2mo`. Built here if not given. Integrals
            passed in are used for every fragment and left intact for the caller
            (`delete_full_eris` only applies to integrals built here). They may be given
            dressed (`mlno.get_dress_eris`).
    '''
    nfrag = len(frag_lolist)
    if lno_pct_occ is None:
        lno_pct_occ = [None, None]
    if lno_norb is None:
        lno_norb = [[None,None]] * nfrag

    mf = mlno._scf

    log = logger.new_logger(mlno)

    cput0 = (logger.process_clock(), logger.perf_counter())

    # Restart: unpack `mlno.frag_lno` / `mlno.frag_eners` and seed the chkfile.
    frag_lno, frag_eners, frag_res = mlno._restart_frags(lo_coeff, frag_lolist)
    nlas = sum(frag_res[i] is None and frag_lno[i] is None for i in range(nfrag))

    # The integrals passed in (`eris`) are kept for the caller. Those built here (`full_eris`
    # and `dressed_eris`) are torn down by `delete_full_eris` before each impurity problem,
    # and at the end otherwise.
    delete_eris = mlno.delete_full_eris and eris is None
    if mlno.delete_full_eris and eris is not None:
        log.info('delete_full_eris: the integrals passed to kernel are kept')
    if delete_eris and nlas > 1:
        log.info('delete_full_eris: the full-system integrals%s are rebuilt for each of '
                 'the %d fragments that need LNOs',
                 ' and dRPA amplitudes' if mlno._lno_uses_rpa else '', nlas)

    full_eris = dressed_eris = None

    cput2 = cput1 = (logger.process_clock(), logger.perf_counter())

    for ifrag,loidx in enumerate(frag_lolist):
        if frag_res[ifrag] is not None:
            log.info('Fragment %d/%d  Sol: restored', ifrag+1, nfrag)
            continue

        # LNO active space: build it or reuse the one given
        if frag_lno[ifrag] is None:
            if dressed_eris is None:
                if eris is None:   # built here, and again after delete_full_eris
                    full_eris, dressed_eris = mlno._build_eris()
                else:
                    full_eris, dressed_eris = eris, mlno.get_dress_eris(eris)
            orbloc, lno_param = _frag_orbloc(lo_coeff, loidx, ifrag, lno_thresh,
                                             lno_pct_occ, lno_norb)
            lnofrag, frozen, uocc_loc, frag_msg = mlno.make_las(dressed_eris, orbloc,
                                                                lno_type, lno_param)
            if mlno.save_lno:
                mlno._chk_dump_frag(ifrag, 'las',
                                    dict(orbloc=orbloc, **mlno._pack_las(lnofrag, frozen)))
            cput2 = log.timer('Fragment %d make las'%(ifrag+1), *cput2)
            log.info('Fragment %d/%d  LAS: %s', ifrag+1, nfrag, frag_msg)
        else:
            orbloc = _frag_orbloc(lo_coeff, loidx, ifrag)[0]
            lnofrag, frozen, uocc_loc = mlno._unpack_las(frag_lno[ifrag], orbloc)
            frag_lno[ifrag] = None
            log.info('Fragment %d/%d  LAS: restored', ifrag+1, nfrag)

        if delete_eris and dressed_eris is not None:
            log.info('Releasing the full-system integrals')
            mlno._release_full_eris(full_eris, dressed_eris)
            full_eris = dressed_eris = None

        # solve impurity problem
        frag_res[ifrag], frag_msg = mlno.impurity_solve(mf, lnofrag, uocc_loc, full_eris,
                                                        frozen=frozen)
        lnofrag = frozen = uocc_loc = None
        frag_eners[ifrag] = mlno._frag_res_to_dict(frag_res[ifrag])
        mlno._chk_dump_frag(ifrag, 'energy', frag_eners[ifrag])
        cput2 = log.timer('Fragment %d imp sol '%(ifrag+1), *cput2)
        log.info('Fragment %d/%d  Sol: %s', ifrag+1, nfrag, frag_msg)
        cput1 = log.timer('Fragment %d'%(ifrag+1)+' '*(8-len(str(ifrag+1))), *cput1)

    if eris is None and dressed_eris is not None:
        mlno._release_full_eris(full_eris, dressed_eris)
    full_eris = dressed_eris = None

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
    return orbfrag[:,maskact & maskocc], orbfrag[:,maskact & ~maskocc]

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

def _lolist_dumps(frag_lolist):
    r''' `frag_lolist` (nested lists of LO indices, any int types; per fragment a pair of
    lists for ULNO) as a JSON string, the form it is saved in the chkfile.
    '''
    return json.dumps(frag_lolist, default=lambda x: x.tolist())

def _release_eris(eris):
    r''' Free the full-system integrals used to build the LNOs. Any HDF5 file backing an
    outcore eris is closed (and, when it is a scratch file, deleted) so that both its memory
    and its disk space are reclaimed right away rather than on garbage collection. Only
    called on integrals built by `kernel` / `make_lno_active_space` themselves.
    '''
    feri = getattr(eris, 'feri', None)
    if feri is not None:
        try:
            feri.close()
        except Exception:
            pass
        eris.feri = None
    for key in ('ovL', 'ovLR', 'ovLI', 'Lov'):
        if hasattr(eris, key):
            setattr(eris, key, None)


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
    dmoo, dmvv = mlno.make_lo_rdm1(eris, moeocc, moevir, uocc_loc, uvir_loc, lno_type)
    cput1 = log.timer_debug1('make_lo_rdm1', *cput1)
    if mlno._match_oldcode: dmoo *= 0.5 # TO MATCH OLD LNO CODE
    dmoo = reduce(np.dot, (uocc_orth.T.conj(), dmoo, uocc_orth))
    if lno_param[0]['norb'] is not None:
        lno_param[0]['norb'] -= uocc_loc.shape[1] + uocc_std.shape[1]
    uoccact_orth, uoccfrz_orth = natorb_select(dmoo, uocc_orth, **lno_param[0])
    orboccfrz = np.hstack((orboccfrz_core, np.dot(orbocc, uoccfrz_orth)))
    uoccact = subspace_eigh(np.diag(moeocc), np.hstack((uoccact_orth, uocc_std, uocc_loc)))[1]
    orboccact = np.dot(orbocc, uoccact)
    uoccact_loc = np.linalg.multi_dot((orboccact.T.conj(), s1e, orbloc))

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
    cput1 = log.timer_debug1('LNO selection', *cput1)

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

        # low-level amplitudes
        #   lno_amp : amplitudes for the LNO density matrices
        #             'mp2', 'kappa-mp2', 'rpa' or 'sosex' ('rpa' and 'sosex'
        #             both use the drCCD amplitudes; they differ only in the energy)
        #   pt2_amp : amplitudes for the fragment PT2 energy and the impurity CC guess;
        #             None uses lno_amp, otherwise one of the lno_amp choices
        #   kappa   : kappa-MP2 regularization strength, arXiv:2508.15744
        #   rpa_solver : drCCD solver, 'eig' or 'riccati' (other settings are the defaults
        #             of `pyscf.lno.drpa.DRPA`)
        self.lno_amp = 'mp2'
        self.pt2_amp = None
        self.kappa = 1.1
        self.rpa_solver = 'eig'
        # full-system dRPA and SOSEX energies (per unit cell for KLNO), set when the
        # full-system drCCD is solved for the LNOs (lno_amp 'rpa' / 'sosex')
        self.e_corr_rpa_full = None
        self.e_corr_sosex_full = None

        # df eri
        self._ovL = None
        self._ovL_to_save = None
        self.force_outcore_ao2mo = False

        # memory
        #   delete_full_eris : release the full-system integrals (only needed to build the
        #                      LNOs) before each impurity problem, leaving the memory to the
        #                      impurity solver. They are rebuilt for the next fragment that
        #                      needs LNOs (with the full-system drCCD for lno_amp 'rpa' /
        #                      'sosex'), so this suits runs with one or few fragments per
        #                      job.
        self.delete_full_eris = False

        # checkpoint and restart
        #   chkfile    : the LOs (`lo_coeff`) and the fragments (`frag_lolist`) are saved
        #                when `kernel` starts, the energies of each fragment as its impurity
        #                problem is solved, and (if save_lno) its LOs and LNOs as they are
        #                built. `load_chk` restores the LOs and fragments, so a restart uses
        #                those its stored fragments were built from, and a calculation can be
        #                restarted from its chkfile alone, e.g.
        #                    LNOCCSD_T(mf, None, None).load_chk(chkfile).kernel()
        #                As for SCF, the default file is removed with the object; assign
        #                a filename to keep it.
        #   save_lno   : whether to save the LNOs to the chkfile (they can be large)
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
        self.save_lno = True
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
        log.info('chkfile = %s  save_lno = %s', self.chkfile, self.save_lno)
        log.info('delete_full_eris = %s', self.delete_full_eris)
        if self.frag_lno is not None or self.frag_eners is not None:
            log.info('restart: LNOs of %d and energies of %d fragments given',
                     sum(x is not None for x in (self.frag_lno or [])),
                     sum(x is not None for x in (self.frag_eners or [])))
        log.info('_match_oldcode = %s', self._match_oldcode)
        log.info('lno_amp = %s  pt2_amp = %s', self._lno_amp, self._pt2_amp)
        if 'kappa-mp2' in (self._lno_amp, self._pt2_amp):
            log.info('kappa = %s', self.kappa)
        if self._lno_uses_rpa or self._pt2_uses_rpa:
            log.info('rpa_solver = %s', self.rpa_solver)
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

        frag_res = self._solve_frags(eris=eris)

        self._post_proc(frag_res, frag_wghtlist)

        self._finalize()

        return self.e_corr

    def _solve_frags(self, eris=None):
        r''' Build the LNO active space of each fragment and solve its impurity problem
        (see the module-level `kernel`). Returns the `impurity_solve` result of each
        fragment. Domain-based LNO methods (`pyscf.dlno`) replace this step.
        '''
        return kernel(self, self.lo_coeff, self.frag_lolist, self.lno_type, self.lno_thresh,
                      self.lno_pct_occ, self.lno_norb, eris=eris)

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

    def get_ovJ(self, orbocc, orbvir):
        r''' Real DF integrals J[i,a,P] of the orbitals `orbocc` and `orbvir`, with
        (ia|jb) = sum_P J[i,a,P] J[j,b,P], built in core from the AO DF integrals (e.g. for
        the active space of an impurity).
        '''
        eris = _LNODFINCOREERIS(self.with_df, orbocc, orbvir, self.max_memory,
                                verbose=self.verbose, stdout=self.stdout)
        eris.build()
        return drpa.get_ovJ(eris)

    def make_lno_active_space(self, ifrag=None, eris=None):
        r''' Build the LNO active space of one or more fragments, without solving their
        impurity problems.

        This runs only the LNO-construction half of `kernel`, returning each fragment's
        active local natural orbitals in the same packed form as an entry of `frag_lno`:
        a dict {'lno_occ', 'lno_vir'} of AO-basis coefficient matrices (lists over spin
        for unrestricted methods). It is meant for inspecting or debugging the active
        spaces, and for populating `frag_lno` by hand, e.g.

            mlno.frag_lno = mlno.make_lno_active_space()
            mlno.kernel()   # reuses the LNOs instead of rebuilding them

        Args:
            ifrag (int or iterable of int, optional):
                Fragment(s) to build, indexing `frag_lolist`. Default: all fragments.
            eris (optional):
                Full-system integrals. Built via `ao2mo` (and reused across fragments) if
                not given. A caller-supplied eris is left intact; one built here is
                released afterwards.

        Returns:
            One {'lno_occ', 'lno_vir'} dict if `ifrag` is an int, else a list of them, one
            per requested fragment.
        '''
        self._precompute()

        single = isinstance(ifrag, numbers.Integral)
        if ifrag is None:
            ifraglist = list(range(self.nfrag))
        elif single:
            ifraglist = [ifrag]
        else:
            ifraglist = list(ifrag)

        lno_pct_occ = [None, None] if self.lno_pct_occ is None else self.lno_pct_occ
        lno_norb = [[None, None]] * self.nfrag if self.lno_norb is None else self.lno_norb

        if eris is None:
            full_eris, dressed_eris = self._build_eris()
        else:
            full_eris, dressed_eris = eris, self.get_dress_eris(eris)

        frag_lno = []
        for i in ifraglist:
            orbloc, lno_param = _frag_orbloc(self.lo_coeff, self.frag_lolist[i], i,
                                             self.lno_thresh, lno_pct_occ, lno_norb)
            lnofrag, frozen = self.make_las(dressed_eris, orbloc, self.lno_type, lno_param)[:2]
            frag_lno.append(self._pack_las(lnofrag, frozen))

        if eris is None:   # built here
            self._release_full_eris(full_eris, dressed_eris)

        return frag_lno[0] if single else frag_lno

    def get_dress_eris(self, eris):
        r''' The integrals dressed for `lno_amp`: `eris` itself for MP2, damped for kappa-MP2,
        or dressed by the full-system drCCD (solved here) for RPA/SOSEX. Integrals that are
        already dressed for `lno_amp` are returned as they are.
        '''
        amp = self._lno_amp
        dressed_for = getattr(eris, 'dressed_for', 'mp2')   # plain integrals are MP2 ones
        if dressed_for == amp:
            return eris
        if dressed_for != 'mp2':
            raise ValueError('eris are dressed for lno_amp = %s, not %s' % (dressed_for, amp))
        if amp == 'kappa-mp2':
            dressed_eris = _DressedERIS(eris, kappa=self.kappa)
        elif amp in ('rpa', 'sosex'):
            dressed_eris = self._dress_rpa(eris)
        else:
            raise ValueError('lno_amp = %s has no dressing' % amp)
        dressed_eris.dressed_for = amp
        return dressed_eris

    def _dress_rpa(self, eris):
        r''' `eris` dressed by the full-system drCCD (RPA/SOSEX). '''
        return _DressedERIS(eris, ovL=self.solve_rpa(eris))

    def make_lo_rdm1(self, eris, moeocc, moevir, uocc_loc, uvir_loc, lno_type):
        r''' Occupied and virtual LNO density matrices of a fragment for lno_type =
        [occ_type, vir_type] (a type that is None is not made). Pass `eris` dressed
        (`get_dress_eris`), as `make_las` does: plain integrals are dressed on every call,
        which for RPA/SOSEX solves the full-system drCCD again.
        '''
        return make_lo_rdm1(self.get_dress_eris(eris), moeocc, moevir, uocc_loc, uvir_loc,
                            lno_type)

    def make_lo_rdm1_occ(self, eris, moeocc, moevir, uocc_loc, uvir_loc, occ_lno_type):
        return self.make_lo_rdm1(eris, moeocc, moevir, uocc_loc, uvir_loc,
                                 [occ_lno_type, None])[0]

    def make_lo_rdm1_vir(self, eris, moeocc, moevir, uocc_loc, uvir_loc, vir_lno_type):
        return self.make_lo_rdm1(eris, moeocc, moevir, uocc_loc, uvir_loc,
                                 [None, vir_lno_type])[1]

    _AMP_TYPES = tuple(PT2_LABELS)

    def _check_amp(self, amp, name):
        amp = str(amp).lower()
        if amp not in self._AMP_TYPES:
            raise ValueError('%s must be one of %s (got %s)' % (name, self._AMP_TYPES, amp))
        return amp

    @property
    def _lno_amp(self):
        return self._check_amp(self.lno_amp, 'lno_amp')

    @property
    def _pt2_amp(self):
        if self.pt2_amp is None:
            return self._lno_amp
        return self._check_amp(self.pt2_amp, 'pt2_amp')

    @property
    def _lno_uses_rpa(self):
        return self._lno_amp in ('rpa', 'sosex')

    @property
    def _pt2_uses_rpa(self):
        return self._pt2_amp in ('rpa', 'sosex')

    @property
    def _pt2_label(self):
        return PT2_LABELS[self._pt2_amp]

    def DRPA(self):
        r''' The drCCD solver (`rpa_solver`) for the full system and the impurities. '''
        return drpa.DRPA(solver=self.rpa_solver, max_memory=self.max_memory,
                         verbose=self.verbose, stdout=self.stdout)

    def solve_rpa(self, eris, ncells=1):
        r''' Full-system drCCD from `eris`. Sets `e_corr_rpa_full` and `e_corr_sosex_full`, per
        unit cell for the integrals of a supercell of `ncells` cells, and returns the dressed
        integrals U[i,a,P] (see `pyscf.regpt2.driccd`).
        '''
        log = logger.new_logger(self)
        mrpa = self.DRPA()
        mrpa.kernel(drpa.get_ovJ(eris), *self.split_mo_energy()[1:3])
        self.e_corr_rpa_full = mrpa.e_corr / ncells
        log.note('Full-system drCCD (%s):  converged = %s  E_corr(RPA) = %.15g',
                 mrpa.solver, mrpa.converged, self.e_corr_rpa_full)
        if mrpa.e_corr_sosex is not None:
            self.e_corr_sosex_full = mrpa.e_corr_sosex / ncells
            log.note('Full-system drCCD:  E_corr(SOSEX) = %.15g', self.e_corr_sosex_full)
        self._chk_dump_full()
        return mrpa.U

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

    def _build_eris(self):
        r''' Full-system integrals and their dressed integrals (`get_dress_eris`). With
        RPA/SOSEX the plain integrals are released at once (nothing else uses them; the
        impurity problems build their own) and None is returned for them.
        '''
        eris = self.ao2mo()
        dressed_eris = self.get_dress_eris(eris)
        if self._lno_uses_rpa:
            _release_eris(eris)
            eris = None
        return eris, dressed_eris

    def _release_full_eris(self, eris, dressed_eris=None):
        r''' Release full-system integrals built here, and their dressed integrals. '''
        if dressed_eris is not None and dressed_eris is not eris:
            _release_eris(dressed_eris)
        if eris is not None:
            _release_eris(eris)

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

    def _restart_frags(self, lo_coeff, frag_lolist):
        r''' Turn the restart inputs `frag_lno` / `frag_eners` into the three per-fragment
        lists `kernel` works with, and seed the chkfile:

            frag_lno   : active LNOs, or None if they must be built
            frag_eners : stored energy dicts (see `_frag_res_to_dict`)
            frag_res   : decoded `impurity_solve` results, None where a fragment is unsolved
                         (or its stored energies were incomplete and must be recomputed)
        '''
        nfrag = len(frag_lolist)
        frag_lno = [None]*nfrag if self.frag_lno is None else list(self.frag_lno)
        frag_eners = [None]*nfrag if self.frag_eners is None else list(self.frag_eners)
        if len(frag_lno) != nfrag or len(frag_eners) != nfrag:
            logger.error(self, 'frag_lno and frag_eners must have one entry per fragment '
                         '(%d).', nfrag)
            raise ValueError
        frag_res = [None]*nfrag
        for ifrag, dic in enumerate(frag_eners):
            if dic is not None:
                frag_res[ifrag] = self._frag_res_from_dict(dic)
                if frag_res[ifrag] is None:
                    logger.info(self, 'Fragment %d/%d  energies incomplete; the fragment '
                                'is recomputed', ifrag+1, nfrag)
        self._chk_init(lo_coeff, frag_lolist, frag_lno, frag_eners)
        return frag_lno, frag_eners, frag_res

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
                if self.save_lno and frag_lno[ifrag] is not None:
                    orbloc = _frag_orbloc(lo_coeff, loidx, ifrag)[0]
                    self._chk_dump_frag(ifrag, 'las',
                                        dict(orbloc=orbloc, lno_occ=frag_lno[ifrag]['lno_occ'],
                                             lno_vir=frag_lno[ifrag]['lno_vir']))
                self._chk_dump_frag(ifrag, 'energy', frag_eners[ifrag])
        lib.chkfile.dump(chkfile, 'lno/nfrag', len(frag_lolist))
        # the LOs and the fragments made of them, always: a restart must use these
        lib.chkfile.dump(chkfile, 'lno/lo_coeff', lo_coeff)
        lib.chkfile.dump(chkfile, 'lno/frag_lolist', _lolist_dumps(frag_lolist))

    def _chk_dump_frag(self, ifrag, key, value):
        if self.chkfile and value is not None:
            lib.chkfile.dump(self.chkfile, 'lno/frag%d/%s' % (ifrag, key), value)

    def _chk_dump_full(self):
        if self.chkfile:
            for key in ('e_corr_rpa_full', 'e_corr_sosex_full'):
                if getattr(self, key) is not None:
                    lib.chkfile.dump(self.chkfile, 'lno/'+key, getattr(self, key))

    def load_chk(self, chkfile=None):
        r''' Restart from a chkfile written by `kernel` (default: `self.chkfile`).

        Sets `lo_coeff` and `frag_lolist` to the LOs and fragments saved in the chkfile (those
        the stored fragments were built from, which replace any given, so both may be None),
        `frag_lno` and `frag_eners` from the LNOs and energies of the fragments that
        finished, and the full-system energies (`e_corr_rpa_full`, `e_corr_sosex_full`)
        that were computed, so that the next `kernel` call only computes what is missing.
        If `self.chkfile` is the same file, the run continues writing to it.
        '''
        if chkfile is None: chkfile = self.chkfile
        dat = lib.chkfile.load(chkfile, 'lno')
        if dat is None:
            raise KeyError('No LNO data in chkfile %s' % chkfile)
        frag_lolist = dat.get('frag_lolist')
        if frag_lolist is None:
            if self.frag_lolist is None:
                raise KeyError('chkfile %s has no fragments (written before they were saved); '
                               'pass frag_lolist' % chkfile)
        else:
            frag_lolist = json.loads(frag_lolist)
            if (self.frag_lolist is not None and
                    json.loads(_lolist_dumps(self.frag_lolist)) != frag_lolist):
                logger.info(self, 'Using the fragments saved in chkfile %s instead of those '
                            'given.', chkfile)
            self.frag_lolist = frag_lolist
        nfrag = int(dat['nfrag'])
        if nfrag != self.nfrag:
            logger.error(self, 'chkfile %s has %d fragments (expecting %d).',
                         chkfile, nfrag, self.nfrag)
            raise ValueError
        lo_coeff = dat.get('lo_coeff')
        if lo_coeff is None:
            if self.lo_coeff is None:
                raise KeyError('chkfile %s has no LOs (written before they were saved); '
                               'pass lo_coeff' % chkfile)
            logger.warn(self, 'chkfile %s has no LOs (written before they were saved): the '
                        'fragments restored from it are assumed to come from the LOs given.',
                        chkfile)
        else:
            if isinstance(lo_coeff, (list, tuple)):   # Unrestricted
                lo_coeff = [np.asarray(x) for x in lo_coeff]
            if self.lo_coeff is not None and not _same_orbloc(lo_coeff, self.lo_coeff):
                logger.info(self, 'Using the LOs saved in chkfile %s instead of those given.',
                            chkfile)
            self.lo_coeff = lo_coeff
        self.frag_lno = [None] * nfrag
        self.frag_eners = [None] * nfrag
        for ifrag in range(nfrag):
            frag = dat.get('frag%d' % ifrag)
            if frag is not None:
                self.frag_lno[ifrag] = frag.get('las')
                self.frag_eners[ifrag] = frag.get('energy')
        for key in ('e_corr_rpa_full', 'e_corr_sosex_full'):
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
        if self.ovL is None:
            return self.with_df.get_naoaux()
        return self.ovL.shape[-1]

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


class _DressedERIS:
    r''' An LNO ERI object as seen by the density matrices of non-MP2 amplitudes.

    The LNO density matrices (see `make_lno_rdm1`) are built from MP2 amplitudes
    t2 = (ia|jb) / (e_i + e_j - e_a - e_b) of the 3-index integrals of an ERI object. This
    wraps the ERI object `eris` of any LNO class and either

        damps the amplitudes, t2 *= 1 - exp(kappa * (e_i + e_j - e_a - e_b))   (kappa-MP2)

    or replaces its integrals by the in-core, real dressed integrals `ovL`, U = (1 + 2T) J,
    whose MP2 amplitudes are the drCCD ones (dRPA, see `pyscf.regpt2.driccd`).
    '''
    def __init__(self, eris, kappa=None, ovL=None):
        if ovL is not None:
            ovL_eris = _LNODFINCOREERIS(None, eris.orbocc, eris.orbvir, eris.max_memory,
                                        verbose=eris.verbose, stdout=eris.stdout)
            ovL_eris.ovL = np.asarray(ovL, dtype=np.float64)
            ovL_eris.dtype, ovL_eris.dsize = np.float64, 8
            eris = ovL_eris
        self._eris = eris
        self.kappa = kappa

    def __getattr__(self, key):
        if key == '_eris':
            raise AttributeError(key)
        return getattr(self._eris, key)


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
