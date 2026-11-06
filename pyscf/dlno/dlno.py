#!/usr/bin/env python
# Copyright 2014-2025 The PySCF Developers. All Rights Reserved.
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
# Author: Ardavan Farahvash
#

r''' Domain-based local natural orbital (DLNO) methods.

    An LNO method (`pyscf.lno`) builds the LNOs of each fragment from the MP2 density
    matrices of the full system, whose integrals (ia|L) over all occupied and virtual MOs
    make the cost grow steeply with system size. The DLNO methods build the LNOs of a
    fragment from the MP2 of a *domain* of the fragment instead, following the local MP2
    of https://github.com/fishjojo/dlno:

    1. The occupied LOs `lo_coeff` (LMOs; they must span the active occupied space) get
       Boughton-Pulay (BP) atomic domains (completeness `lmo_bp_domain_thr`), the PAOs
       (AOs projected onto the virtual space) get BP domains (`pao_bp_domain_thr`), and
       each LMO gets a primary domain: its BP domain and the BP domains of the PAOs on it.
    2. The MP2 pair energies of the LMOs are estimated by a multipole expansion
       (`multipole_order`) with the semicanonical PAOs of the LMOs' primary domains.
       Pairs with |E_ij| > `pair_energy_thr` are strong.
    3. The domain of a fragment holds
         occupied : its LMOs and their strong pair partners;
         virtual  : the PAOs on the BP-domain atoms of these LMOs (extended domain);
         AO basis : the primary-domain atoms of these LMOs (extended primary domain), with
                    the occupied and virtual orbitals projected onto it, the AO basis and
                    the DF auxiliary basis restricted to it (`domain_basis = 'local'`,
                    molecules), or the full AO basis (`domain_basis = 'full'`).
       Both orbital spaces are semicanonicalized with the full-system Fock matrix.
    4. The LNOs of the fragment are built from the MP2 of the domain and its impurity
       problem is solved exactly as in the parent LNO method.
    5. The local MP2 energy of the fragment in its domain, plus the multipole energy of the
       weak pairs, adds up to an estimate of the full-system MP2 energy (`e_corr_lmp2`)
       for the usual MP2 correction of LNO energies.

    The domains of the fragments with the same domain share its integrals.
'''

import copy
from collections import namedtuple
from functools import reduce

import numpy as np

from pyscf import lib, scf
from pyscf.lib import logger

from pyscf.lno.lno import _frag_orbloc
from pyscf.dlno import domain as dom_mod
from pyscf.dlno.domain import (ao_index_by_atom, get_bp_domain, make_pao,
                               get_primary_domain, domain_pao, bp_completeness,
                               project_mo, orth_against, lowdin, canonical,
                               semicanonicalize, fake_mol_by_atom, atom_distance)
from pyscf.dlno.multipole import LMOMoments, pair_energy
from pyscf.dlno.lmp2 import get_frag_mp2, get_frag_ump2


# per-fragment domain: per-spin LMO indices of the fragment (`frag`) and of the domain
# occupied space (`occ`), the atoms of the PAOs (`pao_atm`) and of the AO basis (`ao_atm`),
# and the multipole energy of the weak pairs of the fragment LMOs (`e_dist`)
FragDomain = namedtuple('FragDomain', ['frag', 'occ', 'pao_atm', 'ao_atm', 'e_dist'])

def _domain_key(dom):
    return (tuple(dom.ao_atm), tuple(dom.pao_atm), tuple(tuple(o) for o in dom.occ))


class _DLNOData:
    r''' Domain data of the LMOs shared by all fragments (built by `build_domains`). '''
    def __init__(self):
        self.lmo = None           # per spin, LMO coefficients
        self.fock = None          # per spin, AO Fock matrix (exxdiv=None for PBC)
        self.pao = None           # per spin, PAO coefficients
        self.ao2pao = None        # per spin, AO -> PAO index (-1: discarded)
        self.lmo_bp = None        # per spin, LMO BP domains
        self.pao_bp = None        # per spin, PAO BP domains
        self.lmo_pd = None        # per spin, LMO primary domains
        self.pair_energy = None   # [s][t] (nlmo_s, nlmo_t) multipole pair energies
        self.pair_mask = None     # [s][t] which pair energies were computed


class DLNO:
    r''' Domain-based LNO, a mixin to put in front of an LNO method class, e.g.

        class DLNOCCSD(DLNO, LNOCCSD): ...

    `lo_coeff` must hold occupied LMOs spanning the active occupied space (per spin for
    unrestricted methods); fragments are groups of these LMOs. See the module doc for the
    domain construction.

    Attributes:
        lmo_bp_domain_thr : float
            BP completeness of the LMO domains. Default 0.999.
        pao_bp_domain_thr : float
            BP completeness of the PAO domains, and of the PAOs kept in a local domain.
            Default 0.98.
        pao_norm_thr : float
            PAOs with a norm below this are discarded. Default 1e-4.
        domain_pao_thr : float
            In a local domain, PAO combinations whose overlap with the AO space of their
            atoms is below this are discarded. Default 1e-4.
        pao_orth_thr : float
            Overlap eigenvalue threshold for the canonical orthogonalization of PAOs.
            Default 1e-6.
        pair_energy_thr : float
            LMO pairs with a multipole MP2 pair energy above this (for a pair of doubly
            occupied LMOs; spin-orbital pairs are compared with a quarter of it) are strong.
            Default 1e-4.
        multipole_order : int
            Order of the multipole expansion of the pair energies (2, 3 or 4). Default 4.
        multipole_domain : str
            Atoms whose AOs expand an LMO and its PAOs for the multipole moments:
            'primary' (its primary domain) or 'bp' (its BP domain, cheaper when the
            primary domains are large, e.g. in solids). Default 'primary'.
        domain_basis : str
            'local' (molecules only): the domain is solved in the AO and auxiliary basis
            of its extended primary domain atoms. 'full': in the full AO basis.
        full_domain_fallback : bool
            Fragments whose virtual domain (PAO atoms) covers the whole system are solved
            as in the parent LNO method, from the full-system integrals (with all occupied
            orbitals). Their local MP2 energies are then taken from the full-system MP2
            (`full_mp2`): exact for `e_corr_lmp2` when all fragments are full-domain, else
            left undefined (NaN). Default False (True for KDLNO).
        full_mp2 : float
            Full-system MP2 correlation energy (per cell for KDLNO) for the fallback above;
            computed when needed if not given.

    Results (after `kernel`, weighted by `frag_wghtlist`):
        efrag_lmp2      : local MP2 fragment energies in their domains
        efrag_lmp2_dist : multipole MP2 energies of the weak pairs of each fragment
        e_corr_lmp2     : the local MP2 correlation energy (their sum), which replaces
                          the full-system MP2 energy in the MP2 correction of LNO energies,
                          e.g. `e_corr_ccsd_t_pt2corrected(e_corr_lmp2)`.
        frag_domain_info: per fragment, the sizes of its domain and LNO space.
    '''

    lmo_bp_domain_thr = 0.999
    pao_bp_domain_thr = 0.98
    pao_norm_thr = 1e-4
    domain_pao_thr = 1e-4
    pao_orth_thr = 1e-6
    pair_energy_thr = 1e-4
    multipole_order = 4
    multipole_domain = 'primary'
    domain_basis = 'local'
    full_domain_fallback = False
    full_mp2 = None
    frag_full_domain = None

    _dlno = None
    efrag_lmp2 = None
    efrag_lmp2_dist = None
    frag_domain_info = None

    # ------------------------------------------------------------------------------
    # spin handling
    @property
    def _nspin(self):
        return 2 if np.ndim(self._scf.mo_occ) == 2 else 1

    def _per_spin(self, x):
        return [x] if self._nspin == 1 else list(x)

    def _frag_lmo_idx(self, loidx):
        if self._nspin == 1:
            return [np.asarray(loidx, dtype=int)]
        return [np.asarray(loidx[0], dtype=int), np.asarray(loidx[1], dtype=int)]

    def _split_mo_spin(self):
        r''' Per spin (frozen occ, active occ, active vir, frozen vir) MO coefficients. '''
        return self._per_spin(self.split_mo_coeff()) if self._nspin == 2 else \
               [self.split_mo_coeff()]

    def _pair_prefac(self):
        return 4. if self._nspin == 1 else 1.

    # ------------------------------------------------------------------------------
    # geometry hooks (overridden for PBC)
    def _atom_distance(self):
        return atom_distance(self.mol)

    def _multipole_mol(self, atmlst, center_atom):
        return fake_mol_by_atom(self.mol, atmlst)

    def _pair_vector(self, ri, rj):
        return rj - ri

    @property
    def _domain_basis_full(self):
        basis = self.domain_basis.lower()
        if basis not in ('local', 'full'):
            raise ValueError('domain_basis must be "local" or "full" (got %s)'
                             % self.domain_basis)
        return basis == 'full'

    # ------------------------------------------------------------------------------
    # Fock matrix
    def _parent_fock(self):
        r''' Per spin AO Fock matrix of the mean-field reference (exxdiv=None for PBC). '''
        if self._nspin == 1:
            from pyscf.lno.lnoccsd import fock_from_mo
            return [fock_from_mo(self._scf, s1e=self.s1e, force_exxdiv_none=True)]
        from pyscf.lno.ulnoccsd import fock_from_mo
        return list(fock_from_mo(self._scf, s1e=self.s1e))

    def _occ_energy_shift(self):
        r''' Shift of the occupied orbital energies from the Fock matrix of `_parent_fock`
        to those of the mean-field reference (the Madelung shift for PBC with
        exxdiv='ewald').
        '''
        return 0.

    # ------------------------------------------------------------------------------
    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        log = logger.new_logger(self, verbose)
        log.info('******** DLNO domains ********')
        log.info('domain_basis = %s', self.domain_basis)
        log.info('lmo_bp_domain_thr = %s', self.lmo_bp_domain_thr)
        log.info('pao_bp_domain_thr = %s', self.pao_bp_domain_thr)
        log.info('pao_norm_thr = %s', self.pao_norm_thr)
        log.info('domain_pao_thr = %s', self.domain_pao_thr)
        log.info('pao_orth_thr = %s', self.pao_orth_thr)
        log.info('pair_energy_thr = %s', self.pair_energy_thr)
        log.info('multipole_order = %s  multipole_domain = %s', self.multipole_order,
                 self.multipole_domain)
        return self

    def _check_amp(self, amp, name):
        amp = super()._check_amp(amp, name)
        if amp != 'mp2':
            raise NotImplementedError('%s = %s: DLNO only supports MP2 amplitudes.'
                                      % (name, amp))
        return amp

    def reset_domains(self):
        self._dlno = None
        return self

    # ------------------------------------------------------------------------------
    # LMO domains and pair screening
    def _check_lmo(self, lmo, orbocc, s):
        log = logger.new_logger(self)
        s1e = self.s1e
        if lmo.shape[1] != orbocc.shape[1]:
            log.error('DLNO: spin %d has %d LOs but %d active occupied MOs. `lo_coeff` must '
                      'be the occupied LMOs spanning the active occupied space.',
                      s, lmo.shape[1], orbocc.shape[1])
            raise ValueError('lo_coeff must span the active occupied space')
        sv = np.linalg.svd(np.dot(orbocc.T.conj(), np.dot(s1e, lmo)), compute_uv=False)
        if sv.size and abs(sv - 1).max() > 1e-6:
            log.error('DLNO: the LOs of spin %d are not an orthonormal basis of the '
                      'active occupied space (max |sigma - 1| = %.2e).', s, abs(sv-1).max())
            raise ValueError('lo_coeff must span the active occupied space')

    def build_domains(self):
        r''' BP domains of the LMOs and PAOs, primary domains, and multipole pair energies
        of the fragment LMOs (stored in `self._dlno`).
        '''
        log = logger.new_logger(self)
        cput0 = cput1 = (logger.process_clock(), logger.perf_counter())
        mol = self.mol
        s1e = self.s1e
        aoslices = mol.aoslice_by_atom()
        dist = self._atom_distance()
        nspin = self._nspin

        d = _DLNOData()
        d.lmo = self._per_spin(self.lo_coeff)
        d.fock = self._parent_fock()
        splits = self._split_mo_spin()
        d.pao, d.ao2pao, d.lmo_bp, d.pao_bp, d.lmo_pd = [], [], [], [], []
        for s in range(nspin):
            frzocc, actocc, actvir, frzvir = splits[s]
            self._check_lmo(d.lmo[s], actocc, s)
            pao, ao2pao = make_pao(s1e, actvir, self.pao_norm_thr)
            lmo_bp = get_bp_domain(mol, d.lmo[s], s1e, self.lmo_bp_domain_thr, dist,
                                   aoslices=aoslices)
            pao_bp = get_bp_domain(mol, pao, s1e, self.pao_bp_domain_thr, dist,
                                   aoslices=aoslices)
            lmo_pd = get_primary_domain(mol, lmo_bp, pao_bp, ao2pao, aoslices)
            d.pao.append(pao)
            d.ao2pao.append(ao2pao)
            d.lmo_bp.append(lmo_bp)
            d.pao_bp.append(pao_bp)
            d.lmo_pd.append(lmo_pd)
            log.info('DLNO spin %d: %d LMOs  %d PAOs', s, d.lmo[s].shape[1], pao.shape[1])
            log.info('    LMO BP domains      : avg %.1f  max %d atoms',
                     np.mean([len(x) for x in lmo_bp]), max(len(x) for x in lmo_bp))
            log.info('    PAO BP domains      : avg %.1f  max %d atoms',
                     np.mean([len(x) for x in pao_bp]), max(len(x) for x in pao_bp))
            log.info('    LMO primary domains : avg %.1f  max %d atoms',
                     np.mean([len(x) for x in lmo_pd]), max(len(x) for x in lmo_pd))
        cput1 = log.timer('DLNO BP domains', *cput1)

        # multipole pair energies of the fragment LMOs with all LMOs
        moments = [[self._lmo_moments(d, s, i, aoslices) for i in range(d.lmo[s].shape[1])]
                   for s in range(nspin)]
        cput1 = log.timer('DLNO LMO moments', *cput1)
        rows = [set() for s in range(nspin)]
        for loidx in self.frag_lolist:
            for s, idx in enumerate(self._frag_lmo_idx(loidx)):
                rows[s].update(idx.tolist())
        prefac = self._pair_prefac()
        nlmo = [d.lmo[s].shape[1] for s in range(nspin)]
        d.pair_energy = [[np.zeros((nlmo[s], nlmo[t])) for t in range(nspin)]
                         for s in range(nspin)]
        d.pair_mask = [[np.zeros((nlmo[s], nlmo[t]), dtype=bool) for t in range(nspin)]
                       for s in range(nspin)]
        for s in range(nspin):
            for i in sorted(rows[s]):
                mi = moments[s][i]
                for t in range(nspin):
                    for j in range(nlmo[t]):
                        if s == t and i == j:
                            continue
                        if d.pair_mask[t][s][j,i]:
                            e = d.pair_energy[t][s][j,i]
                        else:
                            mj = moments[t][j]
                            e = pair_energy(mi, mj, self._pair_vector(mi.center, mj.center),
                                            prefac)
                        d.pair_energy[s][t][i,j] = e
                        d.pair_mask[s][t][i,j] = True
        log.timer('DLNO pair energies', *cput1)
        self._dlno = d
        log.timer('DLNO domains', *cput0)
        return d

    def _lmo_moments(self, d, s, i, aoslices):
        r''' Multipole moments of LMO i (spin s) with the semicanonical PAOs of its BP
        domain atoms, projected onto its primary (or BP, see `multipole_domain`) domain.
        '''
        mol = self.mol
        s1e = self.s1e
        if self.multipole_domain.lower() == 'primary':
            pd = d.lmo_pd[s][i]
        elif self.multipole_domain.lower() == 'bp':
            pd = d.lmo_bp[s][i]
        else:
            raise ValueError('multipole_domain must be "primary" or "bp"')
        ao_idx = ao_index_by_atom(mol, pd, aoslices)
        s21 = s1e[ao_idx]
        s22 = s1e[np.ix_(ao_idx, ao_idx)]
        f22 = d.fock[s][np.ix_(ao_idx, ao_idx)]
        lmo_i = lowdin(project_mo(d.lmo[s][:,i:i+1], s21, s22), s22)
        p = domain_pao(mol, d.pao[s], d.ao2pao[s], s1e, d.lmo_bp[s][i],
                       ovlp_thr=self.domain_pao_thr, orth_thr=self.pao_orth_thr,
                       aoslices=aoslices)
        if p.shape[1] > 0:
            av = bp_completeness(s1e, p, ao_idx)
            p = p[:,av > self.pao_bp_domain_thr - 1e-10]
        p = project_mo(p, s21, s22)
        p = canonical(orth_against(lmo_i, p, s22), s22, self.pao_orth_thr)
        e_occ = np.dot(lmo_i[:,0].conj(), np.dot(f22, lmo_i[:,0])).real
        e_vir, p = semicanonicalize(p, f22)
        pops = abs(dom_mod.atom_population(mol, d.lmo[s][:,i:i+1], s1e, aoslices)[:,0])
        fmol = self._multipole_mol(pd, int(np.argmax(pops)))
        return LMOMoments(fmol, e_occ, lmo_i, e_vir, p, order=self.multipole_order)

    def _strong(self, d, s, i, t):
        r''' Strong pair partners (spin t) of LMO i of spin s. '''
        thr = self.pair_energy_thr if self._nspin == 1 else self.pair_energy_thr / 4.
        mask = abs(d.pair_energy[s][t][i]) > thr
        if s == t:
            mask[i] = True
        return np.where(mask)[0]

    def frag_domain(self, loidx):
        r''' The domain (`FragDomain`) of the fragment with LMOs `loidx`. '''
        d = self._dlno
        nspin = self._nspin
        frag = self._frag_lmo_idx(loidx)
        occ = [set(frag[t].tolist()) for t in range(nspin)]
        for s in range(nspin):
            for i in frag[s]:
                for t in range(nspin):
                    occ[t].update(self._strong(d, s, i, t).tolist())
        occ = [np.asarray(sorted(o), dtype=int) for o in occ]
        pao_atm = reduce(np.union1d, [d.lmo_bp[t][j] for t in range(nspin) for j in occ[t]])
        ao_atm = reduce(np.union1d, [d.lmo_pd[t][j] for t in range(nspin) for j in occ[t]])
        e_dist = 0.
        for s in range(nspin):
            for i in frag[s]:
                for t in range(nspin):
                    weak = np.ones(d.lmo[t].shape[1], dtype=bool)
                    weak[occ[t]] = False
                    e_dist += 0.5 * d.pair_energy[s][t][i,weak].sum()
        return FragDomain(frag, occ, np.asarray(pao_atm, dtype=int),
                          np.asarray(ao_atm, dtype=int), e_dist)

    # ------------------------------------------------------------------------------
    # domain problem
    def _domain_orbitals(self, dom):
        r''' Semicanonical occupied and virtual orbitals of a domain, and its fragment LOs,
        in the AO basis of the domain.

        Returns:
            ao_idx (None for the full AO basis), s1e and per-spin lists of
            (orbocc, moeocc, orbvir, moevir, orbloc).
        '''
        d = self._dlno
        mol = self.mol
        s1e = self.s1e
        aoslices = mol.aoslice_by_atom()
        full = self._domain_basis_full
        if full:
            ao_idx = None
            S = s1e
            F = d.fock
        else:
            ao_idx = ao_index_by_atom(mol, dom.ao_atm, aoslices)
            s21 = s1e[ao_idx]
            S = s1e[np.ix_(ao_idx, ao_idx)]
            F = [f[np.ix_(ao_idx, ao_idx)] for f in d.fock]
        shift = self._occ_energy_shift()
        out = []
        for s in range(self._nspin):
            occ = d.lmo[s][:,dom.occ[s]]
            loc = d.lmo[s][:,dom.frag[s]]
            p = domain_pao(mol, d.pao[s], d.ao2pao[s], s1e, dom.pao_atm,
                           ovlp_thr=None if full else self.domain_pao_thr,
                           orth_thr=self.pao_orth_thr, aoslices=aoslices)
            if not full:
                occ = lowdin(project_mo(occ, s21, S), S)
                loc = lowdin(project_mo(loc, s21, S), S)
                if p.shape[1] > 0:
                    av = bp_completeness(s1e, p, ao_idx)
                    p = p[:,av > self.pao_bp_domain_thr - 1e-10]
                p = orth_against(occ, project_mo(p, s21, S), S)
            p = canonical(p, S, self.pao_orth_thr)
            eo, occ = semicanonicalize(occ, F[s])
            ev, vir = semicanonicalize(p, F[s])
            out.append((occ, eo + shift, vir, ev, loc))
        return ao_idx, S, out

    def _auxbasis(self):
        with_df = self.with_df
        auxmol = getattr(with_df, 'auxmol', None)
        if auxmol is not None:
            return auxmol._basis
        from pyscf import df
        auxbasis = with_df.auxbasis
        if auxbasis is None:
            auxbasis = df.make_auxbasis(self.mol)
        return df.addons.make_auxmol(self.mol, auxbasis)._basis

    def make_domain(self, dom, df_cache=None):
        r''' The LNO object of a domain: a copy of `self` whose mean-field reference holds
        the domain orbitals (and, for a local domain, the domain AO and auxiliary basis).

        Returns:
            dlno : the domain LNO object
            orbloc : the fragment LOs in the AO basis of the domain
            df_cache : (key, DF object) to reuse the DF integrals of a local domain
        '''
        ao_idx, S, orbs = self._domain_orbitals(dom)
        nspin = self._nspin
        full = ao_idx is None

        # pad the spin with fewer domain MOs with frozen virtual orbitals so that both
        # spins have the same number of MOs
        frozen = None
        if nspin == 2:
            nmo = [o[0].shape[1] + o[2].shape[1] for o in orbs]
            frozen = [[], []]
            for s in range(2):
                npad = max(nmo) - nmo[s]
                if npad > 0:
                    occ, eo, vir, ev, loc = orbs[s]
                    pad = _orth_complement(np.hstack((occ, vir)), S, npad)
                    emax = ev.max() if ev.size else eo.max()
                    frozen[s] = list(range(nmo[s], nmo[s]+npad))
                    orbs[s] = (occ, eo, np.hstack((vir, pad)),
                               np.concatenate((ev, np.full(npad, emax + 10.))), loc)

        mo_coeff = [np.hstack((o[0], o[2])) for o in orbs]
        mo_energy = [np.concatenate((o[1], o[3])) for o in orbs]
        occ_val = 2 if nspin == 1 else 1
        mo_occ = [np.concatenate((np.full(o[0].shape[1], occ_val), np.zeros(o[2].shape[1])))
                  for o in orbs]
        orbloc = [o[4] for o in orbs]

        if full:
            dmf = copy.copy(self._scf)
            s1e = self.s1e
            dmf.get_ovlp = lambda *args, **kwargs: s1e
            h1e, vhf = self._h1e, self._vhf
            with_df = self.with_df
        else:
            fmol = fake_mol_by_atom(self.mol, dom.ao_atm)
            key = tuple(dom.ao_atm)
            if df_cache is not None and df_cache[0] == key:
                with_df = df_cache[1]
            else:
                from pyscf import df
                with_df = df.DF(fmol, auxbasis=self._auxbasis())
                with_df.max_memory = self.max_memory
                with_df.verbose = self.verbose
                with_df.stdout = self.stdout
                df_cache = (key, with_df)
            dmf = (scf.RHF(fmol) if nspin == 1 else scf.UHF(fmol))
            dmf = dmf.density_fit()
            dmf.with_df = with_df
            dmf.max_memory = self.max_memory
            dmf.verbose = self.verbose
            dmf.stdout = self.stdout
            nao = S.shape[0]
            dmf.get_ovlp = lambda *args, **kwargs: S
            dmf.get_hcore = lambda *args, **kwargs: np.zeros((nao,nao))
            s1e = S
            h1e = np.zeros((nao,nao))
            vhf = None
        if nspin == 1:
            dmf.mo_coeff, dmf.mo_energy, dmf.mo_occ = mo_coeff[0], mo_energy[0], mo_occ[0]
        else:
            dmf.mo_coeff = np.asarray(mo_coeff)
            dmf.mo_energy = np.asarray(mo_energy)
            dmf.mo_occ = np.asarray(mo_occ)
        dmf.converged = True
        dmf.e_tot = self._scf.e_tot
        if vhf is None:
            # domain Fock matrix: the full-system one in the (semicanonical) domain orbitals,
            # i.e., without occupied-virtual coupling
            if nspin == 1:
                from pyscf.lno.lnoccsd import fock_from_mo
                vhf = fock_from_mo(dmf, s1e=s1e, force_exxdiv_none=False) - h1e
            else:
                from pyscf.lno.ulnoccsd import fock_from_mo
                vhf = np.asarray(fock_from_mo(dmf, s1e=s1e)) - h1e

        dl = copy.copy(self)
        dl._scf = dmf
        dl.mol = dmf.mol
        dl.with_df = with_df
        dl.frozen = frozen
        dl._mo_occ = dl._mo_coeff = dl._mo_energy = None
        dl._nmo = dl._nocc = None
        dl._s1e, dl._h1e, dl._vhf = s1e, h1e, vhf
        dl.chkfile = None
        dl._ovL = dl._ovL_to_save = None
        dl._rpa_eris = None
        dl._dlno = None
        if hasattr(dl, 'k_eris'):
            dl.k_eris = False
        return dl, (orbloc[0] if nspin == 1 else orbloc), df_cache

    def get_frag_lmp2(self, dl, eris, orbloc):
        r''' Local MP2 energy of the fragment LOs `orbloc` in the domain `dl`. '''
        s1e = dl.s1e
        if self._nspin == 1:
            orbocc = dl.split_mo_coeff()[1]
            moeocc, moevir = dl.split_mo_energy()[1:3]
            u = np.dot(orbocc.T.conj(), np.dot(s1e, orbloc))
            return get_frag_mp2(eris, moeocc, moevir, u, max_memory=self.max_memory)
        mos = dl.split_mo_coeff()
        moes = dl.split_mo_energy()
        u = [np.dot(mos[s][1].T.conj(), np.dot(s1e, orbloc[s])) for s in range(2)]
        return get_frag_ump2(eris, [moes[s][1] for s in range(2)],
                             [moes[s][2] for s in range(2)], u, max_memory=self.max_memory)

    def _full_domain(self, dom):
        r''' Whether the fragment of domain `dom` falls back to the parent LNO method (see
        `full_domain_fallback`). '''
        return bool(self.full_domain_fallback) and len(dom.pao_atm) == self.mol.natm

    def _nlo_per_unit(self):
        r''' Number of LOs per unit of the full-system energies (the whole system). '''
        return float(sum(x.shape[1] for x in self._per_spin(self.lo_coeff)))

    def get_full_mp2(self):
        r''' Full-system MP2 correlation energy (see `full_mp2`). '''
        if self.full_mp2 is None:
            raise NotImplementedError('Set `full_mp2` for the full-domain fallback.')
        return self.full_mp2

    # ------------------------------------------------------------------------------
    # driver
    def _solve_frags(self, eris=None):
        log = logger.new_logger(self)
        cput0 = cput1 = (logger.process_clock(), logger.perf_counter())
        if eris is not None:
            log.warn('DLNO does not use full-system integrals; the given eris is ignored.')
        if self.frag_lno is not None:
            log.warn('DLNO cannot restart from saved LNOs (they live in the domains); '
                     '`frag_lno` is ignored.')
            self.frag_lno = None
        save_lno, self.save_lno = self.save_lno, False

        nfrag = self.nfrag
        lno_pct_occ = [None, None] if self.lno_pct_occ is None else self.lno_pct_occ
        lno_norb = [[None,None]] * nfrag if self.lno_norb is None else self.lno_norb
        frag_lno, frag_eners, frag_res = self._restart_frags(self.lo_coeff, self.frag_lolist)

        if self._dlno is None:
            self.build_domains()
        cput1 = log.timer('DLNO build domains', *cput1)
        doms = [self.frag_domain(loidx) for loidx in self.frag_lolist]
        nspin = self._nspin

        full = np.asarray([self._full_domain(dom) for dom in doms], dtype=bool)
        self.frag_full_domain = full
        if full.any():
            log.info('DLNO: %d/%d fragments have a full-system domain and are solved as in '
                     'the parent LNO method', full.sum(), nfrag)
        efrag_lmp2 = np.zeros(nfrag)
        efrag_dist = np.asarray([0. if f else dom.e_dist for dom, f in zip(doms, full)])
        lmp2_done = np.zeros(nfrag, dtype=bool)
        for i, dic in enumerate(frag_eners):
            if dic is not None and dic.get('e_lmp2') is not None:
                efrag_lmp2[i] = float(dic['e_lmp2'])
                lmp2_done[i] = True
        self.frag_domain_info = [None] * nfrag

        # domain fragments sorted by domain (to share integrals), full-domain ones last
        order = sorted(range(nfrag), key=lambda i: (full[i], _domain_key(doms[i])))
        cache_key = dl = eris_dom = None
        df_cache = None
        eris_full = None
        for ifrag in order:
            if frag_res[ifrag] is not None and lmp2_done[ifrag]:
                log.info('Fragment %d/%d  Sol: restored', ifrag+1, nfrag)
                continue
            if full[ifrag]:
                if eris_dom is not None:
                    dl._release_full_eris(eris_dom)
                dl = eris_dom = cache_key = df_cache = None
                if eris_full is None:
                    eris_full = self.ao2mo()
                frag_res[ifrag], info = self._solve_full_domain_frag(ifrag, eris_full,
                                                                     lno_pct_occ, lno_norb)
                efrag_lmp2[ifrag] = np.nan
                dic = self._frag_res_to_dict(frag_res[ifrag]) or {}
                dic['e_lmp2'] = np.nan
                dic['e_lmp2_dist'] = 0.
                frag_eners[ifrag] = dic
                self._chk_dump_frag(ifrag, 'energy', dic)
                self.frag_domain_info[ifrag] = info
                cput1 = log.timer('Fragment %d' % (ifrag+1) + ' '*(8-len(str(ifrag+1))), *cput1)
                continue
            dom = doms[ifrag]
            key = _domain_key(dom)
            if key != cache_key:
                if eris_dom is not None:
                    dl._release_full_eris(eris_dom)
                dl = eris_dom = None
                dl, orbloc, df_cache = self.make_domain(dom, df_cache)
                eris_dom = dl.ao2mo()
                cache_key = key
                cput1 = log.timer('Fragment %d domain' % (ifrag+1), *cput1)
            else:
                orbloc = self._domain_frag_lo(dl, dom)
            nocc_d = [int(np.count_nonzero(m > 1e-10)) for m in self._per_spin(dl.mo_occ)]
            nmo_d = [m.size for m in self._per_spin(dl.mo_occ)]
            info = dict(natm_ao=len(dom.ao_atm), natm_pao=len(dom.pao_atm),
                        nao=dl.mol.nao_nr(), nocc=nocc_d, nvir=[n-o for n,o in zip(nmo_d,nocc_d)])
            log.info('Fragment %d/%d  Domain: %d/%d atoms (AO) | %d atoms (PAO) | %d AOs | '
                     'occ %s | vir %s', ifrag+1, nfrag, len(dom.ao_atm), self.mol.natm,
                     len(dom.pao_atm), info['nao'], info['nocc'], info['nvir'])

            # local MP2 of the fragment in its domain
            elmp2 = self.get_frag_lmp2(dl, eris_dom, orbloc)
            efrag_lmp2[ifrag] = float(elmp2)
            cput1 = log.timer_debug1('Fragment %d local MP2' % (ifrag+1), *cput1)

            # LNOs of the fragment in its domain and the impurity problem
            loidx = self.frag_lolist[ifrag]
            lno_param = _frag_orbloc(self.lo_coeff, loidx, ifrag, self.lno_thresh,
                                     lno_pct_occ, lno_norb)[1]
            lnofrag, frozen, uocc_loc, frag_msg = dl.make_las(eris_dom, orbloc, self.lno_type,
                                                              lno_param)
            info['las'] = frag_msg
            log.info('Fragment %d/%d  LAS: %s', ifrag+1, nfrag, frag_msg)
            frag_res[ifrag], frag_msg = dl.impurity_solve(dl._scf, lnofrag, uocc_loc, eris_dom,
                                                          frozen=frozen)
            lnofrag = frozen = uocc_loc = None
            dic = self._frag_res_to_dict(frag_res[ifrag])
            if dic is None:
                dic = {}
            dic['e_lmp2'] = efrag_lmp2[ifrag]
            dic['e_lmp2_dist'] = efrag_dist[ifrag]
            frag_eners[ifrag] = dic
            self._chk_dump_frag(ifrag, 'energy', dic)
            log.info('Fragment %d/%d  Sol: %s  E_corr(lMP2) = %.15g  E_corr(dist) = %.15g',
                     ifrag+1, nfrag, frag_msg, efrag_lmp2[ifrag], efrag_dist[ifrag])
            self.frag_domain_info[ifrag] = info
            cput1 = log.timer('Fragment %d' % (ifrag+1) + ' '*(8-len(str(ifrag+1))), *cput1)

        if eris_dom is not None:
            dl._release_full_eris(eris_dom)
        if eris_full is not None:
            self._release_full_eris(eris_full)
        dl = eris_dom = eris_full = None
        self.save_lno = save_lno
        self.efrag_lmp2 = efrag_lmp2
        self.efrag_lmp2_dist = efrag_dist
        classname = self.__class__.__name__
        log.timer(classname+' '*(17-len(classname)), *cput0)
        return frag_res

    def _solve_full_domain_frag(self, ifrag, eris, lno_pct_occ, lno_norb):
        r''' LNOs and impurity problem of a full-domain fragment, as in the parent LNO
        method, from the full-system integrals `eris`. '''
        log = logger.new_logger(self)
        nfrag = self.nfrag
        loidx = self.frag_lolist[ifrag]
        orbloc, lno_param = _frag_orbloc(self.lo_coeff, loidx, ifrag, self.lno_thresh,
                                         lno_pct_occ, lno_norb)
        lnofrag, frozen, uocc_loc, las_msg = self.make_las(eris, orbloc, self.lno_type,
                                                           lno_param)
        log.info('Fragment %d/%d  Domain: full system', ifrag+1, nfrag)
        log.info('Fragment %d/%d  LAS: %s', ifrag+1, nfrag, las_msg)
        res, msg = self.impurity_solve(self._scf, lnofrag, uocc_loc, eris, frozen=frozen)
        log.info('Fragment %d/%d  Sol: %s', ifrag+1, nfrag, msg)
        nocc = [int(np.count_nonzero(m > 1e-10)) for m in self._per_spin(self.mo_occ)]
        nmo = [m.size for m in self._per_spin(self.mo_occ)]
        info = dict(full_domain=True, natm_ao=self.mol.natm, natm_pao=self.mol.natm,
                    nao=self.mol.nao_nr(), nocc=nocc, nvir=[n-o for n,o in zip(nmo,nocc)],
                    las=las_msg)
        return res, info

    def _domain_frag_lo(self, dl, dom):
        r''' The fragment LOs of `dom` in the AO basis of the domain object `dl`. '''
        d = self._dlno
        out = []
        for s in range(self._nspin):
            loc = d.lmo[s][:,dom.frag[s]]
            if not self._domain_basis_full:
                ao_idx = ao_index_by_atom(self.mol, dom.ao_atm)
                s1e = self.s1e
                S = s1e[np.ix_(ao_idx, ao_idx)]
                loc = lowdin(project_mo(loc, s1e[ao_idx], S), S)
            out.append(loc)
        return out[0] if self._nspin == 1 else out

    def _post_proc(self, frag_res, frag_wghtlist):
        super()._post_proc(frag_res, frag_wghtlist)
        efrag_lmp2 = np.array(self.efrag_lmp2, dtype=float)
        nan = np.isnan(efrag_lmp2)
        if nan.all():
            # all fragments full-domain: share the full-system MP2 (of one unit) by the
            # number of LOs, which reproduces it in e_corr_lmp2 for the units the fragments
            # cover (e.g. one cell, or the whole supercell)
            nlo = np.asarray([sum(len(x) for x in self._frag_lmo_idx(loidx))
                              for loidx in self.frag_lolist], dtype=float)
            efrag_lmp2 = self.get_full_mp2() * nlo / self._nlo_per_unit()
        elif nan.any():
            logger.warn(self, 'DLNO: the local MP2 of the %d full-domain fragments is not '
                        'defined when other fragments are not; e_corr_lmp2 is NaN.',
                        nan.sum())
        self.efrag_lmp2 = efrag_lmp2 * frag_wghtlist
        self.efrag_lmp2_dist = self.efrag_lmp2_dist * frag_wghtlist

    @property
    def e_corr_lmp2(self):
        r''' Local MP2 correlation energy: the fragment MP2 energies in their domains plus
        the multipole energies of the weak pairs. '''
        return float(np.sum(self.efrag_lmp2) + np.sum(self.efrag_lmp2_dist))

    @property
    def e_corr_lmp2_dist(self):
        return float(np.sum(self.efrag_lmp2_dist))

    def _finalize(self):
        super()._finalize()
        log = logger.new_logger(self)
        elmp2 = self.e_corr_lmp2
        log.note('E_corr(lMP2) = %.15g  (weak pairs %.15g)', elmp2, self.e_corr_lmp2_dist)
        if hasattr(self, 'e_corr_ccsd_pt2corrected'):
            log.note('E_corr(DLNOCCSD+dlMP2) = %.15g', self.e_corr_ccsd_pt2corrected(elmp2))
            if getattr(self, 'ccsd_t', False):
                log.note('E_corr(DLNOCCSD_T+dlMP2) = %.15g',
                         self.e_corr_ccsd_t_pt2corrected(elmp2))
        return self


def _orth_complement(orb, s, n):
    r''' `n` orthonormal orbitals (metric `s`) orthogonal to the orthonormal `orb`. '''
    nao = s.shape[0]
    x = canonical(np.eye(nao), s, 1e-10)
    x = orth_against(orb, x, s)
    sx = np.dot(x.T.conj(), np.dot(s, x))
    e, u = np.linalg.eigh(sx)
    u = u[:,::-1][:,:n]
    e = e[::-1][:n]
    return np.dot(x, u) / np.sqrt(e)
