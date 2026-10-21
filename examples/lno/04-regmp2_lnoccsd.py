#!/usr/bin/env python
r'''
kappa-regularized MP2 (kappa-MP2) in restricted LNO-CCSD: behavior of the
low-level amplitude settings.

See LNO.__init__ for the settings and examples/lno/06-regmp2_dimer_compare.py
for the physical motivation:
    lno_amp : amplitudes for the LNO density matrices (default 'mp2')
    pt2_amp : amplitudes for the fragment MP2 energy and the CCSD guess
              (default None: same as lno_amp)

This script sanity-checks the implementation on a single water molecule:
  * with a huge kappa the damping factor g -> 1, so lno_amp='kappa-mp2' must
    reproduce the default (MP2) result exactly;
  * pt2_amp=None follows lno_amp, i.e. equals pt2_amp='kappa-mp2';
  * pt2_amp='kappa-mp2' alone changes the reported fragment MP2 energy
    (kappa-MP2 energy), while lno_amp='kappa-mp2' with pt2_amp='mp2' leaves it
    essentially unchanged (only the LNO active-space selection is affected).
'''

import numpy as np
from pyscf import gto, scf, lno, lo

mol = gto.M(atom='''O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24''',
            basis='cc-pvdz', verbose=3)
mf = scf.RHF(mol).density_fit().run()
frozen = 1

orbocc = mf.mo_coeff[:, frozen:np.count_nonzero(mf.mo_occ)]
lo_coeff = lo.PipekMezey(mol, orbocc).kernel()
frag_lolist = [[i] for i in range(lo_coeff.shape[1])]

def run(**kw):
    mcc = lno.LNOCCSD(mf, lo_coeff, frag_lolist, frozen=frozen)
    mcc.lno_thresh = [1e-4, 1e-5]
    for k, v in kw.items():
        setattr(mcc, k, v)
    mcc.kernel()
    return mcc.e_corr_pt2, mcc.e_corr_ccsd

base = run()
big  = run(lno_amp='kappa-mp2', kappa=1e6)
full = run(lno_amp='kappa-mp2', kappa=1.1)
lno_only = run(lno_amp='kappa-mp2', pt2_amp='mp2', kappa=1.1)
cc_only  = run(pt2_amp='kappa-mp2', kappa=1.1)
both     = run(lno_amp='kappa-mp2', pt2_amp='kappa-mp2', kappa=1.1)

print()
print('                                          mp2              ccsd')
print('default (MP2)                        % .10f  % .10f' % base)
print('lno_amp=kappa-mp2         kappa=1e6  % .10f  % .10f' % big)
print('lno_amp=kappa-mp2         kappa=1.1  % .10f  % .10f' % full)
print('lno_amp=kappa, pt2_amp=mp2  kappa=1.1  % .10f  % .10f' % lno_only)
print('pt2_amp=kappa-mp2         kappa=1.1  % .10f  % .10f' % cc_only)
print('lno_amp=pt2_amp=kappa-mp2 kappa=1.1  % .10f  % .10f' % both)

assert np.allclose(base, big,  atol=1e-8), 'large-kappa must match baseline'
assert np.allclose(full, both, atol=1e-9), 'pt2_amp=None must follow lno_amp'
assert not np.isclose(cc_only[0], base[0], atol=1e-6), 'pt2_amp=kappa-mp2 must change MP2 energy'
print('\nall checks passed')
