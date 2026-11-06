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

r''' Domain-based KLNO-CCSD and KLNO-CCSD(T) (see `pyscf.pbc.dlno.kdlno`).

    Usage is the same as `pyscf.pbc.lno.KLNOCCSD`, with `lo_coeff` all the occupied
    supercell LMOs sorted by cell and fragments made of reference-cell LMOs:

        mcc = KDLNOCCSD_T(kmf, lo_coeff, [[i] for i in range(nlo)], mf=smf, frozen=frozen)
        mcc.kernel()
        e_corr = mcc.e_corr_ccsd_t_pt2corrected(mcc.e_corr_lmp2)    # per cell
'''

from pyscf.pbc.lno.klnoccsd import KLNOCCSD
from pyscf.pbc.dlno.kdlno import KDLNO


class KDLNOCCSD(KDLNO, KLNOCCSD):
    pass

class KDLNOCCSD_T(KDLNOCCSD):
    def __init__(self, *args, **kwargs):
        KDLNOCCSD.__init__(self, *args, **kwargs)
        self.ccsd_t = True
