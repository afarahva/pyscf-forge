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

r''' Domain-based LNO-CCSD and LNO-CCSD(T), spin-restricted and unrestricted.

    These are the LNO-CCSD(T) classes of `pyscf.lno` whose LNOs are built in the domain of
    each fragment (see `pyscf.dlno.dlno`). Usage is the same as `pyscf.lno.LNOCCSD`, with
    `lo_coeff` the occupied LMOs (all of them) and fragments made of LMOs:

        mcc = DLNOCCSD_T(mf, lo_coeff, [[i] for i in range(nlmo)], frozen=frozen)
        mcc.lno_thresh = [1e-5, 1e-6]
        mcc.kernel()
        e_corr = mcc.e_corr_ccsd_t_pt2corrected(mcc.e_corr_lmp2)
'''

from pyscf.lno.lnoccsd import LNOCCSD
from pyscf.lno.ulnoccsd import ULNOCCSD
from pyscf.dlno.dlno import DLNO


class DLNOCCSD(DLNO, LNOCCSD):
    pass

class DLNOCCSD_T(DLNOCCSD):
    def __init__(self, *args, **kwargs):
        DLNOCCSD.__init__(self, *args, **kwargs)
        self.ccsd_t = True


class DULNOCCSD(DLNO, ULNOCCSD):
    pass

class DULNOCCSD_T(DULNOCCSD):
    def __init__(self, *args, **kwargs):
        DULNOCCSD.__init__(self, *args, **kwargs)
        self.ccsd_t = True
