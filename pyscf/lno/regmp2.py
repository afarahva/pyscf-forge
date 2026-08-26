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

''' kappa-regularized MP2 (kappa-MP2)

    The MP2 amplitude is damped by multiplying the integral (equivalently the
    amplitude) with a single power of the regularization factor

        g_ijab = 1 - exp(-kappa * Delta_ijab)

    where Delta_ijab = e_a + e_b - e_i - e_j > 0.  In terms of the (negative)
    MP2 denominator `denom = e_i + e_j - e_a - e_b = -Delta`, this reads

        g_ijab = 1 - exp(kappa * denom_ijab).

    Because any MP2 energy or density matrix is quadratic in `t2`, a single
    power on the amplitude yields the `(1 - exp(-kappa*Delta))**2` factor on the
    energy given in the reference (arXiv:2508.15744).

    Ref: arXiv:2508.15744
'''

import numpy as np


def kappa_factor(denom, kappa):
    r''' Regularization factor g = 1 - exp(kappa * denom).

        Args:
            denom : np.ndarray
                MP2 energy denominator e_i + e_j - e_a - e_b (negative).
            kappa : float
                Regularization strength.
    '''
    return 1.0 - np.exp(kappa * denom)
