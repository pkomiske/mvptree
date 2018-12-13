"""
# -*- coding: utf-8 -*-
Cython linker with C solver
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License
#
# Modified by Patrick Komiske <pkomiske@mit.edu>

from __future__ import absolute_import

import numpy as np
import warnings

cimport numpy as np
cimport cython
from libc.stdlib cimport calloc, free
from mvptree.utils.typedefs cimport D_t, I_t

cdef extern from "EMD.h" nogil:
    I_t EMD_wrap(I_t n1, I_t n2, D_t* X, D_t* Y, D_t* D, D_t* G, D_t* alpha, D_t* beta, D_t* cost, I_t maxIter)
    cdef enum ProblemType:
        INFEASIBLE, OPTIMAL, UNBOUNDED, MAX_ITER_REACHED

cdef check_result(I_t result_code):
    if result_code == OPTIMAL:
        return None

    if result_code == INFEASIBLE:
        message = 'Problem infeasible. Check that a and b are in the simplex'
    elif result_code == UNBOUNDED:
        message = 'Problem unbounded'
    elif result_code == MAX_ITER_REACHED:
        message = 'numItermax reached before optimality. Try to increase numItermax.'

    warnings.warn(message)

cpdef D_t emd_c(D_t[::1] a, D_t[::1] b, D_t[:,::1] M, I_t max_iter=100000) nogil except -1:
    """
        Solves the Earth Movers distance problem and returns the optimal transport matrix

        gamm=emd(a,b,M)

    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F

        s.t. \gamma 1 = a

             \gamma^T 1= b

             \gamma\geq 0
    where :

    - M is the metric cost matrix
    - a and b are the sample weights

    Parameters
    ----------
    a : (ns,) ndarray, float64
        source histogram
    b : (nt,) ndarray, float64
        target histogram
    M : (ns,nt) ndarray, float64
        loss matrix
    max_iter : I_t
        The maximum number of iterations before stopping the optimization
        algorithm if it has not converged.


    Returns
    -------
    gamma: (ns x nt) ndarray
        Optimal transportation matrix for the given parameters

    """

    cdef I_t n1= M.shape[0]
    cdef I_t n2= M.shape[1]
    cdef D_t cost = 0

    cdef D_t* G = <D_t*> calloc(n1*n2, sizeof(D_t))
    if G == NULL:
        raise MemoryError()

    cdef D_t* alpha = <D_t*> calloc(n1, sizeof(D_t))
    if alpha == NULL:
        free(G)
        raise MemoryError()

    cdef D_t* beta = <D_t*> calloc(n2, sizeof(D_t))
    if beta == NULL:
        free(G)
        free(alpha)
        raise MemoryError()

    # calling the function
    cdef I_t result_code = EMD_wrap(n1, n2, &a[0], &b[0], &M[0,0], G, alpha, beta, &cost, max_iter)

    # free memory that we allocated
    free(G)
    free(alpha)
    free(beta)

    with gil:
        check_result(result_code)

    return cost
