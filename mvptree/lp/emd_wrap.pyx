"""
# -*- coding: utf-8 -*-
Cython linker with C solver
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License
#
# Modified by Patrick Komiske <pkomiske@mit.edu>

import numpy as np
import warnings

cimport numpy as np
cimport cython
from libc.stdlib cimport calloc, free

cdef extern from "EMD.h":
    int EMD_wrap(int n1,int n2, double* X, double* Y,double* D, double* G, double* alpha, double* beta, double* cost, int maxIter) nogil
    cdef enum ProblemType: 
        INFEASIBLE, OPTIMAL, UNBOUNDED, MAX_ITER_REACHED

cdef check_result(int result_code):
    if result_code == OPTIMAL:
        return None

    if result_code == INFEASIBLE:
        message = 'Problem infeasible. Check that a and b are in the simplex'
    elif result_code == UNBOUNDED:
        message = 'Problem unbounded'
    elif result_code == MAX_ITER_REACHED:
        message = 'numItermax reached before optimality. Try to increase numItermax.'

    warnings.warn(message)

cpdef double emd_c(double[::1] a, double[::1] b, double[:,::1] M, int max_iter=100000) nogil except -1:
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
    max_iter : int
        The maximum number of iterations before stopping the optimization
        algorithm if it has not converged.


    Returns
    -------
    gamma: (ns x nt) ndarray
        Optimal transportation matrix for the given parameters

    """

    cdef int n1= M.shape[0]
    cdef int n2= M.shape[1]
    cdef double cost = 0

    cdef double* G = <double*> calloc(n1*n2, sizeof(double))
    cdef double* alpha = <double*> calloc(n1, sizeof(double))
    cdef double* beta = <double*> calloc(n2, sizeof(double))

    # calling the function
    cdef int result_code = EMD_wrap(n1, n2, &a[0], &b[0], &M[0,0], G, alpha, beta, &cost, max_iter)

    # free memory that we allocated
    free(G)
    free(alpha)
    free(beta)

    with gil:
        check_result(result_code)

    return cost
