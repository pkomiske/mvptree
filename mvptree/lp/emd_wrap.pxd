from __future__ import absolute_import

from mvptree.utils.typedefs cimport D_t, I_t

cpdef D_t emd_c(D_t[::1] a, D_t[::1] b, D_t[:,::1] M, I_t max_iter=?) nogil except -1