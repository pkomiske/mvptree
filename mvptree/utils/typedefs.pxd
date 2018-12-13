cimport numpy as np

ctypedef np.float64_t D_t # WARNING: should match D in typedefs.pyx
ctypedef np.intp_t I_t # WARNING: should match I in typedefs.pyx

ctypedef fused DI_t:
    I_t
    D_t