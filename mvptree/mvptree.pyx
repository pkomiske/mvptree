from __future__ import absolute_import

import numpy as np

cimport numpy as np
from libc.math cimport sqrt
from mvptree.lp.emd_wrap cimport emd_c

__all__ = ['MVPTree']

cdef enum MetricType:
    m_emd, m_euclidean

np.import_array()

cdef struct NodeData_t:
    int idx_start
    int idx_end
    int is_leaf

cdef NodeData_t nd_tmp
NodeData = np.asarray(<NodeData_t[:1]>(&nd_tmp)).dtype

# inlined euclidean metric
cdef inline double euclidean_metric(double* x, double* y, int size) nogil except -1:
    cdef double tmp, d = 0
    cdef int j
    for j in range(size):
        tmp = x[j] - y[j]
        d += tmp * tmp
    return sqrt(d)

cdef inline double euclidean_2dim(double* x, double* y) nogil except -1:
    cdef double tmp0, tmp1
    tmp0 = x[0] - y[0]
    tmp1 = x[1] - y[1]
    return sqrt(tmp0*tmp0 + tmp1*tmp1)

cdef inline void euclidean_2dim_cdists(double [:,::1] dists, 
                                       double [:,::1] xs,
                                       double [:,::1] ys) nogil:
    cdef int i, j
    cdef double* x
    for i in range(xs.shape[0]):
        x = &xs[i,0]
        for j in range(ys.shape[0]):
            dists[i,j] = euclidean_2dim(x, &ys[j,0])

cpdef double emd_metric(double[::1] zs1,
                        double[::1] zs2,
                        double[:,::1] xys1,
                        double[:,::1] xys2,
                        int numIterMax=10000) except -1:
    cdef int s1, s2
    s1, s2 = zs1.shape[0], zs2.shape[0]
    cdef double [:,::1] dists = np.zeros((s1, s2), dtype=float)
    euclidean_2dim_cdists(dists, xys1, xys2)
    
    return emd_c(zs1, zs2, dists, numIterMax)

cdef class MVPTree:
    
    cdef np.ndarray data_arr
    cdef np.ndarray idxs_arr
    cdef np.ndarray nodes_arr
    cdef np.ndarray vantage_points_arr
    cdef np.ndarray node_bounds_arr
    
    cdef public int[::1] idxs
    cdef public NodeData_t[::1] nodes
    cdef public double[:,::1] vantage_points
    cdef public double[:,::1] node_bounds
    
    cdef int m, v, p
    cdef int n_levels, n_nodes
    
    cdef MetricType metric
    
    def __cinit__(self):
        
        self.data_arr = np.empty((1,1), dtype=float, order='C')
        self.idxs_arr = np.empty(1, dtype=int, order='C')
        self.nodes_arr = np.empty(1, dtype=NodeData, order='C')
        self.node_bounds_arr = np.empty((1,1), dtype=float, order='C')
        
        self.idxs = self.idx_array_arr
        self.nodes = self.node_data_arr
        self.node_bounds = self.node_bounds_arr
        
    def __init__(self, data, m=2, v=1, p=50, metric='euclidean'):
        
        if metric == 'emd':
            self.data_arr = np.asarray(data, dtype=np.object_, order='C')
            self.metric = m_emd
        elif metric == 'euclidean':
            self.data_arr = np.asarray(data, dtype=float, order='C')
            self.metric = m_euclidean
        else:
            raise ValueError('metric must be one of emd or euclidean')
        
        # store and validate parameters
        self.m, self.v, self.p = m, v, p
        if self.m < 2:
            raise ValueError('m must be greater than or equal to 2')
        if self.v < 1:
            raise ValueError('v must be greater than or equal to 1')
        if self.p < 1:
            raise ValueError('p must be greater than or equal to 1')