from __future__ import absolute_import, division, print_function

# Python standard library
from collections import deque
import itertools
import numbers
import time
import warnings

# Numpy/Scipy
import numpy as np

# Cython imports
from cython.operator cimport dereference, preincrement
from libc.math cimport sqrt, fabs
from libc.stdlib cimport calloc, malloc, realloc, free
from libc.string cimport memcpy
from libcpp.list cimport list as cpplist
cimport numpy as np
from numpy.math cimport INFINITY

cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

np.import_array()

__all__ = ['MVPTree']

################################################################################
# Types
################################################################################

eps = np.finfo(np.double).eps

ctypedef np.intp_t ind_t
ctypedef fused numeric_t:
    ind_t
    double

cdef struct Node_t:
    ind_t data_start_idx  # starting index of data
    ind_t data_end_idx    # ending index of data
    ind_t child_start_idx # starting index of child nodes
    ind_t child_end_idx   # ending index of child nodes
    ind_t node_bounds_idx # index of bode bounds in the node_bounds array
    ind_t nvps            # number of vantage points, zero means leaf

cdef Node_t nd_tmp
Node = np.asarray(<Node_t[:1]> (&nd_tmp)).dtype

ctypedef (double **, double **, ind_t *) dptrtup_t

cdef enum MetricType:
    m_emd, m_euclidean

################################################################################
# Eulidean metrics
################################################################################

# inlined euclidean metric
cdef inline double squared_euclidean_metric(const double * x, const double * y, int dim) nogil:
    cdef double tmp, d = 0
    cdef int j
    for j in range(dim):
        tmp = x[j] - y[j]
        d += tmp * tmp
    return d

cdef inline double euclidean_metric(const double * x, const double * y, int dim) nogil except -1:
    cdef double tmp, d = 0
    cdef int j
    for j in range(dim):
        tmp = x[j] - y[j]
        d += tmp * tmp
    return sqrt(d)

cdef inline void squared_euclidean_cdists(const double * xs, ind_t len_xs, const double * ys, ind_t len_ys, double * dists, int dim) nogil:
    cdef ind_t i, j
    cdef int_t k = 0
    cdef const double * x
    for i in range(len_xs):
        x = &x[i*dim]
        for j in range(len_ys):
            dists[k] = squared_euclidean_metric(x, &ys[j*dim], dim)
            k += 1

cdef inline void euclidean_cdists(const double * xs, ind_t len_xs, const double * ys, ind_t len_ys, double * dists, int dim) nogil:
    cdef ind_t i, j
    cdef int_t k = 0
    cdef const double * x
    for i in range(len_xs):
        x = &x[i*dim]
        for j in range(len_ys):
            dists[k] = euclidean_metric(x, &ys[j*dim], dim)
            k += 1

cdef inline double squared_euclidean_metric_2dim(const double* x, const double* y) nogil:
    cdef double tmp0, tmp1
    tmp0 = x[0] - y[0]
    tmp1 = x[1] - y[1]
    return tmp0*tmp0 + tmp1*tmp1

cdef inline double euclidean_metric_2dim(const double* x, const double* y) nogil except -1:
    cdef double tmp0, tmp1
    tmp0 = x[0] - y[0]
    tmp1 = x[1] - y[1]
    return sqrt(tmp0*tmp0 + tmp1*tmp1)

cdef inline double squared_euclidean_metric_3dim(const double* x, const double* y) nogil:
    cdef double tmp0, tmp1, tmp2
    tmp0 = x[0] - y[0]
    tmp1 = x[1] - y[1]
    tmp2 = x[2] - y[2]
    return tmp0*tmp0 + tmp1*tmp1 + tmp2*tmp2

cdef inline double euclidean_metric_3dim(const double* x, const double* y) nogil except -1:
    cdef double tmp0, tmp1, tmp2
    tmp0 = x[0] - y[0]
    tmp1 = x[1] - y[1]
    tmp2 = x[2] - y[2]
    return sqrt(tmp0*tmp0 + tmp1*tmp1 + tmp2*tmp2)

# specialize the 2dim case
cdef inline void squared_euclidean_cdists_2dim(const double * xs, ind_t len_xs, const double * ys, ind_t len_ys, double * dists) nogil:
    cdef ind_t i, j
    cdef ind_t k = 0
    cdef const double * x
    for i in range(len_xs):
        x = &xs[2*i]
        for j in range(len_ys):
            dists[k] = squared_euclidean_metric_2dim(x, &ys[2*j])
            k += 1

cdef inline void euclidean_cdists_2dim(const double * xs, ind_t len_xs, const double * ys, ind_t len_ys, double * dists) nogil:
    cdef ind_t i, j
    cdef ind_t k = 0
    cdef const double * x
    for i in range(len_xs):
        x = &xs[2*i]
        for j in range(len_ys):
            dists[k] = euclidean_metric_2dim(x, &ys[2*j])
            k += 1

cdef inline void squared_euclidean_cdists_3dim(const double * xs, ind_t len_xs, const double * ys, ind_t len_ys, double * dists) nogil:
    cdef ind_t i, j
    cdef ind_t k = 0
    cdef const double * x
    for i in range(len_xs):
        x = &xs[2*i]
        for j in range(len_ys):
            dists[k] = squared_euclidean_metric_3dim(x, &ys[2*j])
            k += 1

cdef inline void euclidean_cdists_3dim(const double * xs, ind_t len_xs, const double * ys, ind_t len_ys, double * dists) nogil:
    cdef ind_t i, j
    cdef ind_t k = 0
    cdef const double * x
    for i in range(len_xs):
        x = &xs[2*i]
        for j in range(len_ys):
            dists[k] = euclidean_metric_3dim(x, &ys[2*j])
            k += 1

###############################################################################
# helper functions taken from scikit-learn's binary_tree.pxi file
###############################################################################

cdef inline void swap(numeric_t* arr, const ind_t i1, const ind_t i2) nogil:
    """swap the values at index i1 and i2 of arr"""
    cdef numeric_t tmp = arr[i1]
    arr[i1] = arr[i2]
    arr[i2] = tmp

cdef inline void dual_swap(double* darr, ind_t* iarr, const ind_t i1, const ind_t i2) nogil:
    """swap the values at inex i1 and i2 of both darr and iarr"""
    cdef double dtmp = darr[i1]
    darr[i1] = darr[i2]
    darr[i2] = dtmp

    cdef ind_t itmp = iarr[i1]
    iarr[i1] = iarr[i2]
    iarr[i2] = itmp

cdef void simultaneous_sort(double* dist, ind_t* idx, const ind_t size) nogil:
    """
    Perform a recursive quicksort on the dist array, simultaneously
    performing the same swaps on the idx array.  The equivalent in
    numpy (though quite a bit slower) is

    def simultaneous_sort(dist, idx):
        i = np.argsort(dist)
        return dist[i], idx[i]
    """

    cdef ind_t pivot_idx, i, store_idx
    cdef double pivot_val

    # in the small-array case, do things efficiently
    if size <= 1:
        pass
    elif size == 2:
        if dist[0] > dist[1]:
            dual_swap(dist, idx, 0, 1)
    elif size == 3:
        if dist[0] > dist[1]:
            dual_swap(dist, idx, 0, 1)
        if dist[1] > dist[2]:
            dual_swap(dist, idx, 1, 2)
            if dist[0] > dist[1]:
                dual_swap(dist, idx, 0, 1)
    else:
        # Determine the pivot using the median-of-three rule.
        # The smallest of the three is moved to the beginning of the array,
        # the middle (the pivot value) is moved to the end, and the largest
        # is moved to the pivot index.
        pivot_idx = size // 2
        if dist[0] > dist[size - 1]:
            dual_swap(dist, idx, 0, size - 1)
        if dist[size - 1] > dist[pivot_idx]:
            dual_swap(dist, idx, size - 1, pivot_idx)
            if dist[0] > dist[size - 1]:
                dual_swap(dist, idx, 0, size - 1)
        pivot_val = dist[size - 1]

        # partition indices about pivot.  At the end of this operation,
        # pivot_idx will contain the pivot value, everything to the left
        # will be smaller, and everything to the right will be larger.
        store_idx = 0
        for i in range(size - 1):
            if dist[i] < pivot_val:
                dual_swap(dist, idx, i, store_idx)
                store_idx += 1
        dual_swap(dist, idx, store_idx, size - 1)
        pivot_idx = store_idx

        # recursively sort each side of the pivot
        if pivot_idx > 1:
            simultaneous_sort(dist, idx, pivot_idx)
        if pivot_idx + 2 < size:
            simultaneous_sort(dist+pivot_idx+1, idx+pivot_idx+1, size-pivot_idx-1)

###############################################################################

cdef class NeighborsHeap:
    """A max-heap structure to keep track of distances/indices of neighbors

    This implements an efficient pre-allocated set of fixed-size heaps
    for chasing neighbors, holding both an index and a distance.
    When any row of the heap is full, adding an additional point will push
    the furthest point off the heap.

    Parameters
    ----------
    n_pts : int
        the number of heaps to use
    n_nbrs : int
        the size of each heap.
    """

    cdef readonly ind_t n_pts, n_nbrs
    cdef np.ndarray distances_arr, indices_arr
    cdef double[:,::1] distances
    cdef ind_t[:,::1] indices
    cdef double* distances_ptr
    cdef ind_t* indices_ptr

    def __cinit__(self):
        self.distances_arr = np.zeros((1, 1), dtype=np.double, order='C')
        self.indices_arr = np.zeros((1, 1), dtype=np.intp, order='C')
        self.distances = self.distances_arr
        self.indices = self.indices_arr
        self.distances_ptr = self.indices_ptr = NULL

    def __init__(self, n_pts, n_nbrs):
        self.n_pts, self.n_nbrs = n_pts, n_nbrs
        self.distances_arr = np.full((n_pts, n_nbrs), np.inf, dtype=np.double, order='C')
        self.indices_arr = np.zeros((n_pts, n_nbrs), dtype=np.intp, order='C')
        self.distances = self.distances_arr
        self.indices = self.indices_arr
        self.distances_ptr = &self.distances[0,0]
        self.indices_ptr = &self.indices[0,0]

    def get_arrays(self, sort=True):
        """Get the arrays of distances and indices within the heap.

        If sort=True, then simultaneously sort the indices and distances,
        so the closer points are listed first.
        """
        if sort:
            self._sort()
        return self.distances_arr, self.indices_arr

    cdef inline double largest(self, ind_t row) nogil:
        """Return the largest distance in the given row"""
        return self.distances_ptr[row*self.n_nbrs]

    def push(self, ind_t row, double val, ind_t i_val):
        self._push(row, val, i_val)

    cdef void _push(self, ind_t row, const double val, ind_t i_val) nogil:
        """push (val, i_val) into the given row"""
        
        # check if val should be in heap
        cdef double* dist_ptr = self.distances_ptr + row*self.n_nbrs
        if val > dist_ptr[0]:
            return

        cdef ind_t i, ic1, ic2, i_swap
        cdef ind_t* ind_ptr = self.indices_ptr + row*self.n_nbrs

        # insert val at position zero
        dist_ptr[0] = val
        ind_ptr[0] = i_val

        # descend the heap, swapping values until the max heap criterion is met
        i = 0
        while True:
            ic1 = 2 * i + 1
            ic2 = ic1 + 1

            if ic1 >= self.n_nbrs:
                break
            elif ic2 >= self.n_nbrs:
                if dist_ptr[ic1] > val:
                    i_swap = ic1
                else:
                    break
            elif dist_ptr[ic1] >= dist_ptr[ic2]:
                if val < dist_ptr[ic1]:
                    i_swap = ic1
                else:
                    break
            else:
                if val < dist_ptr[ic2]:
                    i_swap = ic2
                else:
                    break

            dist_ptr[i] = dist_ptr[i_swap]
            ind_ptr[i] = ind_ptr[i_swap]
            i = i_swap

        dist_ptr[i] = val
        ind_ptr[i] = i_val

    cdef void _sort(self) nogil:
        """simultaneously sort the distances and indices"""
        #cdef DTYPE_t[:, ::1] distances = self.distances
        #cdef ITYPE_t[:, ::1] indices = self.indices
        cdef ind_t row
        for row in range(self.n_pts):
            simultaneous_sort(self.distances_ptr + row*self.n_nbrs, self.indices_ptr + row*self.n_nbrs, self.n_nbrs)

################################################################################

def reverse_container_of_tuples(tuples):
    return [tuple(reversed(t)) for t in tuples]

def sort_by_reversed(x):
    return reverse_container_of_tuples(sorted(reverse_container_of_tuples(x)))

def newObj(obj):
    return obj.__new__(obj)

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

################################################################################
# MVPTree class
################################################################################

cdef class MVPTree:
    
    cdef: 
        readonly np.ndarray data_arr, idx_arr, node_arr, node_bounds_arr, vp_dist_arr
        readonly np.ndarray mpows_arr

        object random
    
        ind_t[::1] idxs, mpows
        Node_t[::1] nodes
        double[:,::1] vp_dists
        double[:,:,::1] node_bounds

        double** data1
        double** data2 # arrays of pointers to primary and secondary data 
        ind_t* sizes
        double* _dists  # array that will hold ground distances for EMD
        double* _scores # array that will hold the kNN search scores

        readonly dict tree_template
        dict _regions, _ordered_regions
    
        readonly ind_t n, m, v, p, l, pmin, n_nodes, n_leaves, n_in_leaves, dim
        ind_t _r, _lm1, _dists_size
        double _one_plus_eps
    
        readonly str metric_str
        MetricType metric

        size_t n_calls

    def __cinit__(self):
        
        self.data_arr = np.empty((1,1), dtype=np.double, order='C')
        self.idx_arr = np.empty(1, dtype=np.intp, order='C')
        self.node_arr = np.empty(1, dtype=Node, order='C')
        self.node_bounds_arr = np.empty((1,1,1), dtype=np.double, order='C')
        self.vp_dist_arr = np.empty((1,1), dtype=np.double, order='C')
        self.mpows_arr = np.empty(1, dtype=np.intp, order='C')
        
        self.idxs = self.idx_arr
        self.nodes = self.node_arr
        self.node_bounds = self.node_bounds_arr
        self.vp_dists = self.vp_dist_arr
        self.mpows = self.mpows_arr

        self.data1 = self.data2 = self.sizes = self._dists = NULL
        self.n_calls = 0
        
    def __init__(self, data, m=2, v=1, p=50, pmin=None, metric='euclidean', random_state=None):

        # validate and store parameters
        if m < 2 or not isinstance(m, int):
            raise ValueError('m must be an integer greater than or equal to 2')
        if v < 1 or not isinstance(v, int):
            raise ValueError('v must be an integer greater than or equal to 1')
        if p < 1 or not isinstance(p, int):
            raise ValueError('p must be an integer greater than or equal to 1')
        if pmin is not None and (pmin < 1 or not isinstance(pmin, int)):
            raise ValueError('pmin must be None or an integer greater than or equal to 1')

        # store random number generator
        self.random = check_random_state(random_state)

        self.m, self.v, self.p, self.n = m, v, p, len(data)
        self.pmin = max(1, self.p//2) if pmin is None else pmin

        # determine number of levels in the tree
        self.l = 0
        while self.n > self.p * self.m**(self.l*self.v) + self.v*(self.m**(self.l*self.v) - 1)//(self.m**self.v - 1):
            self.l += 1

        # initialize
        self._initialize(data, metric)
        
        # build tree template
        self._build_tree_template()

        # allocate arrays
        self.idxs = self.idx_arr = np.arange(self.n, dtype=np.intp)
        self.nodes = self.node_arr = np.zeros(self.n_nodes, dtype=Node, order='C')

        # flattened m-ary tree
        #self.node_bounds_arr = np.zeros((self.n_nodes, self.v + 1), dtype=np.double, order='C')
        nb_shape = (max(1, self.n_nodes - self.n_leaves), (self.mpows[self.v] - 1)/(self.m - 1), self.m + 1)
        self.node_bounds = self.node_bounds_arr = np.zeros(nb_shape, dtype=np.double, order='C')
        self.vp_dists = self.vp_dist_arr = np.full((self.n, max(1, self.v*self.l)), -1, dtype=np.double, order='C')

        # initialize nodes from tree template
        self._init_nodes()

        # build actual tree
        self._build_tree()

    def __dealloc__(self):
        free(self.data1)
        free(self.data2)
        free(self.sizes)
        free(self._dists)
        free(self._scores)

    def __reduce__(self):
        return (newObj, (type(self),), self.__getstate__())

    def __getstate__(self):
        # need to handle metric choice
        return (self.data_arr, self.idx_arr, self.node_arr, self.node_bounds_arr, self.vp_dist_arr,
                self.tree_template, self.metric_str,
                int(self.m), int(self.v), int(self.p), int(self.pmin), 
                int(self.l), int(self.n_leaves), int(self.n_in_leaves))

    def __setstate__(self, state):

        # store arrays
        self.data_arr = state[0]
        self.idx_arr = state[1]
        self.node_arr = state[2]
        self.node_bounds_arr = state[3]
        self.vp_dist_arr = state[4]

        # get memoryviews from arrays
        self.idxs = self.idx_arr
        self.nodes = self.node_arr
        self.node_bounds = self.node_bounds_arr
        self.vp_dists = self.vp_dist_arr

        # store metric
        self.metric_str = state[6]

        # store integers
        self.m = state[7]
        self.v = state[8]
        self.p = state[9]
        self.pmin = state[10]
        self.l = state[11]
        self.n_leaves = state[12]
        self.n_in_leaves = state[13]

        # initialize
        self._initialize(self.data_arr, self.metric_str)

        # store tree
        self.tree_template = state[5]

        # get derived integers
        self.n = len(self.data_arr)
        self.n_nodes = len(self.node_arr)

    def get_n_calls(self):
        return self.n_calls

    cdef int _initialize(self, data, str metric) except -1:

        self.metric_str = metric

        if metric == 'emd':
            self.metric = m_emd
        elif metric == 'euclidean':
            self.metric = m_euclidean
        else:
            raise ValueError('metric must be one of emd or euclidean')

        self.mpows = self.mpows_arr = self.m**np.arange(self.v+1, dtype=np.intp)

        self._lm1 = max(0, self.l - 1)
        self._one_plus_eps = 1 + 100*eps # hundred just for safety
        self._dists_size = self._r = self.dim = self.n_leaves = self.n_in_leaves = self.n_nodes = 0

        self.tree_template, self._regions, self._ordered_regions = {}, {}, {}

        self.data_arr = self._process_data(data)
        self.data1, self.data2, self.sizes = self._get_data_pointers(self.data_arr)

        # m^v sized arrays for holding data for kNN scores during searching
        self._scores = <double*> malloc(self.mpows[self.v] * sizeof(double))

        return 0

    def _process_data(self, data):
        if self.metric == m_euclidean:
            data = np.asarray(data, dtype=np.double, order='C')
            if data.ndim != 2:
                raise ValueError('input data must be a two-dimensional array')

            # set dimension
            if self.dim == 0:
                self.dim = data.shape[1]
            elif self.dim != data.shape[1]:
                raise ValueError('dimensions of arrays must match')

        elif self.metric == m_emd:
            data = np.asarray(data)
            if data.ndim != 2:
                raise ValueError('input data must be a two-dimensional array')
            if data.shape[1] != 2:
                raise ValueError('input data must be an array with shape (num_samples, 2)')

        return np.asarray(data)

    cdef dptrtup_t _get_data_pointers(self, data_arr) except *:        

        cdef ind_t i, max_size
        cdef double[:,::1] data_view_2d
        cdef double[::1] data_view_1d
        cdef double** data1 = NULL
        cdef double** data2 = NULL
        cdef ind_t* sizes = NULL
        cdef ind_t n = data_arr.shape[0]

        if self.metric == m_euclidean:

            # allocate memory for data array
            data1 = <double**> malloc(n * sizeof(double*))
            if data1 == NULL:
                raise MemoryError()

            # set pointers
            data_view_2d = data_arr
            for i in range(n):
                data1[i] = &data_view_2d[i,0]

        elif self.metric == m_emd:

            # allocate arrays
            data1 = <double**> malloc(n * sizeof(double*))
            data2 = <double**> malloc(n * sizeof(double*))
            sizes = <ind_t*> malloc(n * sizeof(ind_t))

            # check for NULL
            if data1 == NULL or data2 == NULL or sizes == NULL:
                free(data1)
                free(data2)
                free(sizes)
                raise MemoryError()

            # ensure C order for subarrays and assign pointers
            max_size = 0
            for i in range(n):
                data_arr[i,0] = np.asarray(data_arr[i,0], dtype=np.double, order='C')
                data_arr[i,1] = np.asarray(data_arr[i,1], dtype=np.double, order='C')

                
                if data_arr[i,0].ndim != 1 or data_arr[i,0].shape[0] == 0:
                    m = 'weight array of sample {} has wrong number of dimensions or is length zero'.format(i)
                    raise ValueError(m)

                sizes[i] = data_arr[i,0].shape[0]
                max_size = max(sizes[i], max_size)

                if (data_arr[i,1].ndim != 2 or 
                    data_arr[i,1].shape[0] != self.sizes[i] or 
                    data_arr[i,1].shape[1] == 0):
                    m = 'ground array of sample {} has wrong shape'.format(i)
                    raise ValueError(m)

                # check for same ground space dimension
                if self.dim == 0:
                    self.dim = data_arr[i,1].shape[1]
                else:
                    if data_arr[i,1].shape[1] != self.dim:
                        raise ValueError('shape of ground array of sample {} mismatched'.format(i))

                # set pointers to data
                data_view_1d, data_view_2d = data_arr[i,0], data_arr[i,1]
                data1[i] = &data_view_1d[0]
                data2[i] = &data_view_2d[0,0]

            # allocate dists array that can hold the maximum internal pairwise dist
            self._dists_size = max(max_size**2, self._dists_size)
            self._dists = <double*> realloc(self._dists, self._dists_size * sizeof(double))
            if self._dists == NULL:
                raise MemoryError()

        return (data1, data2, sizes)

    cdef int _build_tree_template(self) except -1:

        if self.l == 0:
            self.tree_template.update({'nl': self.n, 'sig': (), 'i': 0, 'c': self.n})
            self.n_leaves = 1
            self.n_in_leaves += self.n

        else:

            # build full tree at level lm1
            self._r = self.n
            leaf_sigs = set()
            self._build_tree_template_recursive(self.tree_template, 0, (), leaf_sigs)

            sorted_leaves = [self._goto_leaf(self.tree_template, sig) for sig in sort_by_reversed(leaf_sigs)]
            n_leaves = len(sorted_leaves)
            nper_l = self._r//n_leaves
            
            thresholds = sorted([(self.p, 0)] + 
                                [(v + self.pmin * self.mpows[v], v) for v in range(1, self.v+1) 
                                                                    if v + self.pmin * self.mpows[v] != self.p] +
                                [(self.v + self.p * self.mpows[self.v], -1)], key=lambda x: x[0])
            thresholds = thresholds[thresholds.index((self.p, 0)):]

            vprev = None
            for thresh,v in thresholds:
                
                # construct leaves according to previous threshold
                if thresh > nper_l:
                    next_v = v
                    v = vprev
                    for leaf in sorted_leaves:
                        self._change_leaf_v(leaf, v)
                    break
                vprev = v
            
            # now go through leaves and try to add vantage points
            i = 0
            if 0 <= v < self.v:
                for i,leaf in enumerate(sorted_leaves):

                    # next_v might be -1 if p is bigger than everything involving a pmin
                    if not self._change_leaf_v(leaf, next_v if next_v != -1 else self.v):
                        break
            elif v == -1:
                # no need to check here since we have enough for v vps at pmin
                for leaf in sorted_leaves:
                    self._change_leaf_v(leaf, self.v)
                        
            # place remaining , start from where previous loop left off
            while self._r > 0:
                self._add_to_leaf(sorted_leaves[i%n_leaves])
                i += 1

        # update node capacities and indices
        nb_i = 0
        for node in self._bfs():
            if 'nl' in node:
                self.n_leaves += 1
                node['c'] = node['nl']
            else:
                node['c'] = node['nvp']
                node['nb_i'] = nb_i
                nb_i += 1

            # assign node an index
            node['i'] = self.n_nodes
            self.n_nodes += 1

        # assign start and end data indices to each node with a dfs
        assert self.n == self._assign_idxs(self.tree_template, 0), 'assigning indices went wrong'
        assert self.n == self.get_n(), 'total n does not match'

        return 0

    cdef void _build_tree_template_recursive(self, dict tree, ind_t level, tuple sig, set leaf_sigs):

        if level == self._lm1:
            tree.update({'sig': sig, 'c': 0})
            leaf_sigs.add(sig)

        elif level < self._lm1:
            tree.update({'nvp': self.v, 'sig': sig})
            self._r -= self.v
            
            for region in self._get_regions(self.v):
                tree[region] = {}
                self._build_tree_template_recursive(tree[region], level + 1, sig + region, leaf_sigs)

    cdef dict _goto_leaf(self, dict tree, tuple sig):
        return self._goto_leaf(tree[sig[:self.v]], sig[self.v:]) if len(sig) != 0 else tree

    cdef list _get_regions(self, v):
        if v not in self._regions:
            self._regions[v] = list(itertools.product(range(self.m), repeat=v))
        return self._regions[v]

    cdef list _get_ordered_regions(self, v):
        if v not in self._ordered_regions:
            self._ordered_regions[v] = sort_by_reversed(self._get_regions(v))
        return self._ordered_regions[v]

    cdef _change_leaf_v(self, dict leaf, ind_t new_v):
            
        n_required = new_v + self.pmin * self.mpows[new_v] - leaf.get('c', 0)
            
        #assert n_required > 0 
        if self._r < n_required:
            return False
        
        # clean old leaf
        cdef ind_t nl
        if 'nl' in leaf:
            nl = leaf['nl']
            self._r += nl
            self.n_in_leaves -= nl
            del leaf['nl']
        
        # clean old node
        elif 'nvp' in leaf:
            for region in self._get_regions(leaf['nvp']):
                nl = leaf[region]['nl']
                self._r += nl
                self.n_in_leaves -= nl
                del leaf[region]
            self._r += leaf['nvp']

        # make new leaf
        if new_v == 0:
            leaf['nl'] = leaf['c'] = self.p
            self.n_in_leaves += self.p
            self._r -= self.p

        # make new node
        else:
            leaf['nvp'] = new_v
            self._r -= new_v
            sig = leaf['sig']
            for region in self._get_regions(new_v):    
                leaf[region] = {'nl': self.pmin, 'sig': sig + region}

            pmin_m2v = self.pmin * self.mpows[new_v]
            leaf['c'] = new_v + pmin_m2v
            self._r -= pmin_m2v
            self.n_in_leaves += pmin_m2v
            
        return True

    cdef void _add_to_leaf(self, dict leaf):
        if 'nl' in leaf:
            subleaf = leaf
        else:
            if 'i' not in leaf:
                leaf['i'] = 0
            regions = self._get_ordered_regions(leaf['nvp'])
            subleaf = leaf[regions[leaf['i']%len(regions)]]
            leaf['i'] += 1
        
        subleaf['nl'] += 1
        self._r -= 1
        self.n_in_leaves += 1

    def _bfs(self, tree=None):
        if tree is None:
            tree = self.tree_template

        d = deque([tree])
        while len(d) > 0:
            node = d.popleft()
            yield node
            if 'nvp' in node:
                for sig in self._get_regions(node['nvp']):
                    d.append(node[sig])

    def get_n(self):
        n = 0
        for node in self._bfs():
            n += node['c']
        return n

    # with the prescription herein, vps will always be [idx_start,idx_start+vps)
    cdef _assign_idxs(self, tree, i):
        tree['i_s'] = i
        if 'nl' in tree:
            tree['i_e'] = i + tree['c']
            return tree['i_e']

        i += tree['nvp']
        for sig in self._get_regions(tree['nvp']):
            i = self._assign_idxs(tree[sig], i)
        tree['i_e'] = i
        return i

    cdef void _init_nodes(self):

        cdef ind_t node_i
        for node in self._bfs():
            node_i = node['i']
            self.nodes[node_i].data_start_idx = node['i_s']
            self.nodes[node_i].data_end_idx = node['i_e']

            # if this is not a leaf
            if 'nvp' in node:
                self.nodes[node_i].nvps = node['nvp']
                self.nodes[node_i].node_bounds_idx = node['nb_i']

                # assign child indices
                regions = self._get_regions(node['nvp'])
                self.nodes[node_i].child_start_idx = node['c_s'] = node[regions[0]]['i']
                self.nodes[node_i].child_end_idx = node['c_e'] = node[regions[len(regions)-1]]['i'] + 1 # for python indexing
            else:
                self.nodes[node_i].node_bounds_idx = -1

    cdef int _build_tree(self) nogil except -1:

        # local integers
        cdef ind_t node_i, i, prev_vps, j, vr, v, vp_i, vp_col, sub_stride, stride, start_i, end_i, reg, k

        # integers to hold node data
        cdef ind_t data_start_idx, data_end_idx
        
        # pointers to node bound array
        cdef double* nb

        # array to hold distance results
        cdef double[::1] dists
        with gil:
            dists_arr = np.empty(self.n, dtype=np.double, order='C')
            dists = dists_arr
        cdef double* dists_ptr = &dists[0]

        # iterate over nodes
        cdef Node_t node
        for node_i in range(self.n_nodes):
            node = self.nodes[node_i]
            data_start_idx, data_end_idx = node.data_start_idx, node.data_end_idx

            # if leaf, skip
            if node.nvps == 0:
                continue

            # determine number of previous vps seen by points in this node
            i = self.idxs[data_start_idx]
            for prev_vps in range(self.vp_dists.shape[1]):
                if self.vp_dists[i, prev_vps] == -1:
                    break

            # iterate over vantage points
            j, vr = 0, node.nvps
            nb = &self.node_bounds[node.node_bounds_idx,0,0]
            for v in range(node.nvps):

                # randomly choose first vantage point
                if v == 0:
                    with gil:
                        vp_i = self.random.randint(data_start_idx, data_end_idx, dtype=np.intp)

                # choose vantage point to be farthest point according to last sort
                else:
                    vp_i = data_end_idx - 1

                # move selected vantage point into place at beginning of indices allocated for this node
                for i in range(vp_i - data_start_idx - v):
                    k = vp_i - 1
                    swap(&self.idxs[0], vp_i, k)
                    vp_i = k

                # calculate distance between this vantage point and all other points
                self._calc_internal_dists_1_many(vp_i, vp_i+1, data_end_idx, dists_ptr)

                # store distances to vantage point
                vp_col = prev_vps + v
                self.vp_dists[self.idxs[vp_i], vp_col] = 0
                for i in range(vp_i+1, data_end_idx):
                    self.vp_dists[self.idxs[i], vp_col] = dists_ptr[i-vp_i-1]

                # increment v as we now have selected another vantage point
                vr -= 1
                sub_stride = self.mpows[vr]
                stride = self.mpows[vr + 1]

                i = node.child_start_idx
                while i < node.child_end_idx:
                    
                    # get start of indices for this region, note shift dependent on v
                    start_i = self.nodes[i].data_start_idx - vr

                    # enlarge end of final region to account for vps
                    end_i = (data_end_idx if (i == node.child_end_idx-stride) else 
                            (self.nodes[i+stride].data_start_idx - vr))

                    # simultaneous sort this region of the arrays
                    simultaneous_sort(&dists_ptr[start_i-vp_i-1], &self.idxs[start_i], end_i - start_i)

                    # determine node boundaries
                    reg = j*(self.m+1)
                    for k in range(self.m):
                        start_i = self.nodes[i + k*sub_stride].data_start_idx - vr
                        nb[reg + k] = self.vp_dists[self.idxs[start_i], vp_col]

                    # take care of final boundary
                    nb[reg + self.m] = self._one_plus_eps * self.vp_dists[self.idxs[end_i-1], vp_col]

                    j += 1
                    i += stride

        return 0

    cdef void _calc_internal_dists_1_many(self, ind_t i, ind_t idx_start, ind_t idx_end, double * results) nogil:
        cdef ind_t i_idx, j, s_i, s_j
        i_idx = self.idxs[i]
        self.n_calls += (idx_end - idx_start)
        if self.metric == m_emd:
            pass
            #s_i = self.sizes[self.idxs[i]]
            #if self.dim == 2:
            #    for j in range(idx_start, idx_end):
            #        s_j = self.sizes[j]
            #        euclidean_cdists_2dim(self.data2[i], s_i, self.data2[j], s_j, self._dists)
            #        results[j - idx_start] = emd_c(s_i, s_j, self.data1[i], self.data1[j], self._dists)
            #else:
            #    for j in range(idx_start, idx_end):
            #        s_j = self.sizes[j]
            #        euclidean_cdists(self.data2[i], s_i, self.data2[j], s_j, self._dists, self.dim)
            #        results[j - idx_start] = emd_c(s_i, s_j, self.data1[i], self.data1[j], self._dists)
        elif self.metric == m_euclidean:
            if self.dim == 2:
                for j in range(idx_start, idx_end):
                    results[j - idx_start] = euclidean_metric_2dim(self.data1[i_idx], self.data1[self.idxs[j]])
            elif self.dim == 3:
                for j in range(idx_start, idx_end):
                    results[j - idx_start] = euclidean_metric_3dim(self.data1[i_idx], self.data1[self.idxs[j]])
            else:
                for j in range(idx_start, idx_end):
                    results[j - idx_start] = euclidean_metric(self.data1[i_idx], self.data1[self.idxs[j]], self.dim)
    
    def query_radius(self, X, const double r, const int return_distances=False, const int count_only=False, const int sort_results=False):

        if count_only and return_distances:
            raise ValueError('count_only and return_distances cannot both be True')
        if sort_results and not return_distances:
            raise ValueError('return_distances must be True if sort_results is True')
        
        X = self._process_data(X)
        cdef dptrtup_t data_ptrs = self._get_data_pointers(X)

        cdef ind_t i, nX = X.shape[0]

        cdef ind_t** indices = NULL
        cdef double** distances = NULL
        if not count_only:
            indices = <ind_t**> calloc(nX, sizeof(ind_t*))
            if indices == NULL:
                raise MemoryError()
            if return_distances:
                distances = <double**> calloc(nX, sizeof(double*))
                if distances == NULL:
                    free(indices)
                    raise MemoryError()

        cdef double[::1] dists_i
        cdef double[:,::1] q_vp_dists
        cdef ind_t[::1] idxs_i
        cdef np.intp_t[::1] counts

        dists_i = dists_i_arr = np.zeros(self.n, dtype=np.double, order='C')
        cdef double* dists_i_ptr = &dists_i[0]

        # second axis is for -r, +r
        q_vp_dists = q_vp_dists_arr = np.zeros((max(1, self.v*self.l), 2), dtype=np.double, order='C')
        cdef double* q_vp_dists_ptr = &q_vp_dists[0,0]

        idxs_i = idxs_i_arr = np.zeros(self.n, dtype=np.intp, order='C')
        cdef ind_t* idxs_i_ptr = &idxs_i[0]

        counts = counts_arr = np.zeros(nX, dtype=np.intp, order='C')

        memory_error = False
        with nogil:
            for i in range(nX):
                counts[i] = self._query_radius_single(i, r, 0, 0, 0,
                                    data_ptrs, q_vp_dists_ptr, idxs_i_ptr, dists_i_ptr,
                                    count_only, return_distances)

                if count_only:
                    continue

                if sort_results:
                    simultaneous_sort(dists_i_ptr, idxs_i_ptr, counts[i])

                # equivalent to: indices[i] = np_idx_arr[:counts[i]].copy()
                indices[i] = <ind_t*> malloc(counts[i] * sizeof(ind_t))
                if indices[i] == NULL:
                    memory_error = True
                    break
                memcpy(indices[i], idxs_i_ptr, counts[i] * sizeof(ind_t))

                if return_distances:
                    # equivalent to: distances[i] = np_dist_arr[:counts[i]].copy()
                    distances[i] = <double*> malloc(counts[i] * sizeof(double))
                    if distances[i] == NULL:
                        memory_error = True
                        break
                    memcpy(distances[i], dists_i_ptr, counts[i] * sizeof(double))

        try:
            if memory_error:
                raise MemoryError()

            if count_only:
                return counts_arr

            indices_arr = np.zeros(nX, dtype='object')
            if return_distances:
                distances_arr = np.zeros(nX, dtype='object')
                for i in range(nX):
                    # make a new numpy array that wraps the existing data
                    indices_arr[i] = np.PyArray_SimpleNewFromData(1, &counts[i], np.NPY_INTP, indices[i])
                    # make sure the data will be freed when the numpy array is garbage collected
                    PyArray_ENABLEFLAGS(indices_arr[i], np.NPY_OWNDATA)
                    # make sure the data is not freed twice
                    indices[i] = NULL

                    # make a new numpy array that wraps the existing data
                    distances_arr[i] = np.PyArray_SimpleNewFromData(1, &counts[i], np.NPY_DOUBLE, distances[i])
                    # make sure the data will be freed when the numpy array is garbage collected
                    PyArray_ENABLEFLAGS(distances_arr[i], np.NPY_OWNDATA)
                    # make sure the data is not freed twice
                    distances[i] = NULL

                return indices_arr, distances_arr

            else:
                for i in range(nX):
                    # make a new numpy array that wraps the existing data
                    indices_arr[i] = np.PyArray_SimpleNewFromData(1, &counts[i], np.NPY_INTP, indices[i])
                    # make sure the data will be freed when the numpy array is garbage collected
                    PyArray_ENABLEFLAGS(indices_arr[i], np.NPY_OWNDATA)
                    # make sure the data is not freed twice
                    indices[i] = NULL

                # deflatten results
                return indices_arr

        except:
            # free any buffer that is not owned by a numpy array
            for i in range(nX):
                free(indices[i])
                if return_distances:
                    free(distances[i])
            raise

        finally:
            free(indices)
            free(distances)

    cdef np.intp_t _query_radius_single(self, const ind_t q_i, const double r, 
                                              const ind_t node_i, const ind_t prev_vps, np.intp_t count, 
                                              const dptrtup_t data_ptrs, double* q_vp_dists, ind_t* indices, double* distances,
                                              const int count_only, const int return_distances) nogil:

        cdef ind_t i, v, j, start_i, end_i, status
        cdef double d
        cdef double[::1] dists
        cdef Node_t node = self.nodes[node_i]

        # this node is a leaf
        if node.nvps == 0:

            # move if conditions outside of loops

            # iterate over points held by leaf
            for i in range(node.data_start_idx, node.data_end_idx):
                dists = self.vp_dists[self.idxs[i]]

                # check that the points are potentially in scope according to vp_dists
                # consider changing this iteration order 
                status = 1 # 0: out, 1: check 2: in
                for v in range(prev_vps):
                    j = 2*v

                    # check for automatic inclusion (large r)
                    if dists[v] <= -q_vp_dists[j]:
                        if not return_distances:
                            status = 2
                        break

                    # rule out point for being too far away (small r)
                    if dists[v] < q_vp_dists[j] or dists[v] > q_vp_dists[j+1]:
                        status = 0
                        break

                if status == 0:
                    continue

                # check distance
                if status == 1:
                    d = self._calc_single_dist_external(data_ptrs, q_i, i)
                    if d > r:
                        continue

                if count_only:
                    pass
                else:
                    indices[count] = self.idxs[i]
                    if return_distances:
                        distances[count] = d

                count += 1

            return count

        # else, this is an internal node, calc distances to all vps
        cdef cpplist[ind_t] nb_idx_list, sub_nb_idx_list
        nb_idx_list.push_back(0)
        cdef double* nb_ptr = &self.node_bounds[node.node_bounds_idx, 0, 0]
        cdef double* nb_j_ptr
        cdef ind_t k, l, jm, reg, subrow_start, sub_stride
        for v in range(node.nvps):
            sub_nb_idx_list.clear()

            # consider adding vantage point
            d = self._calc_single_dist_external(data_ptrs, q_i, node.data_start_idx + v)
            i = 2*(prev_vps + v)
            q_vp_dists[i] = d - r
            q_vp_dists[i + 1] = d + r
            if d <= r:
                if count_only:
                    pass
                else:
                    indices[count] = self.idxs[node.data_start_idx + v]
                    if return_distances:
                        distances[count] = d
                count += 1

            # iterate in similar fashion as building the tree
            sub_stride = self.mpows[node.nvps - v - 1]
            subrow_start = (self.mpows[v + 1] - 1)/(self.m - 1)
            while not nb_idx_list.empty():
                j = nb_idx_list.front() # get nb index at front
                nb_idx_list.pop_front() # remove nb index at front

                jm = j*self.m
                nb_j_ptr = nb_ptr + jm + j

                # iterate over subregions
                for k in range(self.m):

                    # check automatic region inclusion condition
                    if nb_j_ptr[k+1] <= -q_vp_dists[i]:
                        reg = node.child_start_idx + (jm + 1 + k - subrow_start)*sub_stride
                        start_i = self.nodes[reg].data_start_idx
                        end_i = self.nodes[reg + sub_stride - 1].data_end_idx

                        if count_only:
                            count += end_i - start_i
                        else:
                            for l in range(start_i, end_i):
                                indices[count] = self.idxs[l]
                                if return_distances:
                                    distances[count] = self._calc_single_dist_external(data_ptrs, q_i, l)
                                count += 1

                    # see if we can rule out region
                    elif q_vp_dists[i+1] < nb_j_ptr[k] or q_vp_dists[i] > nb_j_ptr[k+1]:
                        continue

                    # consider region further
                    else:
                        sub_nb_idx_list.push_back(jm + 1 + k)

            # add sub_nb_idx_list to nb_idx_list (which is currently empty)
            nb_idx_list.splice(nb_idx_list.begin(), sub_nb_idx_list)

        # recursively check child nodes
        j = (self.mpows[node.nvps] - 1)/(self.m - 1)
        while not nb_idx_list.empty():
            count = self._query_radius_single(q_i, r, node.child_start_idx + nb_idx_list.front() - j, 
                                              prev_vps + node.nvps, count, 
                                              data_ptrs, q_vp_dists, indices, distances,
                                              count_only, return_distances)
            nb_idx_list.pop_front()

        return count

    cdef inline double _calc_single_dist_external(self, const dptrtup_t data_ptrs, ind_t q_i, ind_t i) nogil:
        cdef ind_t s_i, s_j, i_idx
        i_idx = self.idxs[i]
        self.n_calls += 1
        if self.metric == m_emd:
            pass
            #s_i, s_j = data_ptrs[2][i], self.sizes[j]
            #if self.dim == 2:
            #    euclidean_cdists_2dim(data_ptrs[1][i], s_i, self.data2[j], s_j, self._dists)
            #else:
            #    euclidean_cdists(data_ptrs[1][i], s_i, self.data2[j], s_j, self._dists, self.dim)
            #return emd_c(s_i, s_j, data_ptrs[0][i], self.data1[j], self._dists)
        elif self.metric == m_euclidean:
            if self.dim == 2:
                return euclidean_metric_2dim(data_ptrs[0][q_i], self.data1[i_idx])
            elif self.dim == 3:
                return euclidean_metric_3dim(data_ptrs[0][q_i], self.data1[i_idx])
            else:
                return euclidean_metric(data_ptrs[0][q_i], self.data1[i_idx], self.dim)

    def query(self, X, ind_t k=1, return_distances=True, sort_results=True):

        if k < 1:
            raise ValueError('k must be a positive integer')
        if k > self.n:
            raise ValueError('k must be less than or equal to the number of training points')

        X = self._process_data(X)
        cdef dptrtup_t data_ptrs = self._get_data_pointers(X)
        cdef NeighborsHeap nb_heap = NeighborsHeap(X.shape[0], k)

        q_vp_dists_arr = np.zeros(max(1, self.v*self.l), dtype=np.double, order='C')
        cdef double[::1] q_vp_dists = q_vp_dists_arr
        cdef double* q_vp_dists_ptr = &q_vp_dists[0]

        cdef ind_t q_i
        with nogil:
            for q_i in range(nb_heap.n_pts):
                self._query_single_depthfirst(0, 0, q_i, data_ptrs, q_vp_dists_ptr, nb_heap)

        distances, indices = nb_heap.get_arrays(sort=sort_results)

        if return_distances:
            return distances, indices
        else:
            return indices

    cdef ind_t _query_single_depthfirst(self, const ind_t node_i, const ind_t prev_vps, const ind_t q_i, 
                                            const dptrtup_t data_ptrs, double* q_vp_dists, 
                                            NeighborsHeap nb_heap) nogil except -1:

        cdef Node_t node = self.nodes[node_i]

        cdef ind_t i, v
        cdef double largest, largest2, qmr, d
        cdef double[::1] dists
        
        # this node is a leaf
        if node.nvps == 0:

            # iterate over points held by leaf
            for i in range(node.data_start_idx, node.data_end_idx):
                dists = self.vp_dists[self.idxs[i]]

                # check that the points are potentially in scope according to vp_dists
                # consider changing this iteration order 
                status = 1 # 0: out, 1: check 2: in
                largest = nb_heap.largest(q_i)
                if largest < INFINITY:
                    largest2 = 2*largest
                    for v in range(prev_vps):
                        qmr = q_vp_dists[v] - largest

                        # check for automatic inclusion (large r)
                        #qmr = q_vp_dists[v] - largest
                        #if dists[v] <= -qmr:
                        #    break

                        # rule out point for being too far away (small r)
                        if dists[v] < qmr or dists[v] > qmr + largest2:
                            status = 0
                            break

                if status == 0:
                    continue

                # push value onto heap
                d = self._calc_single_dist_external(data_ptrs, q_i, i)
                #with gil:
                #    print('pushing leaf', d, self.idxs[i])
                nb_heap._push(q_i, d, self.idxs[i])

            return 0

        # else, this is an internal node, calc distances to all vps
        cdef cpplist[ind_t] nb_idx_list, sub_nb_idx_list
        nb_idx_list.push_back(0)
        cdef double* nb_ptr = &self.node_bounds[node.node_bounds_idx, 0, 0]
        cdef double* nb_j_ptr
        cdef double qpr, score
        cdef ind_t j, k, jm, reg, subrow_start, sub_stride

        # reset scores to infinity
        for i in range(self.mpows[node.nvps]):
            self._scores[i] = INFINITY

        for v in range(node.nvps):
            sub_nb_idx_list.clear()

            # consider vantage point
            i = prev_vps + v
            q_vp_dists[i] = d = self._calc_single_dist_external(data_ptrs, q_i, node.data_start_idx + v)
            #with gil:
            #    print('pushing vp', d, self.idxs[node.data_start_idx + v])
            nb_heap._push(q_i, d, self.idxs[node.data_start_idx + v])

            # iterate in similar fashion as building the tree
            sub_stride = self.mpows[node.nvps - v - 1]
            subrow_start = (self.mpows[v + 1] - 1)/(self.m - 1)
            while not nb_idx_list.empty():
                j = nb_idx_list.front() # get nb index at front
                nb_idx_list.pop_front() # remove nb index at front

                jm = j*self.m
                nb_j_ptr = nb_ptr + jm + j

                largest = nb_heap.largest(q_i)
                qmr = d - largest
                qpr = d + largest

                # iterate over subregions
                for k in range(self.m):

                    # see if we can rule out a region
                    if qpr < nb_j_ptr[k] or qmr > nb_j_ptr[k+1]:
                        continue

                    # assign score as the minimum distance between query point and the region
                    reg = (jm + 1 + k - subrow_start)*sub_stride
                    score = fabs((nb_j_ptr[k] + nb_j_ptr[k+1])/2 - d)
                    if score < self._scores[reg]:
                        for i in range(reg, reg + sub_stride):
                            self._scores[i] = score

                    # consider region further
                    sub_nb_idx_list.push_back(jm + 1 + k)

            # add sub_nb_idx_list to nb_idx_list (which is currently empty)
            nb_idx_list.splice(nb_idx_list.begin(), sub_nb_idx_list)

        # copy list values into arrays to be sorted
        cdef cpplist[ind_t].iterator it
        cdef double* search_scores
        cdef ind_t* search_nodes
        if not nb_idx_list.empty():
            it = nb_idx_list.begin()
            search_scores = <double*> malloc(nb_idx_list.size() * sizeof(double))
            search_nodes = <ind_t*> malloc(nb_idx_list.size() * sizeof(ind_t))
            j = (self.mpows[node.nvps] - 1)/(self.m - 1)
            i = 0
            while it != nb_idx_list.end():
                search_nodes[i] = dereference(it) - j
                search_scores[i] = self._scores[search_nodes[i]]
                #with gil:
                #    print('i, node, score', i, search_nodes[i], search_scores[i])
                preincrement(it)
                i += 1

            #with gil:
            #    print('i', i)

            # sort nodes to search according to the score
            simultaneous_sort(search_scores, search_nodes, i)

            # recursively check child nodes
            for k in range(i):
                #with gil:
                #    print('searching node', search_nodes[k])
                self._query_single_depthfirst(node.child_start_idx + search_nodes[k], prev_vps + node.nvps, 
                                              q_i, data_ptrs, q_vp_dists, nb_heap)

        return 0
