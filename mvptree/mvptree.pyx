from __future__ import absolute_import

import numpy as np
from collections import deque
import itertools
from mvptree.utils.typedefs import D, I

cimport numpy as np
from libc.math cimport ceil, floor, log, sqrt
from mvptree.lp.emd_wrap cimport emd_c
from mvptree.utils.typedefs cimport D_t, I_t, DI_t

np.import_array()

__all__ = ['MVPTree']

cdef enum MetricType:
    m_emd, m_euclidean

cdef struct Node_t:
    I_t data_start_idx  # starting index of data
    I_t data_end_idx    # ending index of data
    I_t child_start_idx # starting index of child nodes
    I_t child_end_idx   # ending index of child nodes
    I_t bounds_idx      # index of bounds array corresponding to this node
    I_t nvps            # number of vantage points, zero means leaf

cdef Node_t nd_tmp
Node = np.asarray(<Node_t[:1]> (&nd_tmp)).dtype

# inlined euclidean metric
cdef inline D_t euclidean_metric(D_t* x, D_t* y, I_t size) nogil except -1:
    cdef D_t tmp, d = 0
    cdef I_t j
    for j in range(size):
        tmp = x[j] - y[j]
        d += tmp * tmp
    return sqrt(d)

cdef inline D_t euclidean_2dim(D_t* x, D_t* y) nogil except -1:
    cdef D_t tmp0, tmp1
    tmp0 = x[0] - y[0]
    tmp1 = x[1] - y[1]
    return sqrt(tmp0*tmp0 + tmp1*tmp1)

cdef inline void euclidean_2dim_cdists(D_t [:,::1] dists, 
                                       D_t [:,::1] xs,
                                       D_t [:,::1] ys) nogil:
    cdef I_t i, j
    cdef D_t* x
    for i in range(xs.shape[0]):
        x = &xs[i,0]
        for j in range(ys.shape[0]):
            dists[i,j] = euclidean_2dim(x, &ys[j,0])

cpdef inline D_t emd_metric(D_t[::1] zs1,
                        D_t[::1] zs2,
                        D_t[:,::1] xys1,
                        D_t[:,::1] xys2,
                        I_t numIterMax=10000) except -1:
    cdef I_t s1, s2
    s1, s2 = zs1.shape[0], zs2.shape[0]
    cdef D_t [:,::1] dists = np.zeros((s1, s2), dtype=D)

    with nogil:
        euclidean_2dim_cdists(dists, xys1, xys2)
        return emd_c(zs1, zs2, dists, numIterMax)

cdef inline I_t safe_ceil(D_t x, D_t tol=10**-12) nogil:
    return <I_t> ceil(x - tol)

cdef inline I_t safe_floor(D_t x, D_t tol=10**-12) nogil:
    return <I_t> floor(x + tol)

def reverse_container_of_tuples(tuples):
    return [tuple(reversed(t)) for t in tuples]

def sort_by_reversed(x):
    return reverse_container_of_tuples(sorted(reverse_container_of_tuples(x)))

cdef class MVPTree:
    
    cdef: 
        np.ndarray data_arr
        np.ndarray idx_arr
        np.ndarray node_arr
        np.ndarray node_bound_arr
    
        I_t[::1] idxs
        Node_t[::1] nodes
        D_t[:,::1] node_bounds
    
        readonly I_t n, m, v, p, l, pmin
        readonly I_t _r, _lm1, _n_nodes, _n_leaves, _n_in_leaves
    
        MetricType metric

        readonly dict _regions, _ordered_regions, _tree
        readonly set  _leaf_sigs
        readonly list _sorted_leaves
    
    def __cinit__(self):
        
        self.data_arr = np.empty((1,1), dtype=D, order='C')
        self.idx_arr = np.empty(1, dtype=I, order='C')
        self.node_arr = np.empty(1, dtype=Node, order='C')
        self.node_bound_arr = np.empty((1,1), dtype=D, order='C')
        
        self.idxs = self.idx_arr
        self.nodes = self.node_arr
        self.node_bounds = self.node_bound_arr

        self.n, self.m, self.v, self.p, self.l, self.pmin = 0, 2, 1, 1, 0, 1
        self._r, self._lm1, self._n_nodes, self._n_leaves, self._n_in_leaves = 0, 0, 0, 0, 0

        self._regions, self._ordered_regions, self._tree = {}, {}, {}
        self._leaf_sigs = set()
        self._sorted_leaves = []

        self.metric = m_euclidean
        
    def __init__(self, data, m=2, v=1, p=50, pmin=None, metric='euclidean'):
        
        # handle choosing metric (should probably be more advanced later)
        if metric == 'emd':
            self.data_arr = np.asarray(data, dtype=np.object_, order='C')
            self.metric = m_emd
        elif metric == 'euclidean':
            self.data_arr = np.asarray(data, dtype=D, order='C')
            self.metric = m_euclidean
        else:
            raise ValueError('metric must be one of emd or euclidean')
        
        # validate and store parameters
        if m < 2 or not isinstance(m, int):
            raise ValueError('m must be an integer greater than or equal to 2')
        if v < 1 or not isinstance(v, int):
            raise ValueError('v must be an integer greater than or equal to 1')
        if p < 1 or not isinstance(p, int):
            raise ValueError('p must be an integer greater than or equal to 1')
        if pmin is not None and (pmin < 1 or not isinstance(pmin, int)):
            raise ValueError('pmin must be None or an integer greater than or equal to 1')
        self.m, self.v, self.p = m, v, p
        self.pmin = max(1, self.p//2) if pmin is None else pmin

        # determine number of levels in the tree
        self.n = self._r = self.data_arr.shape[0]
        m2vm1 = self.m**self.v - 1
        self.l = max(0, safe_ceil(log((self.v + self.n*m2vm1)/(self.v + self.p*m2vm1))/(self.v*log(self.m))))
        self._lm1 = max(0, self.l - 1)

        # handle case of a single leaf
        if self._r < self.p:
            self._tree.update({'nl': self._r, 'sig': (), 'i': 0, 'c': self._r})
            self._n_nodes = self._n_leaves = 1
            self._n_in_leaves += self._r
            self._r = 0

        # handle general case
        else:
            self._build_tree_template()

        # allocate arrays
        self.idx_arr = np.arange(self.n, dtype=I)
        self.idxs = self.idx_arr

        self.node_arr = np.zeros(self._n_nodes, dtype=Node)
        self.nodes = self.node_arr

        self.node_bound_arr = np.zeros((self._n_nodes-1, self.v+1), dtype=D)
        self.node_bounds = self.node_bound_arr

        # build actual tree
        self._build_tree()

    def _build_tree_template(self):

        # build full tree at level lm1
        self._build_tree_template_recursive(self._tree, 0, ())

        self._sorted_leaves = [self._goto_leaf(self._tree, sig) for sig in sort_by_reversed(self._leaf_sigs)]
        self._leaf_sigs.clear()
        n_leaves = len(self._sorted_leaves)
        nper_l = self._r//n_leaves
        
        thresholds = sorted([(self.p, 0)] + 
                            [(v + self.pmin * self.m**v, v) for v in range(1, self.v+1) 
                                                                if v + self.pmin * self.m**v != self.p] +
                            [(self.v + self.p * self.m**self.v, -1)], key=lambda x: x[0])
        thresholds = thresholds[thresholds.index((self.p, 0)):]

        vprev = None
        for thresh,v in thresholds:
            
            # construct leaves according to previous threshold
            if thresh > nper_l:
                next_v = v
                v = vprev
                for leaf in self._sorted_leaves:
                    self._change_leaf_v(leaf, v)
                break
            vprev = v
        
        # now go through leaves and try to add vantage points
        i = 0
        if 0 <= v < self.v:
            for i,leaf in enumerate(self._sorted_leaves):

                # next_v might be -1 if p is bigger than everything involving a pmin
                if not self._change_leaf_v(leaf, next_v if next_v != -1 else self.v):
                    break
        elif v == -1:
            # no need to check here since we have enough for v vps at pmin
            for leaf in self._sorted_leaves:
                self._change_leaf_v(leaf, self.v)
                    
        # place remaining , start from where previous loop left off
        while self._r > 0:
            self._add_to_leaf(self._sorted_leaves[i%n_leaves])
            i += 1

        # update node capacities and indices
        for node in self._bfs():
            if 'nl' in node:
                self._n_leaves += 1
                node['c'] = node['nl']
            else:
                node['c'] = node['nvp']

            # remove counter from construction to not get confused
            if 'i' in node:
                del node['i']

            self._n_nodes += 1

        # assign start and end indices to each node with a dfs
        assert self.n == self._assign_idxs(self._tree, 0), 'assigning indices went wrong'

    cdef void _build_tree_template_recursive(self, dict tree, I_t level, tuple sig):

        if level == self._lm1:
            tree.update({'sig': sig, 'c': 0})
            self._leaf_sigs.add(sig)

        elif level < self._lm1:
            tree.update({'nvp': self.v, 'sig': sig})
            self._r -= self.v
            
            for region in self._get_regions(self.v):
                tree[region] = {}
                self._build_tree_template_recursive(tree[region], level + 1, sig + region)

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

    cdef _change_leaf_v(self, dict leaf, new_v):
            
        n_required = new_v + self.pmin * self.m**new_v - leaf.get('c', 0)
            
        #assert n_required > 0 
        if self._r < n_required:
            return False
        
        # clean old leaf
        cdef I_t nl
        if 'nl' in leaf:
            nl = leaf['nl']
            self._r += nl
            self._n_in_leaves -= nl
            del leaf['nl']
        
        # clean old node
        elif 'nvp' in leaf:
            for region in self._get_regions(leaf['nvp']):
                nl = leaf[region]['nl']
                self._r += nl
                self._n_in_leaves -= nl
                del leaf[region]
            self._r += leaf['nvp']

        # make new leaf
        if new_v == 0:
            leaf['nl'] = leaf['c'] = self.p
            self._n_in_leaves += self.p
            self._r -= self.p

        # make new node
        else:
            leaf['nvp'] = new_v
            self._r -= new_v
            sig = leaf['sig']
            for region in self._get_regions(new_v):    
                leaf[region] = {'nl': self.pmin, 'sig': sig + region}

            pmin_m2v = self.pmin * self.m**new_v
            leaf['c'] = new_v + pmin_m2v
            self._r -= pmin_m2v
            self._n_in_leaves += pmin_m2v
            
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
        self._n_in_leaves += 1

    def _bfs(self, tree=None):
        if tree is None:
            tree = self._tree

        d = deque([tree])
        while len(d) > 0:
            node = d.popleft()
            yield node
            if 'nvp' in node:
                for sig in self._get_regions(node['nvp']):
                    d.append(node[sig])

    # with the prescription herein, vps will always be [idx_start,idx_start+vps)
    def _assign_idxs(self, tree, i):
        tree['i_s'] = i
        if 'nl' in tree:
            tree['i_e'] = i + tree['c']
            return tree['i_e']

        i += tree['nvp']
        for sig in self._get_regions(tree['nvp']):
            i = self._assign_idxs(tree[sig], i)
        tree['i_e'] = i
        return i

    cdef void _build_tree(self):
        pass

    def get_n(self):
        n = 0
        for node in self._bfs():
            n += node['c']
        return n
