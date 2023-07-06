import numpy as np
cimport numpy as np
cimport cython
"""
Original code from https://github.com/mattapow/dodonaphy/blob/b0e7e27ce2b46c74c15cc4f964c72be535784b8e/dodonaphy/cython/Cpeeler.pyx
"""

cdef np.double_t infty = np.finfo(np.double).max
cdef np.double_t eps = np.finfo(np.double).eps

cdef class Node:
    cdef int taxon
    cdef dict _nj_distances
    cdef double _nj_xsub

    def __init__(self, int taxon):
        self.taxon = taxon
        self._nj_distances = {}
        self._nj_xsub = 0.0

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef nj(double[:, ::1] pdm):
    """ Calculate neighbour joining tree.
    Credit to Dendropy for python implentation.

    Args:
        pdm (ndarray): Pairwise distance matrix
    """

    cdef int n_pool = len(pdm)
    cdef int n_taxa = len(pdm)
    cdef int n_ints = n_taxa - 1
    cdef int node_count = 2 * n_taxa - 2

    cdef np.ndarray[long, ndim=2] peel = np.zeros((n_ints, 3), dtype=long)
    cdef np.ndarray[np.double_t, ndim=1] blens = np.zeros(node_count, dtype=np.double)

    # initialise node pool
    node_pool = [Node(taxon=taxon) for taxon in range(n_pool)]

    cdef np.double_t dist
    cdef np.double_t v1
    cdef np.double_t v3
    cdef np.double_t v4
    cdef np.double_t qvalue
    cdef np.double_t min_q
    cdef int int_i
    cdef int parent
    cdef np.double_t delta_f
    cdef np.double_t delta_g
    cdef np.double_t two = 2.

    # More hints to use Node
    cdef Node nd
    cdef Node nd1
    cdef Node nd2
    cdef Node node_to_join1
    cdef Node node_to_join2

    # cache calculations
    for nd1 in node_pool:
        nd1._nj_xsub = <np.double_t>0.0
        for nd2 in node_pool:
            if nd1 is nd2:
                continue
            dist = pdm[nd1.taxon, nd2.taxon]
            nd1._nj_distances[nd2] = dist
            nd1._nj_xsub += dist

    while n_pool > 1:
        # calculate argmin of Q-matrix
        min_q = infty
        n_pool = len(node_pool)
        for idx1, nd1 in enumerate(node_pool[:-1]):
            for _, nd2 in enumerate(node_pool[idx1 + 1 :]):
                v1 = (<np.double_t>(n_pool) - two) * <np.double_t>(nd1._nj_distances[nd2])
                qvalue = v1 - <np.double_t>nd1._nj_xsub - <np.double_t>nd2._nj_xsub
                if qvalue < min_q:
                    min_q = qvalue
                    #nodes_to_join = (nd1, nd2)
                    node_to_join1 = nd1
                    node_to_join2 = nd2

        # create the new node
        int_i = n_taxa - n_pool
        parent = int_i + n_taxa
        new_node = Node(parent)
        peel[int_i, 2] = parent

        # attach it to the tree
        #peel[int_i, 0] = nodes_to_join[0].taxon
        #peel[int_i, 1] = nodes_to_join[1].taxon
        #node_pool.remove(nodes_to_join[0])
        #node_pool.remove(nodes_to_join[1])
        peel[int_i, 0] = node_to_join1.taxon
        peel[int_i, 1] = node_to_join2.taxon
        node_pool.remove(node_to_join1)
        node_pool.remove(node_to_join2)

        # calculate the distances for the new node
        for nd in node_pool:
            # actual node-to-node distances
            v1 = 0.0
            #for node_to_join in nodes_to_join:
            #    v1 += <np.double_t>nd._nj_distances[node_to_join]
            #v3 = nodes_to_join[0]._nj_distances[nodes_to_join[1]]
            v1 += <np.double_t>nd._nj_distances[node_to_join1]
            v1 += <np.double_t>nd._nj_distances[node_to_join2]
            v3 = node_to_join1._nj_distances[node_to_join2]
            dist = 0.5 * (v1 - v3)
            new_node._nj_distances[nd] = dist
            nd._nj_distances[new_node] = dist

            # Adjust/recalculate the values needed for the Q-matrix calculation
            new_node._nj_xsub += dist
            nd._nj_xsub += dist
            #for node_to_join in nodes_to_join:
            #    nd._nj_xsub -= <np.double_t>node_to_join._nj_distances[nd]
            nd._nj_xsub -= <np.double_t>node_to_join1._nj_distances[nd]
            nd._nj_xsub -= <np.double_t>node_to_join2._nj_distances[nd]

        # calculate the branch lengths
        if n_pool > 2:
            #v1 = 0.5 * nodes_to_join[0]._nj_distances[nodes_to_join[1]]
            v1 = 0.5 * node_to_join1._nj_distances[node_to_join2]
            v4 = (
                1.0
                / (two * (<np.double_t>n_pool - two))
                #* (nodes_to_join[0]._nj_xsub - nodes_to_join[1]._nj_xsub)
                * (node_to_join1._nj_xsub - node_to_join2._nj_xsub)
            )
            delta_f = v1 + v4
            #delta_g = <np.double_t>nodes_to_join[0]._nj_distances[nodes_to_join[1]] - delta_f
            #blens[nodes_to_join[0].taxon] = delta_f
            #blens[nodes_to_join[1].taxon] = delta_g
            delta_g = <np.double_t>node_to_join1._nj_distances[node_to_join2] - delta_f
            blens[node_to_join1.taxon] = delta_f
            blens[node_to_join2.taxon] = delta_g
        else:
            #dist = nodes_to_join[0]._nj_distances[nodes_to_join[1]]
            #blens[nodes_to_join[0].taxon] = dist / two
            #blens[nodes_to_join[1].taxon] = dist / two
            dist = node_to_join1._nj_distances[node_to_join2]
            blens[node_to_join1.taxon] = dist / two
            blens[node_to_join2.taxon] = dist / two

        # clean up
        #for node_to_join in nodes_to_join:
        #    node_to_join._nj_distances = {}
        #    node_to_join._nj_xsub = 0.0
        node_to_join1._nj_distances = {}
        node_to_join1._nj_xsub = 0.0
        node_to_join2._nj_distances = {}
        node_to_join2._nj_xsub = 0.0

        # add the new node to the pool of nodes
        node_pool.append(new_node)

        # adjust count
        n_pool -= 1
    blens = np.maximum(blens, eps)
    return peel, blens