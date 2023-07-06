import logging
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass


@dataclass
class TreeNode:
    index: int
    parent: int
    children: list[int]
    is_leaf: bool
    is_root: bool


class TreeTopologyBase:
    """ Tree topology
    """

    @property
    def parent_indices(self):
        raise NotImplementedError

    @property
    def children_indices(self):
        raise NotImplementedError

    def preorder_traverse(self):
        n_tips = self.n_tips
        root_index = len(self.parent_indices)
        stacks = [root_index]
        parents = self.parent_indices
        children = self.children_indices
        while stacks:
            idx = stacks.pop(0)
            is_leaf = idx < n_tips
            is_root = idx == root_index
            node = TreeNode(index=idx, parent=-1 if is_root else parents[idx], children=children[idx], is_leaf=is_leaf, is_root=is_root)
            yield node
            for ch_idx in children[idx]:
                stacks.append(ch_idx)

    def postorder_traverse(self):
        n_tips = self.n_tips
        root_index = len(self.parent_indices)
        stacks = [root_index]
        nodes = []
        parents = self.parent_indices
        children = self.children_indices
        while stacks:
            idx = stacks.pop()   # take last
            is_leaf = idx < n_tips
            is_root = idx == root_index
            node = TreeNode(index=idx, parent=-1 if is_root else parents[idx], children=children[idx], is_leaf=is_leaf, is_root=is_root)
            nodes.append(node)
            for ch_idx in children[idx]:
                stacks.append(ch_idx)
        return iter(nodes[::-1])

    @property
    def rooted(self):
        raise NotImplementedError

    @property
    def n_tips(self):
        raise NotImplementedError

    @property
    def n_nodes(self):
        raise NotImplementedError

    @property
    def n_branches(self):
        raise NotImplementedError

    def as_ete_tree(self, taxon_names=None, branch_lengths=None):
        return convert_to_ete_tree(self, taxon_names=taxon_names, branch_lengths=branch_lengths)


def get_edge_index_tensor(utree: TreeTopologyBase):  # torch.Tensor with shape (n_node, 3)  # only available for binary utree
    assert not utree.rooted
    edges = [[] for _ in range(utree.n_nodes)]
    for i, idx in enumerate(utree.parent_indices):
        edges[i].append(idx)
    for i, idxs in enumerate(utree.children_indices):
        if idxs:
            edges[i].extend(idxs)
        else:
            edges[i].extend([-1, -1])
    return torch.as_tensor(edges, dtype=torch.long)


# Original code from https://github.com/zcrabbit/vbpi-gnn/blob/963045d9568019eeace0d115d41321b51e18d4ce/gnn_branchModel.py#L30-L67
# Reference: C.Zhang (ICLR 2023)
def get_learnable_feature(utree: TreeTopologyBase):  # torch.Tensor with shape (n_node,)
    assert not utree.rooted
    n_tips = utree.n_tips
    n_nodes = utree.n_nodes
    leaf_features = torch.eye(n_tips)
    #c = torch.zeros((n_nodes,))
    c = torch.zeros((n_nodes, n_tips))
    d = torch.zeros((n_nodes, n_tips))
    f = torch.zeros((n_nodes, n_tips))

    for node in utree.postorder_traverse():  # root node is definitely not a leaf node
        idx = node.index
        #print (node)
        if node.is_leaf:
            #c[idx] = c[idx] #0.
            d[idx] = leaf_features[idx]
        else:
            ch_c_sum = c[node.children].sum(dim=0)
            ch_d_sum = d[node.children].sum(dim=0)
            c[idx] = 1./(3. - ch_c_sum)
            d[idx] = c[idx] * ch_d_sum

    for node in utree.preorder_traverse():
        idx = node.index
        if not node.is_root:
            d[idx] = c[idx] * d[node.parent] + d[idx]
        #f[idx] = d[idx]

    return d


def convert_to_ete_tree(tree: TreeTopologyBase, taxon_names=None, use_branch_length=None, branch_lengths=None):
    import ete3
    if taxon_names is None:
        taxon_names = [str(i+1) for i in range(tree.n_tips)]
    assert len(taxon_names) == tree.n_tips, taxon_names
    taxons = dict(enumerate(taxon_names))   # index to taxon name
    if branch_lengths is None:
        if use_branch_length is None:
            use_branch_length = isinstance(tree, TreeMetricBase)
        if use_branch_length:
            ch_blens = tree.branch_lengths.detach().cpu().numpy()
        else:
            ch_blens = None
    else:
        ch_blens = branch_lengths.detach().cpu().numpy()

    root_idx = max(tree.parent_indices)
    ete_root = ete3.Tree()
    idx_ete_nodes = {root_idx: ete_root}

    for node in tree.preorder_traverse():
        if node.is_leaf:
            continue
        ete_node = idx_ete_nodes.pop(node.index)
        for ch_idx in node.children:
            name = taxons.get(ch_idx)
            props = {'name': name}
            if ch_blens is not None:
                props['dist'] = ch_blens[ch_idx]
            ete_ch_node = ete_node.add_child(**props)
            if name is None:  # internal
                idx_ete_nodes[ch_idx] = ete_ch_node

    return ete_root


class TreeMetricBase(TreeTopologyBase):
    """ Tree topology and edge lengths
    """

    @property
    def n_branches(self):
        return self.branch_lengths

    @property
    def branch_lengths(self):
        raise NotImplementedError


class TreeMetric(TreeMetricBase):
    @property
    def parent_indices(self):
        return self._parent_indices

    @property
    def children_indices(self):
        return self._children_indices

    @property
    def n_tips(self):
        return self._n_tips

    @property
    def n_nodes(self):
        return self._n_nodes

    @property
    def rooted(self):
        return self._rooted

    @property
    def branch_lengths(self):
        return self._branch_lengths

    def __init__(self, parent_indices, branch_lengths, rooted=True):
        self._parent_indices = np.asarray(parent_indices)  # should not be a tensor
        children_indices = [[] for _ in range(len(self._parent_indices) + 1)]  # TODO efficiency?
        for idx, pa_idx in enumerate(self._parent_indices):
            children_indices[pa_idx].append(idx)
        self._children_indices = children_indices
        self._n_tips = min(self._parent_indices)
        self._n_nodes = len(self._parent_indices) + 1
        self._branch_lengths = torch.as_tensor(branch_lengths)
        self._rooted = rooted

    def get_unrooted(self):
        assert self.rooted, 'only apply once to binary rooted tree'
        root_idx = max(self.parent_indices)  # list
        assert self.parent_indices[-1] == root_idx  # the last parent idx should be the current root
        new_root_idx = root_idx - 1
        root_child_idxs = np.arange(len(self.parent_indices))[self.parent_indices == root_idx]
        assert len(root_child_idxs) == 2, (root_child_idxs, root_idx, self.parent_indices, self.parent_indices == root_idx)
        new_branch_lengths = self.branch_lengths[:-1].clone().detach()
        new_branch_lengths[root_child_idxs[0]] = self.branch_lengths[root_child_idxs].sum()   # merge blens values to new_root
        new_parent_indices = self.parent_indices[:-1]
        new_parent_indices[root_child_idxs[0]] = new_root_idx
        #new_parent_indices = [min(i, new_root_idx) for i in self.parent_indices[:-1]]
        return self.__class__(parent_indices=new_parent_indices, branch_lengths=new_branch_lengths, rooted=False)

    def clone(self, branch_lengths=None):
        blens = self.branch_lengths if branch_lengths is None else branch_lengths
        assert len(blens) == len(self.branch_lengths)
        return self.__class__(parent_indices=self.parent_indices, branch_lengths=blens, rooted=self.rooted)


class TipEmbedBase:
    """
    - coord  (N, d)
    """
    def detach(self):
        raise NotImplementedError


class TipEmbedModelBase(nn.Module):
    @property
    def state_size(self) -> torch.Size:
        raise NotImplementedError

    @torch.no_grad()
    def sample_embeds(self, mc_samples=1):
        return self.rsample_embeds(mc_samples=mc_samples)

    def rsample_embeds(self, mc_samples=1):
        raise NotImplementedError

    def log_prior(self, embed: TipEmbedBase):
        raise NotImplementedError

    def get_mean_distance_matrix(self):
        return self.mean_distance_matrix


class CondEmbedModelBase(nn.Module):
    """
    """
    def get_log_prob(self, embed: TipEmbedBase, utree_metric: TreeMetricBase):
        """
        Returns log p(tip_embed | utree_metric)
        """
        raise NotImplementedError
