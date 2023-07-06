import torch
import numpy as np


class SequenceData:
    @property
    def names(self):
        raise NotImplementedError

    @property
    def n_seqs(self):
        raise NotImplementedError

    @property
    def seq_len(self):
        raise NotImplementedError

    @property
    def n_chars(self):
        raise NotImplementedError

    def get_tensor(self):
        raise NotImplementedError

    def __len__(self):
        return self.n_seqs


class DNASequenceData(SequenceData):
    # From https://github.com/zcrabbit/vbpi-torch/blob/ff86cf0c47a5753f5cc5b4dfe0b6ed783ab22669/unrooted/phyloModel.py#L7-L11
    nuc2vec = {'A':[1.,0.,0.,0.], 'G':[0.,1.,0.,0.], 'C':[0.,0.,1.,0.], 'T':[0.,0.,0.,1.],
           '-':[1.,1.,1.,1.], '?':[1.,1.,1.,1.], 'N':[1.,1.,1.,1.], 'R':[1.,1.,0.,0.],
           'Y':[0.,0.,1.,1.], 'S':[0.,1.,1.,0.], 'W':[1.,0.,0.,1.], 'K':[0.,1.,0.,1.],
           'M':[1.,0.,1.,0.], 'B':[0.,1.,1.,1.], 'D':[1.,1.,0.,1.], 'H':[1.,0.,1.,1.],
           'V':[1.,1.,1.,0.], '.':[1.,1.,1.,1.], 'U':[0.,0.,0.,1.]}

    def __init__(self, seq_strs, names=None):  # [seq strs]
        self._seq_strs = seq_strs
        self._names = names

    @property
    def names(self):
        return self._names

    @property
    def n_seqs(self):
        return len(self._seq_strs)

    @property
    def seq_len(self):
        return len(self._seq_strs[0])

    @property
    def n_chars(self):
        return 4

    def get_tensor(self, dtype=None):   # (n_seqs, seq_len, n_chars)
        dtype = dtype or torch.get_default_dtype()
        return torch.tensor(np.asarray([[self.nuc2vec[c] for c in st.upper()] for st in self._seq_strs]), dtype=dtype)
