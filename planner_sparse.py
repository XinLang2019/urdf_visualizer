"""Block-assignment Jacobian buffer with the upstream callback's NumPy semantics.

Only declared Jacobian entries are stored, avoiding a dense constraints × variables
allocation on every IPOPT callback. Duplicate structure entries retain upstream order.
"""
import numpy as np
from scipy.sparse import csr_matrix

class SparseJacobian:
    def __init__(self,rows,cols,shape):
        codes=np.asarray(rows,dtype=np.int64)*shape[1]+np.asarray(cols,dtype=np.int64)
        unique,self.inverse=np.unique(codes,return_inverse=True)
        self.index=csr_matrix((np.arange(len(unique),dtype=np.int64)+1,(unique//shape[1],unique%shape[1])),shape=shape)
        self.values=np.zeros(len(unique));self.blocks={}
    def clear(self):self.values.fill(0)
    def __setitem__(self,key,value):
        row,col=key
        cache_key=tuple((s.start,s.stop,s.step) if isinstance(s,slice) else int(s) for s in key)
        if cache_key not in self.blocks:
            block=self.index[row,col]
            indices=np.asarray(block.toarray() if hasattr(block,'toarray') else block)
            if np.isscalar(row) and not np.isscalar(col):indices=indices.reshape(-1)
            if not np.isscalar(row) and np.isscalar(col):indices=indices.reshape(-1)
            self.blocks[cache_key]=(indices,indices!=0)
        indices,mask=self.blocks[cache_key];v=np.empty(indices.shape);v[...]=value
        if np.any(np.abs(v[~mask])>1e-12):raise ValueError('Undeclared nonzero constraint Jacobian entry')
        self.values[indices[mask]-1]=v[mask]
    def result(self):return self.values[self.inverse]
