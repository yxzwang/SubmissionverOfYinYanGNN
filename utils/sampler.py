import dgl
from dgl.dataloading.negative_sampler import PerSourceUniform
import dgl.backend as F
class Uniform_and_bidirect(PerSourceUniform):
    """Negative sampler that randomly chooses negative destination nodes
    for each source node according to a uniform distribution and bidirect the sampling edges.

    return dgl.Graph
    """

    def __init__(self, k):
        self.k = k

    def _generate(self, g, eids, canonical_etype):
        _, _, vtype = canonical_etype
        shape = F.shape(eids)
        dtype = F.dtype(eids)
        ctx = F.context(eids)
        shape = (shape[0] * self.k,)
        src, _ = g.find_edges(eids, etype=canonical_etype)
        src = F.repeat(src, self.k, 0)
        dst = F.randint(shape, dtype, ctx, 0, g.num_nodes(vtype))
        graph=dgl.graph((src,dst))
        graph=dgl.to_bidirected(graph)
        return graph
