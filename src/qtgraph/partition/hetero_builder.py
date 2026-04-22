import torch
from torch_geometric.data import HeteroData

from qtgraph.partition._partition import _HeteroPartition, NodeType
from qtgraph.partition.hetero_data import PartitionedHeteroData


class PartitionedHeteroDataBuilder:
    """Builds a PartitionedHeteroData from a heterogeneous graph.

    Permutes the graph in place and constructs the _HeteroPartition
    index. Follows the same steps as PartitionedDataBuilder but
    operates per node and edge type.

    Parameters
    ----------
    data : HeteroData
        Input graph. Modified in place during build().
    labels : dict[NodeType, torch.Tensor]
        Integer partition label per node, keyed by node type.
    k : int
        Number of hops to expand from each chunk's boundary nodes.
    """

    def __init__(
        self,
        data: HeteroData,
        labels: dict[NodeType, torch.Tensor],
        k: int,
    ):
        ...

    def build(self) -> PartitionedHeteroData:
        """Permute the graph and return a PartitionedHeteroData.

        Returns
        -------
        PartitionedHeteroData
            Wrapper around the permuted graph and its partition index.
        """
        ...
