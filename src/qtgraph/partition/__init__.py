import torch
from torch_geometric.data import Data, HeteroData

from qtgraph.partition._partition import NodeType
from qtgraph.partition.builder import PartitionedDataBuilder
from qtgraph.partition.data import PartitionedData
from qtgraph.partition.hetero_builder import PartitionedHeteroDataBuilder
from qtgraph.partition.hetero_data import PartitionedHeteroData

__all__ = [
    "PartitionedData",
    "PartitionedHeteroData",
    "PartitionedDataBuilder",
    "PartitionedHeteroDataBuilder",
    "partition",
    "hetero_partition",
]


def partition(
    data: Data,
    labels: torch.Tensor,
    k: int,
) -> PartitionedData:
    """Partition a homogeneous graph into contiguous chunks.

    Convenience wrapper around PartitionedDataBuilder. Permutes the
    graph in place and returns a PartitionedData ready for retrieval.

    Parameters
    ----------
    data : Data
        Input graph. Modified in place.
    labels : torch.Tensor
        Shape (N,). Integer partition label per node.
    k : int
        Number of hops to expand from each chunk's boundary nodes.

    Returns
    -------
    PartitionedData
    """
    return PartitionedDataBuilder(data, labels, k).build()


def hetero_partition(
    data: HeteroData,
    labels: dict[NodeType, torch.Tensor],
    k: int,
) -> PartitionedHeteroData:
    """Partition a heterogeneous graph into contiguous chunks.

    Convenience wrapper around PartitionedHeteroDataBuilder. Permutes
    the graph in place and returns a PartitionedHeteroData ready for
    retrieval.

    Parameters
    ----------
    data : HeteroData
        Input graph. Modified in place.
    labels : dict[NodeType, torch.Tensor]
        Integer partition label per node, keyed by node type.
    k : int
        Number of hops to expand from each chunk's boundary nodes.

    Returns
    -------
    PartitionedHeteroData
    """
    return PartitionedHeteroDataBuilder(data, labels, k).build()
