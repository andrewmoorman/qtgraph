import torch
from torch_geometric.data import Data

from qtgraph.partition._partition import _Partition
from qtgraph.partition.data import PartitionedData


class PartitionedDataBuilder:
    """Builds a PartitionedData from a homogeneous graph and labels.

    Permutes the graph in place and constructs the _Partition index.
    Steps performed during build():
        1. Identify boundary nodes for each core chunk.
        2. Compute k-hop expansions from boundary nodes.
        3. Assign nodes to overlap chunks by expansion intersection.
        4. Permute nodes into contiguous chunk order.
        5. Categorize and permute edges into intra- and inter-chunk
           blocks.
        6. Build the _Partition index.

    Parameters
    ----------
    data : Data
        Input graph. Modified in place during build().
    labels : torch.Tensor
        Shape (N,). Integer partition label per node.
    k : int
        Number of hops to expand from each chunk's boundary nodes.
    """

    def __init__(
        self,
        data: Data,
        labels: torch.Tensor,
        k: int,
    ):
        ...

    def build(self) -> PartitionedData:
        """Permute the graph and return a PartitionedData.

        Returns
        -------
        PartitionedData
            Wrapper around the permuted graph and its partition index.
        """
        ...
