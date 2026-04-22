from torch_geometric.data import Data

from qtgraph.partition._base import _PartitionedBase
from qtgraph.partition._partition import _Partition


class PartitionedData(_PartitionedBase):
    """Permuted homogeneous graph with a pre-built partition index.

    Wraps a permuted PyG Data object and a _Partition index. The graph
    has been physically reordered so that each chunk's nodes and edges
    occupy a contiguous block in memory. Retrieval reads directly from
    those blocks without any further permutation.

    Constructed by PartitionedDataBuilder, or loaded from disk by
    passing a pre-built _Partition alongside the permuted Data.

    Parameters
    ----------
    data : Data
        Permuted graph. Node and edge attributes are stored in chunk
        order: core chunk 0, core chunk 1, ..., overlap chunk 0, ...
    partition : _Partition
        Index into the permuted arrays. Tracks node ranges, intra-chunk
        edge ranges, inter-chunk edge ranges, and overlap origins.
    """

    def __init__(self, data: Data, partition: _Partition):
        ...

    def __getitem__(self, index: int | list[int]) -> Data:
        """Return the merged subgraph for one or more core chunks.

        Parameters
        ----------
        index : int or list[int]
            Index or indices of the core chunk(s) to retrieve.

        Returns
        -------
        Data
            Merged subgraph with remapped edge indices.
        """
        ...

    def __len__(self) -> int:
        """Return the number of core chunks."""
        ...
