from torch_geometric.data import HeteroData

from qtgraph.partition._base import _PartitionedBase
from qtgraph.partition._partition import _HeteroPartition


class PartitionedHeteroData(_PartitionedBase):
    """Permuted heterogeneous graph with a pre-built partition index.

    Wraps a permuted PyG HeteroData object and a _HeteroPartition
    index. Node and edge ranges are tracked per type; overlap origins
    and inter-chunk pairs are shared across types.

    Constructed by PartitionedHeteroDataBuilder, or loaded from disk
    by passing a pre-built _HeteroPartition alongside the permuted
    HeteroData.

    Parameters
    ----------
    data : HeteroData
        Permuted graph. Node and edge attributes per type are stored
        in chunk order.
    partition : _HeteroPartition
        Index into the permuted arrays per node and edge type.
    """

    def __init__(
        self,
        data: HeteroData,
        partition: _HeteroPartition,
    ):
        ...

    def __getitem__(self, index: int | list[int]) -> HeteroData:
        """Return the merged subgraph for one or more core chunks.

        Parameters
        ----------
        index : int or list[int]
            Index or indices of the core chunk(s) to retrieve.

        Returns
        -------
        HeteroData
            Merged subgraph with remapped edge indices per edge type.
        """
        ...

    def __len__(self) -> int:
        """Return the number of core chunks."""
        ...
