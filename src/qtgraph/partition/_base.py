from abc import ABC, abstractmethod

from torch_geometric.data import Data, HeteroData


class _PartitionedBase(ABC):
    """Abstract base for PartitionedData and PartitionedHeteroData.

    Defines the shared retrieval interface. Subclasses implement
    __getitem__ for their respective graph types.
    """

    @abstractmethod
    def __getitem__(
        self,
        index: int | list[int],
    ) -> Data | HeteroData:
        """Return the merged subgraph for one or more core chunks.

        Parameters
        ----------
        index : int or list[int]
            Index or indices of the core chunk(s) to retrieve.
            All overlap chunks whose origins include at least one
            requested core chunk are included automatically. All
            intra-chunk and inter-chunk edges for the included set
            are included and remapped to local node indices.

        Returns
        -------
        Data or HeteroData
            Merged subgraph with node and edge attributes and
            locally remapped edge indices.
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of core chunks (input partitions)."""
        ...
