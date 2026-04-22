import torch
from dataclasses import dataclass
from torch_geometric.data import Data


@dataclass
class SpatialGraphCase:
    """Output of SpatialGridGraph.build().

    Attributes
    ----------
    data : Data
        PyG Data object with node coordinates as features and a
        4-connected grid edge structure.
    labels : torch.Tensor
        Integer partition label per node, assigned by tile membership.
    n_tiles : int
        Total number of square tiles (partitions).
    tile_size : int
        Side length of each tile in nodes.
    """
    data: Data
    labels: torch.Tensor
    n_tiles: int
    tile_size: int


class SpatialGridGraph:
    """Generates a 2D spatial grid graph partitioned into square tiles.

    Nodes are placed on an integer grid. Edges follow a 4-connected
    (up/down/left/right) pattern. Partition labels are assigned by
    tile membership. Node features are the (x, y) coordinates.

    Parameters
    ----------
    grid_size : int
        Side length of the full grid in nodes. Must be divisible by
        tile_size.
    tile_size : int
        Side length of each square tile in nodes.
    directed : bool
        If True, edges point right and down only. If False, edges are
        added in both directions.
    edge_features : bool
        If True, edge_attr is set to the Euclidean distance for each
        edge (always 1.0 for a grid).
    seed : int
        Random seed (reserved for future stochastic connectivity).
    """

    def __init__(
        self,
        grid_size: int,
        tile_size: int,
        directed: bool = True,
        edge_features: bool = False,
        seed: int = 0,
    ):
        assert grid_size % tile_size == 0, (
            "grid_size must be divisible by tile_size"
        )
        self.grid_size = grid_size
        self.tile_size = tile_size
        self.directed = directed
        self.edge_features = edge_features
        self.seed = seed

    def build(self) -> SpatialGraphCase:
        """Build and return the spatial graph case.

        Returns
        -------
        SpatialGraphCase
            Contains the Data object, partition labels, and tile
            metadata.
        """
        n = self.grid_size
        t = self.tile_size
        n_tiles_per_side = n // t

        # Node positions: row-major order, coords as floats
        idx = torch.arange(n * n)
        row = idx // n
        col = idx % n
        x = torch.stack([col, row], dim=1).float()

        # Tile membership label for each node
        tile_row = row // t
        tile_col = col // t
        labels = (tile_row * n_tiles_per_side + tile_col).long()

        # 4-connected edges (right and down)
        right_mask = col < n - 1
        down_mask = row < n - 1
        src = torch.cat([idx[right_mask], idx[down_mask]])
        dst = torch.cat([idx[right_mask] + 1, idx[down_mask] + n])

        if not self.directed:
            src = torch.cat([src, dst])
            dst = torch.cat([dst, src[:src.shape[0] - dst.shape[0]]])
            # rebuild cleanly to avoid aliasing
            right_src = idx[right_mask]
            right_dst = right_src + 1
            down_src = idx[down_mask]
            down_dst = down_src + n
            src = torch.cat([
                right_src, right_dst, down_src, down_dst
            ])
            dst = torch.cat([
                right_dst, right_src, down_dst, down_src
            ])

        edge_index = torch.stack([src, dst])
        data = Data(x=x, edge_index=edge_index)

        if self.edge_features:
            data.edge_attr = torch.ones(
                edge_index.shape[1], 1
            )

        return SpatialGraphCase(
            data=data,
            labels=labels,
            n_tiles=n_tiles_per_side ** 2,
            tile_size=t,
        )
