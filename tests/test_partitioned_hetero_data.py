"""Tests for PartitionedHeteroDataBuilder and PartitionedHeteroData.

Mirrors test_partitioned_data.py for the heterogeneous graph path.
Node IDs are unique within each type (x = arange per type).
"""
import torch
import pytest
from torch_geometric.data import HeteroData

from qtgraph import hetero_partition, PartitionedHeteroData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _edge_id_set_hetero(
    edge_index: torch.Tensor,
    src_x: torch.Tensor,
    dst_x: torch.Tensor,
) -> set:
    """Set of (src_id, dst_id) int pairs using per-type x as node IDs."""
    src = src_x[edge_index[0]].squeeze(1).long().tolist()
    dst = dst_x[edge_index[1]].squeeze(1).long().tolist()
    return set(zip(src, dst))


def _node_id_set(x: torch.Tensor) -> set:
    return set(x.squeeze(1).long().tolist())


def _all_edge_ranges_hetero(
    pd: PartitionedHeteroData,
    et: tuple,
) -> list[tuple[int, int]]:
    p = pd._partition
    n_core = len(pd)
    n_overlap = p.overlap_node_indptr[et[0]].shape[0] - 1
    n_inter = p.inter_chunk_pairs.shape[0]

    ranges = []
    for i in range(n_core):
        ranges.append((
            p.core_edge_indptr[et][i].item(),
            p.core_edge_indptr[et][i + 1].item(),
        ))
    for o in range(n_overlap):
        ranges.append((
            p.overlap_edge_indptr[et][o].item(),
            p.overlap_edge_indptr[et][o + 1].item(),
        ))
    for m in range(n_inter):
        ranges.append((
            p.inter_edge_indptr[et][m].item(),
            p.inter_edge_indptr[et][m + 1].item(),
        ))
    return ranges


# ---------------------------------------------------------------------------
# Construction invariants
# ---------------------------------------------------------------------------

class TestCountInvariants:
    def test_node_count_preserved(self, minimal_overlap_hetero):
        c = minimal_overlap_hetero
        n_before = {nt: c["data"][nt].num_nodes for nt in c["data"].node_types}
        pd = hetero_partition(c["data"], c["labels"], c["k"])
        for nt in pd._data.node_types:
            assert pd._data[nt].num_nodes == n_before[nt]

    def test_edge_count_preserved(self, minimal_overlap_hetero):
        c = minimal_overlap_hetero
        e_before = {
            et: c["data"][et].edge_index.shape[1]
            for et in c["data"].edge_types
        }
        pd = hetero_partition(c["data"], c["labels"], c["k"])
        for et in pd._data.edge_types:
            assert pd._data[et].edge_index.shape[1] == e_before[et]

    def test_num_core_chunks(self, minimal_overlap_hetero):
        c = minimal_overlap_hetero
        all_labels = torch.cat(list(c["labels"].values()))
        n_partitions = all_labels.unique().shape[0]
        pd = hetero_partition(c["data"], c["labels"], c["k"])
        assert len(pd) == n_partitions

    def test_len(self, minimal_overlap_hetero):
        c = minimal_overlap_hetero
        all_labels = torch.cat(list(c["labels"].values()))
        n_partitions = all_labels.unique().shape[0]
        pd = hetero_partition(c["data"], c["labels"], c["k"])
        assert len(pd) == n_partitions


# ---------------------------------------------------------------------------
# CSR index structure
# ---------------------------------------------------------------------------

class TestCSRIndex:
    def test_core_node_indptr_shape(self, minimal_overlap_hetero):
        c = minimal_overlap_hetero
        pd = hetero_partition(c["data"], c["labels"], c["k"])
        p = pd._partition
        P = len(pd)
        for nt in pd._data.node_types:
            assert p.core_node_indptr[nt].shape[0] == P + 1

    def test_overlap_indptr_consistent_shape(
        self, minimal_overlap_hetero
    ):
        c = minimal_overlap_hetero
        pd = hetero_partition(c["data"], c["labels"], c["k"])
        p = pd._partition
        O = p.overlap_origins_indptr.shape[0] - 1
        for nt in pd._data.node_types:
            assert p.overlap_node_indptr[nt].shape[0] == O + 1
        for et in pd._data.edge_types:
            assert p.overlap_edge_indptr[et].shape[0] == O + 1

    def test_inter_edge_indptr_shape(self, minimal_overlap_hetero):
        c = minimal_overlap_hetero
        pd = hetero_partition(c["data"], c["labels"], c["k"])
        p = pd._partition
        M = p.inter_chunk_pairs.shape[0]
        for et in pd._data.edge_types:
            assert p.inter_edge_indptr[et].shape[0] == M + 1

    def test_core_node_indptr_nondecreasing(self, minimal_overlap_hetero):
        c = minimal_overlap_hetero
        pd = hetero_partition(c["data"], c["labels"], c["k"])
        p = pd._partition
        for nt in pd._data.node_types:
            assert (p.core_node_indptr[nt].diff() >= 0).all()

    def test_overlap_node_indptr_nondecreasing(
        self, minimal_overlap_hetero
    ):
        c = minimal_overlap_hetero
        pd = hetero_partition(c["data"], c["labels"], c["k"])
        p = pd._partition
        for nt in pd._data.node_types:
            assert (p.overlap_node_indptr[nt].diff() >= 0).all()

    def test_core_edge_indptr_nondecreasing(self, minimal_overlap_hetero):
        c = minimal_overlap_hetero
        pd = hetero_partition(c["data"], c["labels"], c["k"])
        p = pd._partition
        for et in pd._data.edge_types:
            assert (p.core_edge_indptr[et].diff() >= 0).all()

    def test_node_ranges_cover_all_nodes_no_overlap(
        self, minimal_overlap_hetero
    ):
        """Every node index appears in exactly one chunk per type."""
        c = minimal_overlap_hetero
        pd = hetero_partition(c["data"], c["labels"], c["k"])
        p = pd._partition
        n_core = len(pd)
        n_overlap = p.overlap_origins_indptr.shape[0] - 1

        for nt in pd._data.node_types:
            N = pd._data[nt].num_nodes
            indices = []
            for i in range(n_core):
                lo = p.core_node_indptr[nt][i].item()
                hi = p.core_node_indptr[nt][i + 1].item()
                indices.extend(range(lo, hi))
            for o in range(n_overlap):
                lo = p.overlap_node_indptr[nt][o].item()
                hi = p.overlap_node_indptr[nt][o + 1].item()
                indices.extend(range(lo, hi))
            assert len(indices) == N
            assert len(set(indices)) == N

    def test_edge_ranges_cover_all_edges_no_overlap(
        self, minimal_overlap_hetero
    ):
        c = minimal_overlap_hetero
        pd = hetero_partition(c["data"], c["labels"], c["k"])
        for et in pd._data.edge_types:
            E = pd._data[et].edge_index.shape[1]
            ranges = sorted(
                (lo, hi)
                for lo, hi in _all_edge_ranges_hetero(pd, et)
                if lo < hi
            )
            covered = 0
            for lo, hi in ranges:
                assert lo == covered
                covered = hi
            assert covered == E


# ---------------------------------------------------------------------------
# Edge reconstruction
# ---------------------------------------------------------------------------

class TestEdgeReconstruction:
    def test_exact_edge_set_preserved(self, minimal_overlap_hetero):
        c = minimal_overlap_hetero
        et = ('A', 'to', 'B')
        original_edges = _edge_id_set_hetero(
            c["data"][et].edge_index,
            c["data"]['A'].x,
            c["data"]['B'].x,
        )
        pd = hetero_partition(c["data"], c["labels"], c["k"])
        permuted_edges = _edge_id_set_hetero(
            pd._data[et].edge_index,
            pd._data['A'].x,
            pd._data['B'].x,
        )
        assert permuted_edges == original_edges


# ---------------------------------------------------------------------------
# Overlap structure
# ---------------------------------------------------------------------------

class TestOverlapStructure:
    def test_has_one_overlap_chunk(self, minimal_overlap_hetero):
        c = minimal_overlap_hetero
        pd = hetero_partition(c["data"], c["labels"], c["k"])
        n_overlap = pd._partition.overlap_origins_indptr.shape[0] - 1
        assert n_overlap == 1

    def test_overlap_B_node_count(self, minimal_overlap_hetero):
        """B node 0 is the only node that enters the overlap chunk."""
        c = minimal_overlap_hetero
        pd = hetero_partition(c["data"], c["labels"], c["k"])
        p = pd._partition
        overlap_b_size = (
            p.overlap_node_indptr['B'][1]
            - p.overlap_node_indptr['B'][0]
        ).item()
        assert overlap_b_size == 1

    def test_overlap_A_node_count(self, minimal_overlap_hetero):
        """No A nodes enter the overlap chunk."""
        c = minimal_overlap_hetero
        pd = hetero_partition(c["data"], c["labels"], c["k"])
        p = pd._partition
        overlap_a_size = (
            p.overlap_node_indptr['A'][1]
            - p.overlap_node_indptr['A'][0]
        ).item()
        assert overlap_a_size == 0

    def test_overlap_origins(self, minimal_overlap_hetero):
        c = minimal_overlap_hetero
        pd = hetero_partition(c["data"], c["labels"], c["k"])
        p = pd._partition
        assert set(p.overlap_origins_values.tolist()) == {0, 1}

    def test_total_nodes_accounted(self, minimal_overlap_hetero):
        c = minimal_overlap_hetero
        pd = hetero_partition(c["data"], c["labels"], c["k"])
        p = pd._partition
        n_core = len(pd)
        n_overlap = p.overlap_origins_indptr.shape[0] - 1
        for nt in pd._data.node_types:
            core_total = sum(
                (p.core_node_indptr[nt][i + 1]
                 - p.core_node_indptr[nt][i]).item()
                for i in range(n_core)
            )
            overlap_total = sum(
                (p.overlap_node_indptr[nt][o + 1]
                 - p.overlap_node_indptr[nt][o]).item()
                for o in range(n_overlap)
            )
            assert core_total + overlap_total == pd._data[nt].num_nodes


# ---------------------------------------------------------------------------
# Feature permutation
# ---------------------------------------------------------------------------

class TestFeaturePermutation:
    def test_node_feature_set_preserved(self, minimal_overlap_hetero):
        c = minimal_overlap_hetero
        original = {
            nt: _node_id_set(c["data"][nt].x)
            for nt in c["data"].node_types
        }
        pd = hetero_partition(c["data"], c["labels"], c["k"])
        for nt in pd._data.node_types:
            assert _node_id_set(pd._data[nt].x) == original[nt]


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class TestRetrieval:
    def test_getitem_returns_heterodata(self, minimal_overlap_hetero):
        c = minimal_overlap_hetero
        pd = hetero_partition(c["data"], c["labels"], c["k"])
        assert isinstance(pd[0], HeteroData)

    def test_chunk0_exact_A_node_ids(self, minimal_overlap_hetero):
        """Core 0 contains both A nodes."""
        c = minimal_overlap_hetero
        pd = hetero_partition(c["data"], c["labels"], c["k"])
        assert _node_id_set(pd[0]['A'].x) == {0, 1}

    def test_chunk0_exact_B_node_ids(self, minimal_overlap_hetero):
        """Core 0 has no B core nodes; B node 0 comes from the overlap."""
        c = minimal_overlap_hetero
        pd = hetero_partition(c["data"], c["labels"], c["k"])
        assert _node_id_set(pd[0]['B'].x) == {0}

    def test_chunk1_exact_B_node_ids(self, minimal_overlap_hetero):
        """Core 1 contains B node 1; overlap adds B node 0."""
        c = minimal_overlap_hetero
        pd = hetero_partition(c["data"], c["labels"], c["k"])
        assert _node_id_set(pd[1]['B'].x) == {0, 1}

    def test_all_chunks_covers_all_nodes(self, minimal_overlap_hetero):
        c = minimal_overlap_hetero
        pd = hetero_partition(c["data"], c["labels"], c["k"])
        sub = pd[[0, 1]]
        assert _node_id_set(sub['A'].x) == {0, 1}
        assert _node_id_set(sub['B'].x) == {0, 1}

    def test_chunk0_exact_edge_set(self, minimal_overlap_hetero):
        """Core 0 retrieval: edges A0→B0 and A1→B0."""
        c = minimal_overlap_hetero
        pd = hetero_partition(c["data"], c["labels"], c["k"])
        sub = pd[0]
        et = ('A', 'to', 'B')
        edges = _edge_id_set_hetero(
            sub[et].edge_index, sub['A'].x, sub['B'].x
        )
        assert edges == {(0, 0), (1, 0)}

    def test_all_chunks_exact_edge_set(self, minimal_overlap_hetero):
        c = minimal_overlap_hetero
        et = ('A', 'to', 'B')
        original_edges = _edge_id_set_hetero(
            c["data"][et].edge_index,
            c["data"]['A'].x,
            c["data"]['B'].x,
        )
        pd = hetero_partition(c["data"], c["labels"], c["k"])
        sub = pd[[0, 1]]
        recovered = _edge_id_set_hetero(
            sub[et].edge_index, sub['A'].x, sub['B'].x
        )
        assert recovered == original_edges

    def test_edge_indices_within_bounds(self, minimal_overlap_hetero):
        c = minimal_overlap_hetero
        pd = hetero_partition(c["data"], c["labels"], c["k"])
        sub = pd[0]
        et = ('A', 'to', 'B')
        assert int(sub[et].edge_index[0].max()) < sub['A'].num_nodes
        assert int(sub[et].edge_index[1].max()) < sub['B'].num_nodes

    def test_edge_indices_within_bounds_multi(
        self, minimal_overlap_hetero
    ):
        c = minimal_overlap_hetero
        pd = hetero_partition(c["data"], c["labels"], c["k"])
        sub = pd[[0, 1]]
        et = ('A', 'to', 'B')
        assert int(sub[et].edge_index[0].max()) < sub['A'].num_nodes
        assert int(sub[et].edge_index[1].max()) < sub['B'].num_nodes
