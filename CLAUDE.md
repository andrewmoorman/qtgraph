# qtgraph

A PyTorch Geometric utility package for memory-efficient graph partitioning
with k-hop overlap, designed for training GNNs on large spatial graphs.

The PRIMARY GOAL of this package is performance.

## Motivation

The target use case is spatial point-cloud graphs with billions of edges. The
2D spatial structure allows nodes to be assigned to non-overlapping spatial
regions (partitions). The trick is to physically permute the nodes and edges
of the graph in place so that each chunk's data occupies a contiguous block in
memory. At training time, a single contiguous read loads one chunk and its
k-hop overlap — no random access across the full graph.

The k-hop overlap is pre-computed to match the GNN's message-passing depth,
so every node in a core chunk has a complete k-hop neighborhood available at
training time. Eventually, the permuted graph will be stored on disk and
chunks will be loaded individually during training without deserializing the
full graph.

## Source layout

```
src/qtgraph/
    __init__.py               # re-exports public API
    partition/
        __init__.py           # public API + partition(), hetero_partition()
        _partition.py         # _Partition, _HeteroPartition (internal)
        _base.py              # _PartitionedBase (abstract)
        data.py               # PartitionedData
        hetero_data.py        # PartitionedHeteroData
        builder.py            # PartitionedDataBuilder
        hetero_builder.py     # PartitionedHeteroDataBuilder
tests/
    conftest.py               # shared fixtures
    generators.py             # SpatialGridGraph for spatial fixtures
    test_*.py                 # unit tests
    test_benchmarks.py        # benchmark tests
```

## Commands

```bash
pixi run pytest            # run unit tests (default env)
pixi run -e dev pytest     # run tests including benchmarks (dev env)
```

## Core concepts

### Terminology

- **Partition** — the input label assignment that groups nodes into
  non-overlapping regions.
- **Core chunk** — a contiguous block in the permuted graph corresponding
  to one input partition, after overlap nodes have been removed.
- **Overlap chunk** — a contiguous block formed by the intersection of two
  or more core chunks' k-hop neighborhoods. Nodes are strictly moved into
  overlap chunks; they are never duplicated.
- **Intra-chunk edges** — edges whose both endpoints are in the same chunk
  (core or overlap).
- **Inter-chunk edges** — edges whose endpoints are in two different chunks.

### Partitioning scheme

Given a graph and a node-level partition label assignment, the graph is
permuted in place — no copy is made, and no nodes or edges are discarded.
The total graph size (nodes and edges) is unchanged after partitioning.

The permuted node array is divided into contiguous chunks of two kinds:

- **Core chunks** — one per input partition, containing the nodes assigned
  to that partition after overlap nodes have been removed.
- **Overlap chunks** — one per unique intersection of k-hop neighborhoods
  across core chunks. An overlap chunk belongs to the set of core chunks
  whose k-hop expansions all include those nodes.

Chunks are assigned a unified integer index: core chunks `0..P-1`, overlap
chunks `P..P+O-1`.

### Overlap formation

**Boundary nodes** — nodes in a core chunk that have at least one outgoing
edge to a node in a different chunk.

**k-hop expansion** — from the boundary nodes of core chunk `p`, follow
edges (in edge direction only for directed graphs) up to `k` hops. Any
foreign node reachable within `k` hops is a candidate overlap node for `p`.

**Overlap assignment** — each candidate node is assigned to the unique
overlap chunk defined by the exact set of core chunks whose k-hop expansion
includes it. A node reachable from both chunk A and chunk C (but no others)
belongs to overlap chunk `(A, C)`. Overlap chunks spanning more than two
core chunks are possible.

### Edge categories

All edges are preserved. The permuted edge array is divided into two
categories, each stored as a contiguous block:

- **Intra-chunk edges** — both endpoints in the same chunk (core or
  overlap). Stored once per chunk.
- **Inter-chunk edges** — endpoints in two different chunks. Stored once
  per chunk pair.

### Retrieval

`__getitem__` accepts `int | list[int]` and returns a single merged subgraph
containing:

1. All nodes from each requested core chunk and every overlap chunk whose
   origins include at least one requested core chunk.
2. All intra-chunk edges for every included chunk.
3. All inter-chunk edges for every pair of included chunks.

Overlap chunks shared between two requested core chunks are included exactly
once. Edge indices in the returned subgraph are remapped to local node
ordering.

## Class architecture

### Internal containers

`_Partition` and `_HeteroPartition` are private dataclasses that index into
the permuted arrays. They are never exposed in the public API.

`_Partition` fields (P core chunks, O overlap chunks, M inter-chunk pairs):

- `core_node_indptr` — `(P+1,)` CSR pointers into permuted node array
- `core_edge_indptr` — `(P+1,)` CSR pointers for intra-chunk edges
- `overlap_node_indptr` — `(O+1,)` CSR pointers for overlap nodes
- `overlap_edge_indptr` — `(O+1,)` CSR pointers for overlap intra-chunk edges
- `overlap_origins_indptr` — `(O+1,)` CSR pointers into origins values
- `overlap_origins_values` — flat tensor of originating core chunk indices
- `inter_edge_indptr` — `(M+1,)` CSR pointers for inter-chunk edge blocks
- `inter_chunk_pairs` — `(M, 2)` unified chunk indices per inter-chunk block

`_HeteroPartition` mirrors this structure with per-type dicts for node and
edge fields. Origins and `inter_chunk_pairs` are shared across types.

### Retrieval classes

`PartitionedData` and `PartitionedHeteroData` extend `_PartitionedBase` and
are kept as separate classes. They wrap the permuted graph and its partition
index, and expose `__getitem__` and `__len__`. Homo and hetero are not unified
into a single class.

### Builder classes

`PartitionedDataBuilder` and `PartitionedHeteroDataBuilder` own all
construction logic. They accept the original graph, dense partition labels,
and `k`, permute the graph in place, and return the corresponding
`PartitionedData` or `PartitionedHeteroData`.

The builder pattern separates the one-time construction step from the
many-times retrieval step, which is important for the eventual disk-storage
use case: the builder writes the permuted graph and partition index to disk
once; `PartitionedData` is then constructed directly from the saved state
without re-running the partitioning.

### Convenience functions

`partition(data, labels, k)` and `hetero_partition(data, labels, k)` wrap
the builders for the common case where construction and retrieval happen
in the same session.

## Open questions

**Chunk memory layout** — storing a core chunk and its associated overlap
chunks contiguously would reduce single-partition retrieval to one read.
The complication is that overlap chunks shared between two core chunks
cannot be adjacent to both, so any layout is a tradeoff between
single-partition and multi-partition access patterns. The CSR-pointer design
in `_Partition` is layout-agnostic and does not constrain this decision.
Defer until spatial benchmarks are in place and discontiguous reads are
confirmed as a bottleneck.

## Key constraints

- Nodes are strictly moved into overlap chunks, never duplicated.
- Total node and edge count is identical before and after partitioning.
- The graph is permuted in place; no copy of the original is retained.
- k-hop expansion follows edge direction for directed graphs.
- k counts hops from the boundary nodes outward.
- Initialization accepts dense partition labels only; `_Partition` is
  internal and never constructed directly by users.
- The graph is static: permuted once, read many times during training.

## Implementation guidelines

Always search for and use existing implementations from `torch_geometric`,
`torch`, or `cugraph` before writing anything from scratch. If a needed
operation is not available in these packages, ask before adding a new
dependency.

All operations must use tensor ops. Python loops over nodes or edges are never
acceptable; this is the primary performance constraint.

## Style

- Line length: 80 characters
- Docstrings: NumPy style
- Type hints: used throughout, on all function signatures
- Comments: succinct, explain the why not the what; no section headers,
  dividers, or decorative formatting
- Variable names: legible and descriptive; avoid single-letter names except
  conventional indices (`i`, `j`, `k`)

## Imports

- Ordering: stdlib → third-party → local (PEP 8)
- Use `from x import y` unless a package has a strong established convention
  (e.g. `import torch`); follow those conventions as-is
- All imports at module level; no inline imports
- No non-standard aliases

## Testing

Tests live in `tests/`. Use `pytest` with real graph fixtures (small, concrete
graphs with known structure). Every component should have:

- Unit tests with explicit checks on node/edge counts, chunk membership,
  index pointers, overlap origins, and retrieved subgraph correctness
- Benchmark tests using `pytest-benchmark`, parameterized over graph sizes
  (e.g. 1k, 10k, 100k nodes), to catch performance regressions

Use `torch.utils.benchmark.Timer` for ad-hoc profiling during development;
use `pytest-benchmark` for regression tracking in CI.
