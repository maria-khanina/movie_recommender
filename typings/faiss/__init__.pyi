import numpy
import numpy.typing

METRIC_INNER_PRODUCT: int

class HierarchicalGraph:
    efSearch: int
    efConstruction: int

class IndexHNSWFlat:
    hnsw: HierarchicalGraph
    ntotal: int

    def __init__(self, dimension: int, neighbor_count: int, metric: int, /) -> None: ...
    def add(self, vectors: numpy.typing.NDArray[numpy.float32], /) -> None: ...
    def search(
        self,
        queries: numpy.typing.NDArray[numpy.float32],
        result_count: int,
        /,
    ) -> tuple[
        numpy.typing.NDArray[numpy.float32],
        numpy.typing.NDArray[numpy.int64],
    ]: ...
