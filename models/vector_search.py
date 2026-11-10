"""Approximate maximum inner product search over movie vectors.

Embedding retrievers (two towers, factorization machine) score a movie by the
inner product of a user vector and a movie vector. The movie vectors are put once
into an HNSW graph index, and a query walks the graph instead of scoring the
whole catalog, so its cost grows roughly with the logarithm of the catalog size.
This is the search a catalog of a billion movies would need, and using it here
keeps the comparison with the other retrievers honest about query cost. The
search breadth is raised to the requested depth when the depth is larger, since
HNSW cannot return more results than its breadth.
"""
import faiss
import numpy
import numpy.typing

from models import models_settings as settings
from models.candidate_retrieval import MovieIds, SearchResult

Vectors = numpy.typing.NDArray[numpy.float32]


class MovieVectorIndex:
    def __init__(self, movie_ids: MovieIds, movie_vectors: Vectors) -> None:
        self._movie_ids = movie_ids
        self._index = faiss.IndexHNSWFlat(
            movie_vectors.shape[1],
            settings.HNSW_GRAPH_DEGREE,
            faiss.METRIC_INNER_PRODUCT,
        )
        self._index.hnsw.efConstruction = settings.HNSW_CONSTRUCTION_BREADTH
        self._index.add(numpy.ascontiguousarray(movie_vectors))

    def search(self, user_vectors: Vectors, depth: int) -> SearchResult:
        self._index.hnsw.efSearch = max(settings.HNSW_SEARCH_BREADTH, depth)
        scores, codes = self._index.search(
            numpy.ascontiguousarray(user_vectors),
            depth,
        )
        found_codes = numpy.maximum(codes, 0)
        found_scores = numpy.where(codes >= 0, scores, -numpy.inf)
        return SearchResult(
            movie_ids=self._movie_ids[found_codes],
            scores=found_scores.astype(numpy.float32),
        )
