import os

from index import InvertedIndex
from lib.semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.doc_lengths_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:

        bm25 = self._bm25_search(query, limit * 500)
        list_bm25 = [score for _, score in bm25]

        sc = self.semantic_search.search_chunks(query, limit * 500)
        list_sc = [doc["score"] for doc in sc]

        normalize_bm25 = normalize(list_bm25)
        normalize_sc = normalize(list_sc)

        combined = {}

        for (doc_id, _), norm_score in zip(bm25, normalize_bm25):
            movie = self.idx.docmap[doc_id]
            combined[doc_id] = {
                "document": {
                    "id": movie["id"],
                    "title": movie["title"],
                    "document": movie["description"][:100],
                },
                "bm25_score": norm_score,
                "semantic_score": 0.0,
            }

        for doc, norm_score in zip(sc, normalize_sc):
            doc_id = doc['id']
            if doc_id in combined:
                combined[doc_id]['semantic_score'] = norm_score
            else:
                movie = self.idx.docmap[doc_id]
                combined[doc_id] = {
                    "document": {
                        "id": movie["id"],
                        "title": movie["title"],
                        "document": movie["description"][:100],
                    },
                    "bm25_score": 0.0,
                    "semantic_score": norm_score,
                }

        results = []
        for entry in combined.values():
            h_score = hybrid_score(entry["bm25_score"], entry["semantic_score"], alpha)
            entry["hybrid_score"] = h_score
            results.append(entry)

        results = sorted(results, key=lambda entry: entry["hybrid_score"], reverse=True)

        results = results[:limit]
        return results

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        bm25 = self._bm25_search(query,limit * 500)


        sc = self.semantic_search.search_chunks(query, limit * 500)



        combined = {}

        for rank, (doc_id, _) in enumerate(bm25, start=1):
            movie = self.idx.docmap[doc_id]
            combined[doc_id] = {
                "document": {
                    "id": movie["id"],
                    "title": movie["title"],
                    "document": movie["description"][:100],
                },
                "bm25_rank": rank,
                "semantic_rank": None,
            }

        for rank, doc in enumerate(sc, start=1):
            doc_id = doc['id']
            if doc_id in combined:
                combined[doc_id]['semantic_rank'] = rank
            else:
                movie = self.idx.docmap[doc_id]
                combined[doc_id] = {
                    "document": {
                        "id": movie["id"],
                        "title": movie["title"],
                        "document": movie["description"][:100],
                    },
                    "bm25_rank": None,
                    "semantic_rank": rank,
                }

        results = []
        for entry in combined.values():
            total = 0.0
            if entry["bm25_rank"] is not None:
                total += rrf_score(entry["bm25_rank"], k)

            if entry["semantic_rank"] is not None:
                total += rrf_score(entry["semantic_rank"], k)
            entry['rrf'] = total
            results.append(entry)

        results = sorted(results, key=lambda entry: entry["rrf"], reverse=True)

        return results



def normalize(lis):
    if len(lis) == 0:
        return []
    mini = min(lis)
    maxi = max(lis)

    if mini == maxi:
        return [1.0 for _ in lis]

    return [(x - mini) / (maxi - mini) for x in lis]


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)
