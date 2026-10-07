

from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
import re

class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        text = text.strip()
        if len(text) == 0:
            raise ValueError("No Text.")
        embeddings = self.model.encode([text])
        return embeddings[0]

    def build_embeddings(self, documents):
        self.documents = documents
        representation = []
        for document in self.documents:
            self.document_map[document['id']] = document
            representation.append(f"{document['title']}: {document['description']}")
        self.embeddings = self.model.encode(representation,show_progress_bar=True)
        np.save('cache/movie_embeddings.npy',self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for document in self.documents:
            self.document_map[document['id']] = document
        if os.path.isfile("cache/movie_embeddings.npy"):
            self.embeddings = np.load('cache/movie_embeddings.npy')
            if len(self.embeddings) == len(documents):
                return self.embeddings
            else:
                result = self.build_embeddings(documents)
                return result
        else:
            result = self.build_embeddings(documents)
            return result
    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        embed = self.generate_embedding(query)
        l = []
        for i in range(len(self.embeddings)):
            some_value = cosine_similarity(embed, self.embeddings[i] )

            l.append((some_value,self.documents[i]))
        pairs_sorted = sorted(l, key=lambda some_value: some_value[0], reverse=True)
        pairs_sorted = pairs_sorted[:limit]
        results = []
        for score , document in pairs_sorted:
            results.append({"score": score , "title": document['title'] , "description": document['description']})
        return results

    


def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")

def embed(text):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    ss = SemanticSearch()
    with open('data/movies.json','r') as f:
        movies = json.load(f)['movies']
    embeddings = ss.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(movies)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
)

def embed_query_text(query):
    ss = SemanticSearch()
    embed = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embed[:3]}")
    print(f"Shape: {embed.shape}")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def semantic_chunk(text, max_chunk_size=4, overlap=1):
        stripped = text.strip()
        if len(stripped) == 0:
            l = []
            return l
        sentences = re.split(r"(?<=[.!?])\s+",stripped)
        if len(sentences) == 1 and not sentences[0].endswith((".", "!", "?")):
            sentences = [sentences[0]]
        start = 0
        
        chunks = []
        while start < len(sentences):
            chunk_sentences = sentences[start : start + max_chunk_size]

            if chunks and len(chunk_sentences) <= overlap:
                break
            start += max_chunk_size - overlap
            newChunk = []
            for chunk in chunk_sentences:
                chunk = chunk.strip()
                newChunk.append(chunk)
            app = " ".join(newChunk)
            if len(app) > 0:
                chunks.append(" ".join(newChunk))
        return chunks

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.document_map = {}
        all_chunks = []
        chunk_metadata = []
        self.documents = documents

        for movie_idx, document in enumerate(documents):
            text = document["description"]
            self.document_map[document["id"]] = document
            if not text.strip():
                continue

            document_chunks = semantic_chunk(
                text,
                max_chunk_size=4,
                overlap=1,
                )

            for chunk_idx, chunk in enumerate(document_chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    "movie_idx": movie_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": len(document_chunks),
                    })

        self.chunk_embeddings = self.model.encode(
        all_chunks,
        show_progress_bar=True)
        self.chunk_metadata = chunk_metadata
        os.makedirs("cache", exist_ok=True)

        np.save("cache/chunk_embeddings.npy", self.chunk_embeddings)

        with open("cache/chunk_metadata.json", "w") as f:
            json.dump(
                {
                    "chunks":chunk_metadata,
                    "total_chunks": len(all_chunks),
                },
                f,
                indent=2,
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}

        for document in documents:
            self.document_map[document["id"]] = document
        embeddings_exist = os.path.exists("cache/chunk_embeddings.npy")
        metadata_exists = os.path.exists("cache/chunk_metadata.json")


        if metadata_exists and embeddings_exist:
            self.chunk_embeddings = np.load("cache/chunk_embeddings.npy")

            with open("cache/chunk_metadata.json", "r") as f:
                data = json.load(f)
                self.chunk_metadata = data['chunks']

            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        embeddings = self.generate_embedding(query)
        chunk_score = []
        for i in range(len(self.chunk_embeddings)):
            cosine = cosine_similarity(self.chunk_embeddings[i],embeddings)
            meta = self.chunk_metadata[i]
            chunk_score.append({
                "chunk_idx": meta["chunk_idx"],
                "movie_idx": meta["movie_idx"],
                "score": cosine,
            })

        best = {}
        for entry in chunk_score:
            key = entry["movie_idx"]
            if key not in best or entry["score"] > best[key]["score"]:
                best[key] = entry
        ranked = sorted(best.values(), key=lambda e: e["score"], reverse=True)
        results = []
        for rank in ranked[:limit]:

            doc = self.documents[rank["movie_idx"]]
            results.append({
                "id": doc['id'],
                "title": doc['title'],
                "document": doc["description"][:100],
                "score": round(rank['score'],3)

            })
        return results

