from PIL import Image
from sentence_transformers import SentenceTransformer
from index import load_movies
from lib.semantic_search import cosine_similarity

class MultimodalSearch:
    def __init__(self,documents, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
        self.text_embeddings = self.model.encode(self.texts,show_progress_bar=True)

    def embed_image(self,image):
        with Image.open(image) as im:
            encodings = self.model.encode([im])
            return encodings[0]
    def search_with_image(self,path):
        image_embedding = self.embed_image(path)
        result = []
        for doc, emb in zip(self.documents, self.text_embeddings):
            score = cosine_similarity(emb,image_embedding)            # one float
            result.append({
                "id": doc["id"],
                "title": doc["title"],
                "description": doc["description"],
                "similarity": score,
            })
        
        result = sorted(result, key=lambda p: p["similarity"], reverse=True)
        return result[:5]
def verify_image_embedding(path):
    movies = load_movies()
    searcher = MultimodalSearch(movies)
    embedding = searcher.embed_image(path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(path):
    movies = load_movies()
    searcher = MultimodalSearch(movies)
    return searcher.search_with_image(path)

