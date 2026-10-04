
from collections import Counter
import os 
import pickle
import json 
import string
from nltk.stem import PorterStemmer
from math import log


BM25_K1 = 1.5
BM25_B = 0.75

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.doc_lengths = {}
        self.doc_lengths_path = os.path.join("cache", "doc_lengths.pkl")
    def __add_document(self, doc_id, text):
        tokens = tokenization(text)
        for token in tokens:
            doc_ids = self.index.setdefault(token, set())
            doc_ids.add(doc_id)
            count = self.term_frequencies.setdefault(doc_id, Counter())
            count[token] += 1
        self.doc_lengths[doc_id] = len(tokens)

    def get_documents(self, term):
        doc_ids = self.index.get(term, set())
        return sorted(doc_ids)

    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            text = movie["title"] + " " + movie["description"]
            self.__add_document(doc_id, text)
    
    def save(self):
        os.makedirs("cache", exist_ok=True)
        with open("cache/index.pkl", "wb") as file:
            pickle.dump(self.index, file)
        with open("cache/docmap.pkl", "wb") as file:
            pickle.dump(self.docmap, file)
        with open("cache/term_frequencies.pkl", "wb") as file:
            pickle.dump(self.term_frequencies, file)
        with open("cache/doc_lengths.pkl", "wb") as file:
            pickle.dump(self.doc_lengths, file)
    
    def load(self):
        
        with open("cache/index.pkl", "rb") as file:
            indCache = pickle.load(file)
            self.index = indCache
        with open("cache/docmap.pkl", "rb") as file:
            mapCache = pickle.load(file)
            self.docmap = mapCache
        with open("cache/term_frequencies.pkl", "rb") as file:
            termCache = pickle.load(file)
            self.term_frequencies = termCache
        with open("cache/doc_lengths.pkl", "rb") as file:
            docLength = pickle.load(file)
            self.doc_lengths = docLength
    
    def get_tf(self, doc_id, term):
        doc_counts = self.term_frequencies.get(doc_id, Counter())
        return doc_counts[term]

    def get_bm25_idf(self, term: str) -> float:
        N = len(self.docmap)
        df = len(self.get_documents(term))
        bm25 = log((N - df + 0.5) / (df + 0.5) + 1)
        return bm25
        
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1,b=BM25_B):
        length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / length)
        tf = self.get_tf(doc_id,term)
        bm_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return bm_tf

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0
        else:
            doc = []
            for length in self.doc_lengths:
                doc.append(self.doc_lengths[length])
            avg = sum(doc) / len(doc)
            return avg

    def bm25(self, doc_id, term):
        bm25 = self.get_bm25_idf(term) * self.get_bm25_tf(doc_id,term)
        return bm25

    def bm25_search(self, query, limit):
        tokens = tokenization(query)
        scores = {}
        for doc_id in self.docmap:
            total = 0
            for term in tokens:
                total += self.bm25(doc_id, term)
                scores[doc_id] = total
        sorted_items = sorted(scores.items(),key=lambda pair: pair[1],reverse=True)
        return sorted_items[:limit]







def tokenization(words):
    stemmer = PorterStemmer()
    words = words.translate(str.maketrans("", "", string.punctuation)).lower()
    words = words.split()
    with open('data/stopwords.txt','r') as file:
        wor = file.read().splitlines()
    stopwords = []
    for word in wor:
        w = word.translate(str.maketrans("", "", string.punctuation)).lower()
        stopwords.append(w)
            
            
            
           

    swords = []
    for token in words:
        if token not in stopwords:
            swords.append(stemmer.stem(token))
    return swords

def load_movies():
    with open("data/movies.json", "r") as file:
        data = json.load(file)
    return data["movies"]

def tokenize(term):
    token = tokenization(term)
    if len(token) == 1:
        return token[0] 
    else:
        raise Exception("Multiple tokens")


def bm25_idf_command(term):
    idx = InvertedIndex()
    idx.load()
    token = tokenize(term)
    bm = idx.get_bm25_idf(token)
    return bm

def bm25_tf_command(doc_id,term,k1=BM25_K1,b=BM25_B):
    idx = InvertedIndex()
    idx.load()
    token =  tokenize(term)
    bm_tf = idx.get_bm25_tf(doc_id,token,k1,b)
    return bm_tf