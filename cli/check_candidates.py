import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'lib')
from index import load_movies
from lib.hybrid_search import HybridSearch

movies = load_movies()
hs = HybridSearch(movies)

query = 'family movie about bears in the woods'
candidates = hs.rrf_search(query, 60, 15)
print('Top 15 RRF candidates:')
for i, c in enumerate(candidates, 1):
    print(f'  {i}. {c["document"]["title"]}')

titles = [c["document"]["title"] for c in candidates]
print()
print('Berenstain in candidates?', any('Berenstain' in t for t in titles))
