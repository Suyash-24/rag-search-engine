import os, sys
sys.path.insert(0, '.')
sys.path.insert(0, 'lib')
from dotenv import load_dotenv
from index import load_movies
from lib.hybrid_search import HybridSearch
from lib.query_enhancement import rerank

load_dotenv('../.env')
movies = load_movies()
hs = HybridSearch(movies)

query = 'family movie about bears in the woods'
limit = 3
candidate_limit = limit * 5  # 15 candidates

print('--- Getting RRF candidates ---')
candidates = hs.rrf_search(query, 60, candidate_limit)
print(f'Got {len(candidates)} candidates:')
for i, c in enumerate(candidates, 1):
    print(f'  {i}. {c["document"]["title"]}')

print()
print('--- Re-ranking with LLM ---')
for c in candidates:
    score = rerank(query, c['document'])
    c['rank'] = score
    print(f'  {c["document"]["title"]}: {score}')

candidates.sort(key=lambda x: x['rank'], reverse=True)
top3 = candidates[:limit]

print()
print('--- Top 3 ---')
for i, c in enumerate(top3, 1):
    print(f'  {i}. {c["document"]["title"]} (score={c["rank"]})')

titles = [c['document']['title'] for c in top3]
print()
print('PASS' if any('Berenstain' in t for t in titles) else 'FAIL - Berenstain Bears not in top 3')
