import argparse
import sys
from lib.hybrid_search import HybridSearch
from index import load_movies
from lib.query_enhancement import spell_correct, spell_rewrite, spell_expansion, rerank, rerank_batch , evaluate_query
from sentence_transformers import CrossEncoder
import json
import re


def debug(msg):
    print(f"[DEBUG] {msg}", file=sys.stderr)
def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    normalize = subparsers.add_parser("normalize", help="normalize the list of scores")
    normalize.add_argument("list",type=float,nargs='*',help="list of scores")
    weighted_search_parser = subparsers.add_parser("weighted-search", help="find the weigthed search")
    weighted_search_parser.add_argument("query", type=str, help="query to weight")
    weighted_search_parser.add_argument("--alpha",type=float,default=0.5,help="alpha for weight")
    weighted_search_parser.add_argument("--limit", type=int, default=5,help="limit for return")
    rrf_search = subparsers.add_parser("rrf-search", help="rrf for searching")
    rrf_search.add_argument("query", type=str, help="query for rrf")
    rrf_search.add_argument("-k",type=int,default=60,help="a constant")
    rrf_search.add_argument("--limit", type=int, default=5,help="limit for return")
    rrf_search.add_argument(
        "--enhance",
        type=str,
        choices=["spell","rewrite","expand"],
        help="Query enhancement method",
    )
    rrf_search.add_argument("--rerank-method",type=str,choices=["individual","batch","cross_encoder"],help="method for re ranking")
    rrf_search.add_argument("--evaluate", action="store_true", help="whether to evaluate or not")
    args = parser.parse_args()

    match args.command:
        case "normalize":
            if len(args.list) == 0:
                return
            scores = []
            mini = min(args.list)
            maxi = max(args.list)
            
            if min(args.list) == max(args.list):
                newscores = []
                for score in args.list:
                    newscores.append(1.0)
                for score in newscores:
                    print(f"* {score:.4f}")

            else:
                for i in range(len(args.list)):
                    scores.append((args.list[i] - mini) / (maxi - mini))
                for score in scores:
                    print(f"* {score:.4f}")
            
        case "weighted-search":
            movies = load_movies()
            hs = HybridSearch(movies)
            dictionary = hs.weighted_search(args.query,args.alpha,args.limit)
            n = 1
            for d in dictionary:
                print(f"{n}. {d['document']['title']}\nHybrid Score: {d['hybrid_score']:.3f}\nBM25: {d['bm25_score']:.3f}, Semantic: {d['semantic_score']:.3f}\n{d['document']['document']}")
                n += 1

        case "rrf-search":
            if args.rerank_method:
                limit = args.limit * 5
                if not args.enhance:
                    debug(f"Original query: '{args.query}'")
                    debug(f"No enhancement applied")
                    movies = load_movies()
                    hs = HybridSearch(movies)
                    dictionary = hs.rrf_search(args.query, args.k, limit)
                    debug(f"RRF results ({len(dictionary)} candidates):")
                    for i, d in enumerate(dictionary[:5], 1):
                        debug(f"  {i}. {d['document']['title']} (RRF={d['rrf']:.3f})")
                    print(f"Re-ranking top {args.limit} results using {args.rerank_method} method...")
                    print(f"Reciprocal Rank Fusion Results for '{args.query}' (k={args.k}):")
                    if args.rerank_method == "individual":
                        for search in dictionary:
                            search["rank"] = rerank(args.query, search["document"])
                        dictionary.sort(key=lambda search: search['rank'], reverse=True)
                    elif args.rerank_method == "batch":
                        batch_result = rerank_batch(args.query, dictionary)
                        for search in dictionary:
                            search["rank"] = batch_result.index(search["document"]["id"]) + 1
                        dictionary.sort(key=lambda search: search['rank'])
                    else:
                        pairs = []
                        cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
                        for doc in dictionary:
                            pairs.append([args.query, f"{doc['document']['title']} - {doc['document']['document']}"])
                        scores = cross_encoder.predict(pairs)
                        for i in range(len(scores)):
                            dictionary[i]["rank"] = scores[i]
                        dictionary.sort(key=lambda search: search['rank'], reverse=True)

                    dictionary = dictionary[:args.limit]
                    debug(f"Final results after re-ranking ({args.rerank_method}):")
                    for i, d in enumerate(dictionary, 1):
                        debug(f"  {i}. {d['document']['title']} (rank={d['rank']:.3f})")
                
                    n = 1
                    for d in dictionary:
                        if args.rerank_method == "individual":
                            print(f"{n}. {d['document']['title']}\nRe-rank Score : {d['rank']:.3f}/10 \nRRF Score: {d['rrf']:.3f}\nBM25 Rank: {d['bm25_rank']}, Semantic Rank: {d['semantic_rank']}\n{d['document']['document'][:100]}...")
                            n += 1
                        elif args.rerank_method == "batch":

                            print(f"{n}. {d['document']['title']}\nRe-rank Rank: {d['rank']} \nRRF Score: {d['rrf']:.3f}\nBM25 Rank: {d['bm25_rank']}, Semantic Rank: {d['semantic_rank']}\n{d['document']['document'][:100]}...")
                            n += 1
                        else:
                            print(f"{n}. {d['document']['title']}\nCross Encoder Score: {d['rank']:.3f} \nRRF Score: {d['rrf']:.3f}\nBM25 Rank: {d['bm25_rank']}, Semantic Rank: {d['semantic_rank']}\n{d['document']['document'][:100]}...")
                            n += 1

                elif args.enhance == "spell":
                    debug(f"Original query: '{args.query}'")
                    enhanced = spell_correct(args.query)
                    debug(f"Enhanced query (spell): '{enhanced}'")
                    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{enhanced}'\n")
                    movies = load_movies()
                    hs = HybridSearch(movies)
                    dictionary = hs.rrf_search(enhanced, args.k, limit)
                    debug(f"RRF results ({len(dictionary)} candidates), top 5: {[d['document']['title'] for d in dictionary[:5]]}")
                    print(f"Re-ranking top {args.limit} results using individual method...")
                    print(f"Reciprocal Rank Fusion Results for '{args.query}' (k={args.k}):")
                    for search in dictionary:
                        search["rank"] = rerank(args.query,search["document"])
                    dictionary.sort(key=lambda search: search['rank'], reverse=True)
                    dictionary = dictionary[:args.limit]
                    n = 1
                    for d in dictionary:
                        print(f"{n}. {d['document']['title']}\nRe-rank Score : {d['rank']:.3f}/10 \nRRF Score: {d['rrf']:.3f}\nBM25 Rank: {d['bm25_rank']}, Semantic Rank: {d['semantic_rank']}\n{d['document']['document'][:100]}...\n")
                        n += 1
                elif args.enhance == "rewrite":
                    debug(f"Original query: '{args.query}'")
                    enhanced = spell_rewrite(args.query)
                    debug(f"Enhanced query (rewrite): '{enhanced}'")
                    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{enhanced}'\n")
                    movies = load_movies()
                    hs = HybridSearch(movies)
                    dictionary = hs.rrf_search(enhanced, args.k, limit)
                    debug(f"RRF results ({len(dictionary)} candidates), top 5: {[d['document']['title'] for d in dictionary[:5]]}")
                    print(f"Re-ranking top {args.limit} results using individual method...")
                    print(f"Reciprocal Rank Fusion Results for '{args.query}' (k={args.k}):")
                    for search in dictionary:
                        search["rank"] = rerank(args.query,search["document"])
                    dictionary.sort(key=lambda search: search['rank'], reverse=True)
                    dictionary = dictionary[:args.limit]
                    n = 1
                    for d in dictionary:
                        print(f"{n}. {d['document']['title']}\nRe-rank Score : {d['rank']:.3f}/10 \nRRF Score: {d['rrf']:.3f}\nBM25 Rank: {d['bm25_rank']}, Semantic Rank: {d['semantic_rank']}\n{d['document']['document'][:100]}...")
                        n += 1
                else:
                    debug(f"Original query: '{args.query}'")
                    enhanced = spell_expansion(args.query)
                    debug(f"Enhanced query (expand): '{enhanced}'")
                    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{enhanced}'\n")
                    movies = load_movies()
                    hs = HybridSearch(movies)
                    dictionary = hs.rrf_search(enhanced, args.k, limit)
                    debug(f"RRF results ({len(dictionary)} candidates), top 5: {[d['document']['title'] for d in dictionary[:5]]}")
                    print(f"Re-ranking top {args.limit} results using individual method...")
                    print(f"Reciprocal Rank Fusion Results for '{args.query}' (k={args.k}):")
                    for search in dictionary:
                        search["rank"] = rerank(args.query,search["document"])
                    dictionary.sort(key=lambda search: search['rank'], reverse=True)
                    dictionary = dictionary[:args.limit]
                    n = 1
                    for d in dictionary:
                        print(f"{n}. {d['document']['title']}\nRe-rank Score : {d['rank']:.3f}/10\nRRF Score: {d['rrf']:.3f}\nBM25 Rank: {d['bm25_rank']}, Semantic Rank: {d['semantic_rank']}\n{d['document']['document'][:100]}...")
                        n += 1
            else:
                if not args.enhance:
                    movies = load_movies()
                    hs = HybridSearch(movies)
                    dictionary = hs.rrf_search(args.query,args.k,args.limit)
                    dictionary = dictionary[:args.limit]
                    n = 1
                    for d in dictionary:
                        print(f"{n}. {d['document']['title']}\nRRF Score: {d['rrf']:.3f}\nBM25 Rank: {d['bm25_rank']}, Semantic Rank: {d['semantic_rank']}\n{d['document']['document']}\n")
                        n += 1
                elif args.enhance == "spell":
                    enhanced = spell_correct(args.query)
                    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{enhanced}'\n")
                    movies = load_movies()
                    hs = HybridSearch(movies)
                    dictionary = hs.rrf_search(enhanced,args.k,args.limit)
                    dictionary = dictionary[:args.limit]
                    n = 1
                    for d in dictionary:
                        print(f"{n}. {d['document']['title']}\nRRF Score: {d['rrf']:.3f}\nBM25 Rank: {d['bm25_rank']}, Semantic Rank: {d['semantic_rank']}\n{d['document']['document']}\n")
                        n += 1
                elif args.enhance == "rewrite":
                    enhanced = spell_rewrite(args.query)
                    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{enhanced}'\n")
                    movies = load_movies()
                    hs = HybridSearch(movies)
                    dictionary = hs.rrf_search(enhanced,args.k,args.limit)
                    dictionary = dictionary[:args.limit]
                    n = 1
                    for d in dictionary:
                        print(f"{n}. {d['document']['title']}\nRRF Score: {d['rrf']:.3f}\nBM25 Rank: {d['bm25_rank']}, Semantic Rank: {d['semantic_rank']}\n{d['document']['document']}\n")
                        n += 1
                else:
                    enhanced = spell_expansion(args.query)
                    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{enhanced}'\n")
                    movies = load_movies()
                    hs = HybridSearch(movies)
                    dictionary = hs.rrf_search(enhanced,args.k,args.limit)
                    dictionary = dictionary[:args.limit]
                    n = 1
                    for d in dictionary:
                        print(f"{n}. {d['document']['title']}\nRRF Score: {d['rrf']:.3f}\nBM25 Rank: {d['bm25_rank']}, Semantic Rank: {d['semantic_rank']}\n{d['document']['document']}\n")
                        n += 1
            if args.evaluate == True:
                formatted_strings = []
                for d in dictionary:
                    formatted_strings.append(d['document']['document'])
                result = evaluate_query(args.query, formatted_strings)
                
                for i, (d, r) in enumerate(zip(dictionary, result), 1):
                    print(f"{i}. {d['document']['title']}: {r}/3")
                

            
                




        case _:
            parser.print_help()


if __name__ == "__main__":
    main()