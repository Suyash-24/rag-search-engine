import argparse

from lib.hybrid_search import HybridSearch
from index import load_movies

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
            movies = load_movies()
            hs = HybridSearch(movies)
            dictionary = hs.rrf_search(args.query,args.k,args.limit)
            dictionary = dictionary[:args.limit]
            n = 1
            for d in dictionary:
                print(f"{n}. {d['document']['title']}\nRRF Score: {d['rrf']:.3f}\nBM25 Rank: {d['bm25_rank']}, Semantic Rank: {d['semantic_rank']}\n{d['document']['document']}\n")
                n += 1



        case _:
            parser.print_help()


if __name__ == "__main__":
    main()