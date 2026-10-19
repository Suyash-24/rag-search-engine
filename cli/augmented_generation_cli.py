import argparse
from lib.query_enhancement import rag_gen ,summarry ,citation ,ques
from index import load_movies
from lib.hybrid_search import HybridSearch

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize = subparsers.add_parser(
        "summarize", help="summarize the answer"
    )
    summarize.add_argument("query", type=str, help="query to summarize")
    summarize.add_argument("--limit", type=int,default=5, help="query to summarize")
    citations = subparsers.add_parser(
        "citations", help="citations for the query"
    )
    citations.add_argument("query", type=str, help="query to citations")
    citations.add_argument("--limit", type=int,default=5, help="number of citations")
    question = subparsers.add_parser(
        "question", help="question to ask"
    )
    question.add_argument("query", type=str, help="query to citations")
    question.add_argument("--limit", type=int,default=5, help="number of question")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            # do RAG stuff here
            movies = load_movies()
            hs = HybridSearch(movies)
            li = hs.rrf_search(query,60,limit=5)
            response = rag_gen(query,li)
            print("Search Results:")
            for l in li:
                print(f"- {l["document"]['title']}")
            print()
            print("RAG Response:")
            print(response)
        
        case "summarize":
            query = args.query
            limit = args.limit
            movies = load_movies()
            hs = HybridSearch(movies)
            li = hs.rrf_search(query,60,limit)
            response = summarry(query,li)
            print("Search Results:")
            for l in li:
                print(f"  - {l["document"]['title']}")
            print()
            print("LLM Summary:")
            print(response)

        case "citations":
            query = args.query
            limit = args.limit
            movies = load_movies()
            hs = HybridSearch(movies)
            li = hs.rrf_search(query,60,limit)
            response = citation(query,li)
            print("Search Results:")
            for l in li:
                print(f"  - {l["document"]['title']}")
            print()
            print("LLM Answer:")
            print(response)
        

        case "question":
            query = args.query
            limit = args.limit
            movies = load_movies()
            hs = HybridSearch(movies)
            li = hs.rrf_search(query,60,limit)
            response = ques(query,li)
            print("Search Results:")
            for l in li:
                print(f"  - {l["document"]['title']}")
            print()
            print("Answer:")
            print(response)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()