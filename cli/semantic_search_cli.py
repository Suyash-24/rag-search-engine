
import re
import json
import argparse
from lib.semantic_search import (
    verify_model,
    embed,
    verify_embeddings,
    embed_query_text,
    SemanticSearch,
    ChunkedSemanticSearch,
    semantic_chunk
)
def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    verify = subparsers.add_parser("verify", help="verify the model")
    embed_text = subparsers.add_parser("embed_text", help="embeds the given text")
    embed_text.add_argument("text",type=str,help="text to embed")
    verify_embed = subparsers.add_parser("verify_embeddings", help="check the embeddings")
    embed_quer = subparsers.add_parser("embed_query", help="turn query into embed")
    embed_quer.add_argument("text",type=str,help="text to embed")
    search = subparsers.add_parser("search", help="semantic search")
    search.add_argument("query",type=str,help="query string")
    search.add_argument("--limit",type=int,help="query string",default=5)
    chunk = subparsers.add_parser("chunk", help="chunks the text")
    chunk.add_argument("text",type=str,help="text to chunk")
    chunk.add_argument("--chunk-size",type=int,help="query string",default=200)
    chunk.add_argument("--overlap",type=int,help="overlap string",default=0)
    semantic_chun = subparsers.add_parser("semantic_chunk", help="semantic chunk the text")
    semantic_chun.add_argument("text",type=str,help="text to semantic chunk")
    semantic_chun.add_argument("--max-chunk-size",type=int,help="max size to break statement",default=4)
    semantic_chun.add_argument("--overlap",type=int,help="overlap string",default=0)
    embed_chunks = subparsers.add_parser("embed_chunks", help="create chunk embeddings")
    search_chunked = subparsers.add_parser("search_chunked", help="search the chunked")
    search_chunked.add_argument("query",type=str,help="query string")
    search_chunked.add_argument("--limit",type=int,help="query string",default=5)

    args = parser.parse_args()


    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed(args.text)


        case "verify_embeddings":
            verify_embeddings()

        case "embed_query":
            embed_query_text(args.text)
        case "search":
            ss = SemanticSearch()
            with open('data/movies.json','r') as f:
                movies = json.load(f)['movies']
            ss.load_or_create_embeddings(movies)
            results = ss.search(args.query , args.limit)
            n = 1
            for result in results:
                print(f"{n}. {result['title']} (score: {result['score']:.4f})\n    {result['description'][:100]}...")
                n += 1
        case "chunk":
            texts = args.text.split(" ") 
            n = 1
            print(f"Chunking {len(args.text)} characters")
            start = 0
            
            while start < len(texts):
                next_start = start + args.chunk_size - args.overlap
                print(f'{n}. {" ".join(texts[start:start+args.chunk_size])}')
                n += 1
                if start + args.chunk_size >= len(texts):
                    break
                
                start = next_start
        case "semantic_chunk":
            
            movies = semantic_chunk(args.text,args.max_chunk_size,args.overlap)
            i = 1
            print(f"Semantically chunking {len(args.text)} characters")
            for movie in movies:
                print(f"{i}. {movie}")
                i += 1

                
        case "embed_chunks":
            with open('data/movies.json','r') as f:
                movies = json.load(f)['movies']

            css = ChunkedSemanticSearch()
            embeddings = css.load_or_create_chunk_embeddings(movies)
            print(f"Generated {len(embeddings)} chunked embeddings")

        case "search_chunked":
            with open('data/movies.json','r') as f:
                movies = json.load(f)['movies']
            css = ChunkedSemanticSearch()
            embeddings = css.load_or_create_chunk_embeddings(movies)
            results = css.search_chunks(args.query,args.limit)
            i = 1 
            for result in results:
                print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {result['document']}...")
                i += 1

        case _:
            parser.print_help()
        




if __name__ == "__main__":
    main()
