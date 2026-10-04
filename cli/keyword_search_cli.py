from index import InvertedIndex, tokenization , tokenize , bm25_idf_command ,BM25_K1 ,bm25_tf_command , BM25_B
import argparse
import json
import string
from nltk.stem import PorterStemmer
import math



def main() -> None:

    stemmer = PorterStemmer()
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")
    subparsers.add_parser("build", help="Build the inverted index")
    tokenizer = subparsers.add_parser("tf", help="tokenize the term")
    tokenizer.add_argument("documentId",type=int,help="Id of the document")
    tokenizer.add_argument("term",type=str,help="The term to tokenize")
    idf = subparsers.add_parser("idf", help="calculates the ibl of token")
    idf.add_argument("term",type=str,help="the token for calculating idf")
    tfidf = subparsers.add_parser("tfidf", help="calculate the tf and df score")
    tfidf.add_argument("docId",type=int,help="Id of the document")
    tfidf.add_argument("term",type=str,help="The term to tokenize")
    bm25_idf_parser = subparsers.add_parser(
    "bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")
    bm25_tf_parser = subparsers.add_parser(
    "bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
    "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter")
    bm25search_parser = subparsers.add_parser(
    "bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, default=5,help="limit for return")
    args = parser.parse_args()
    

    match args.command:
        case "build":
            idx = InvertedIndex()
            idx.build()
            idx.save()
        case "search":
            # print the search query here
            results = []
            seen_ids = set()
            search_lower = args.query.lower()
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError:
                print("some error message")
                return  
            print(f'Searching for: {args.query}')
            swords = tokenization(search_lower)
            for word in swords:
                doc_ids = idx.get_documents(word)
                for doc_id in doc_ids:
                    a = idx.docmap[doc_id]

                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        results.append(a)
                    if len(results) == 5:
                        break
                if len(results) == 5:
                        break        
                
                
            n = 1
            for result in results:
                print(f"{n}. {result['title']}")
                n += 1

        case "tf":
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError:
                print("some error message")
                return  
            token = tokenize(args.term)
            c= idx.get_tf(args.documentId,token)
            print(c)

        case "idf":
            idx = InvertedIndex()
            idx.load()
            token = tokenize(args.term)
            doc_ids = idx.get_documents(token)
            
            idf  = math.log((len(idx.docmap) + 1) / (len(doc_ids) + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            idx = InvertedIndex()
            idx.load()
            token = tokenize(args.term)
            doc_ids = idx.get_documents(token)
            
            idf  = math.log((len(idx.docmap) + 1) / (len(doc_ids) + 1))
            token = tokenize(args.term)
            c= idx.get_tf(args.docId,token)
            tfidf = c * idf 
            print(f"TF-IDF score of '{args.term}' in document '{args.docId}': {tfidf:.2f}")
        
        case  "bm25idf":
            bm = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm:.2f}")

        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id,args.term,args.k1,args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case "bm25search":
            idx = InvertedIndex()
            idx.load()
            results = idx.bm25_search(args.query,args.limit)
            n =1
            for result in results:
                movie = idx.docmap[result[0]]
                title = movie['title']
                print(f'{n}. ({result[0]}) {title} - Score: {result[1]:.2f}')
                n += 1
        case _:
            parser.print_help()







if __name__ == "__main__":
    main()