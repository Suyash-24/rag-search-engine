import argparse
import json
from lib.hybrid_search import HybridSearch
from index import load_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit
    
    # run evaluation logic here
    with open("data/golden_dataset.json", "r") as f:
        data = json.load(f)
    movies = load_movies()
    hs = HybridSearch(movies)
    test_cases = data["test_cases"] 

    print(f"k={limit}")
    print()
    for case in test_cases:
        query = case["query"]  
        results = hs.rrf_search(query, 60, limit) 
        results = results[:limit]
        retrieved_titles = []
        relevant_titles = case["relevant_docs"]
        for r in results:
            retrieved_titles.append(r['document']['title'])
        relevant_count = 0
        for title in retrieved_titles:
            if title in relevant_titles:
                relevant_count += 1
        precision = relevant_count / len(retrieved_titles)
        recall = relevant_count / len(relevant_titles)
        f1 = 2 * (precision * recall) / (precision + recall)
        retrieved = ", ".join(retrieved_titles)
        relevant = ", ".join(relevant_titles)
        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {retrieved}")
        print(f"  - Relevant: {relevant}")
        print()


if __name__ == "__main__":
    main()