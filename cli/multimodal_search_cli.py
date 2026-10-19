
import argparse
from lib.multimodal_search import ( MultimodalSearch , verify_image_embedding , image_search_command)
    
def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    verify = subparsers.add_parser("verify_image_embedding", help="verify the image")

    verify.add_argument("path",type=str,help="path to image")
    image_search = subparsers.add_parser("image_search", help="search the image")

    image_search.add_argument("path",type=str,help="path to image")
    
    args = parser.parse_args()


    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.path)
        case "image_search":
            results = image_search_command(args.path)
            n =1
            for result in results:
                print(f"{n}. {result['title']} (similarity: {result['similarity']:.3f})\n   {result['description'][:100]}...")
                print()
                n += 1
        
        case _:
            parser.print_help()
        




if __name__ == "__main__":
    main()
