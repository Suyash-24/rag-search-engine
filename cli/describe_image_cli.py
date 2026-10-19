import mimetypes
import argparse
import sys
from dotenv import load_dotenv
from openai import OpenAI
import base64
import os

def debug(msg):
    print(f"[DEBUG] {msg}", file=sys.stderr)
def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    
    parser.add_argument("--image",type=str,help="path of the image")
    parser.add_argument("--query",type=str,help="a text query to rewrite based on the image")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"
    with open(args.image , "rb") as f:
        data = f.read()
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    

    system_prompt = """
        Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
    - Synthesize visual and textual information
    - Focus on movie-specific details (actors, scenes, style, etc.)
    - Return only the rewritten query, without any additional commentary
    """
            
            
    data_url = f"data:{mime};base64,{base64.b64encode(data).decode()}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": system_prompt.strip()},
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": args.query.strip()},
            ],
        }
    ]
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=messages,
    )
    content = response.choices[0].message.content
    print(f"Rewritten query: {content.strip()}")
    if response.usage is not None:
        print(f"Total tokens:    {response.usage.total_tokens}")


if __name__ == "__main__":
    main()