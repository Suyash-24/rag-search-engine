# cli/lib/query_enhancement.py
import os
import re
from dotenv import load_dotenv
from openai import OpenAI
import json


def spell_correct(query):
    instruction = f"""Fix any spelling errors in the user-provided movie search query below.
    Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
    Preserve punctuation and capitalization unless a change is required for a typo fix.
    If there are no spelling errors, or if you're unsure, output the original query unchanged.
    Output only the final query text, nothing else.
    User query: "{query}"
    """
    a = enhance_query(instruction)
    a = a.strip()
    if a.endswith('"'):
        a = a[:-1]
    if a.startswith('"'):
        a = a[1:]
    return a


def rerank(query, doc):
    instruction = f"""Rate how well this movie matches the search query.

    Query: "{query}"
    Movie: {doc.get("title", "")} - {doc.get("document", "")}

    Consider:
    - Direct relevance to query
    - User intent (what they're looking for)
    - Content appropriateness

    Rate 0-10 (10 = perfect match).
    Output ONLY the number in your response, no other text or explanation.

    Score:"""
    for attempt in range(3):
        try:
            a = enhance_query(instruction)
        except Exception:
            break
        if a:
            match = re.search(r'\b(\d+(?:\.\d+)?)\b', a)
            if match:
                score = float(match.group(1))
                if 0 <= score <= 10:
                    return score
    return 5.0


def rerank_batch(query, documents):
    movies_text = "\n".join(
        f"ID {doc['document']['id']}: {doc['document']['title']} - {doc['document']['document'][:80]}"
        for doc in documents
    )
    instruction = f"""Rank the movies listed below by relevance to the following search query.

    Query: "{query}"

    Movies:
    {movies_text}

    Return the movie IDs in order of relevance, best match first.

    Your response must be a raw JSON array of integers.
    Do not wrap the JSON in Markdown. Do not use a ```json code block.
    Do not include any explanatory text.

    For example:
    [75, 12, 34, 2, 1]

    Ranking:"""
    for attempt in range(3):
        try:
            a = enhance_query(instruction)
        except Exception:
            break
        if a:
            try:
                # strip markdown code fences if present
                clean = re.sub(r"```[a-z]*\n?", "", a).strip()
                ids = json.loads(clean)
                if isinstance(ids, list):
                    return ids
            except (json.JSONDecodeError, ValueError):
                continue
    # fallback: return IDs in original RRF order
    return [doc["document"]["id"] for doc in documents]



def spell_rewrite(query):
    instruction = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

        Consider:
        - Common movie knowledge (famous actors, popular films)
        - Genre conventions (horror = scary, animation = cartoon)
        - Keep the rewritten query concise (under 10 words)
        - It should be a Google-style search query, specific enough to yield relevant results
        - Don't use boolean logic

        Examples:
        - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
        - "movie about bear in london with marmalade" -> "Paddington London marmalade"
        - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

        If you cannot improve the query, output the original unchanged.
        Output only the rewritten query text, nothing else.

        User query: "{query}"
        """
    
    a  = enhance_query(instruction)
    a = a.strip()
    if a.endswith('"'):
        a = a[:-1]
    if a.startswith('"'):
        a = a[1:]
    return a 

def spell_expansion(query):
    instruction = f"""Expand the user-provided movie search query below with related terms.

        Add synonyms and related concepts that might appear in movie descriptions.
        Keep expansions relevant and focused.
        Output only the additional terms; they will be appended to the original query.

        Examples:
        - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
        - "action movie with bear" -> "action thriller bear chase fight adventure"
        - "comedy with bear" -> "comedy funny bear humor lighthearted"

        User query: "{query}"
        """
       
    
    a  = enhance_query(instruction)
    a = a.strip()
    if a.endswith('"'):
        a = a[:-1]
    if a.startswith('"'):
        a = a[1:]
    return a 


def evaluate_query(query,formatted_results):
    instruction = f"""Rate how relevant each result is to this query on a 0-3 scale:

    Query: "{query}"

    Results:
    {chr(10).join(formatted_results)}

    Scale:
    - 3: Highly relevant
    - 2: Relevant
    - 1: Marginally relevant
    - 0: Not relevant

    Do NOT give any numbers other than 0, 1, 2, or 3.

    Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

    [2, 0, 3, 2, 0, 1]"""
       
    for attempt in range(3):
        try:
            a = enhance_query(instruction)
        except Exception:
            break
        if a:
            try:
                # strip markdown code fences if present
                clean = re.sub(r"```[a-z]*\n?", "", a).strip()
                ids = json.loads(clean)
                if isinstance(ids, list):
                    return ids
            except (json.JSONDecodeError, ValueError):
                continue
    default = [0] * len(formatted_results)
    return [0] * len(formatted_results)

def rag_gen(query,docs):
    instruction = prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
    Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
    Provide a comprehensive answer that addresses the user's query.

    Query: {query}

    Documents:
    {docs}

    Answer:"""
       
    
    a = enhance_query(instruction)
    a = a.strip()
    if a.endswith('"'):
        a = a[:-1]
    if a.startswith('"'):
        a = a[1:]
    return a 
       
def summarry(query,results):
    instruction = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

    The goal is to provide comprehensive information so that users know what their options are.
    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

    This should be tailored to Webflyx users. Webflyx is a movie streaming service.

    Query: {query}

    Search results:
    {results}

    Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
        
    
    a = enhance_query(instruction)
    a = a.strip()
    if a.endswith('"'):
        a = a[:-1]
    if a.startswith('"'):
        a = a[1:]
    return a


def citation(query,documents):
    prompt = f"""Answer the query below and give information based on the provided documents.

    The answer should be tailored to users of Webflyx, a movie streaming service.
    If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

    Query: {query}

    Documents:
    {documents}

    Instructions:
    - Provide a comprehensive answer that addresses the query
    - Cite sources in the format [1], [2], etc. when referencing information
    - If sources disagree, mention the different viewpoints
    - If the answer isn't in the provided documents, say "I don't have enough information"
    - Be direct and informative

    Answer:"""
    a = enhance_query(prompt)
    a = a.strip()
    if a.endswith('"'):
        a = a[:-1]
    if a.startswith('"'):
        a = a[1:]
    return a

def ques(query , context):
    prompt = f"""Answer the user's question based on the provided movies that are available on Webflyx, a streaming service.

    Question: {query}

    Documents:
    {context}

    Instructions:
    - Answer questions directly and concisely
    - Be casual and conversational
    - Don't be cringe or hype-y
    - Talk like a normal person would in a chat conversation

    Answer:"""
    a = enhance_query(prompt)
    a = a.strip()
    if a.endswith('"'):
        a = a[:-1]
    if a.startswith('"'):
        a = a[1:]
    return a

def enhance_query(prompt):
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=120.0,
        max_retries=2,
    )
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=messages,
    )
    return response.choices[0].message.content
