import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv('../.env')
client = OpenAI(base_url='https://api.groq.com/openai/v1', api_key=os.environ.get('GROQ_API_KEY'))

tests = [
    ('The Berenstain Bears Christmas Tree', 'family movie about bears in the woods'),
    ('Transformers', 'family movie about bears in the woods'),
    ('Paddington', 'family movie about bears in the woods'),
]
for title, query in tests:
    snippet = 'Bear family children animated'
    prompt = f'Rate 0-10: query="{query}" movie="{title} {snippet[:50]}". Number only.'
    r = client.chat.completions.create(model='allam-2-7b', messages=[{'role':'user','content':prompt}])
    raw = r.choices[0].message.content.strip()
    score = raw.split()[0].strip('"')
    print(f'{title}: {score}')
