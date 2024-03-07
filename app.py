import sys
import json
import requests
from semantic_text_splitter import HuggingFaceTextSplitter
from tokenizers import Tokenizer
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")


users_all_response = requests.get('http://localhost:5000/users')

if not users_all_response.ok:
    # Print an error message if the request failed
    print(f"Request failed with status code: {users_all_response.status_code}")
    sys.exit(1)

user_data = json.loads(users_all_response.text)
user_ids = [item['id'] for item in user_data]

for user in user_ids: 
    user_posts_response = requests.get(f"http://localhost:5000/users/{user}/posts")
    if not user_posts_response.ok:
        # Print an error message if the request failed
        print(f"Request failed with status code: {user_posts_response.status_code}")
        sys.exit(1)
    user_posts_data = json.loads(user_posts_response.text)
    posts = [[item['id'], item['title'], item['content']] for item in user_posts_data]
    for post  in posts: 
        id = post[0]
        title = post[1]
        content = post[2]
        max_tokens = 50
        # Optionally can also have the splitter not trim whitespace for you
        tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
        splitter = HuggingFaceTextSplitter(tokenizer, trim_chunks=False)

        chunks = splitter.chunks(content, max_tokens)
        for i, chunk in enumerate(chunks): 
            print(f"Index: {i}, Chunk: {chunk}")
            embedding = model.encode(chunk)
            chunk_data = { 
                "user_id": user,
                "post_id": id, 
                "chunk_number": i, 
                "text_chunk" : chunk, 
                "vector": embedding
              }
            if isinstance(chunk_data['vector'], np.ndarray):
                chunk_data['vector'] = chunk_data['vector'].tolist()
            post_chunk = requests.post(f"http://localhost:5000/chunk", json=chunk_data)
            if post_chunk.status_code != 201:
                print("Error: Bad request")
                sys.exit(1)