import sys
import json
import requests
from semantic_text_splitter import HuggingFaceTextSplitter
from tokenizers import Tokenizer
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


users_all_response = requests.get('http://localhost:5000/users/all')

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
    for post in posts: 
        id = post[0]
        title = post[1]
        content = post[2]
        max_tokens = 1000
        # Optionally can also have the splitter not trim whitespace for you
        tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
        splitter = HuggingFaceTextSplitter(tokenizer, trim_chunks=False)

        chunks = splitter.chunks(content, max_tokens)
        embeddings = []        
        for chunk in chunks: 
            embedding = model.encode(chunk)
            embeddings.push(embedding)
            
            

