from flask import Blueprint, jsonify, request
from semantic_text_splitter import HuggingFaceTextSplitter
from tokenizers import Tokenizer
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

prompt_embedding = Blueprint('prompt_embedding', __name__)

@prompt_embedding.route('/prompt_embedding', methods=['POST'])
def embedding():
    data = request.get_json()
    text = data.get('text')
    max_tokens = data.get('max_tokens')
    try:
        max_tokens = int(max_tokens)
    except ValueError:
        return jsonify({'error': 'max_tokens must be an integer'}), 400

    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = HuggingFaceTextSplitter(tokenizer, trim_chunks=False)

    chunks = splitter.chunks(text, max_tokens)
    for i, chunk in enumerate(chunks): 
        embedding = model.encode(chunk)
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

    return jsonify({'embedding': embedding})