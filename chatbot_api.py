import json
import os
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = Flask(__name__)

# Load model once
model = SentenceTransformer('all-MiniLM-L6-v2')

# Cache loaded agents in memory for performance
agent_cache = {}

def load_agent_data(agent_code):
    filename = f"Agents/{agent_code}_Agent.json"
    if not os.path.exists(filename):
        return None, None

    # Return cached if already loaded
    if agent_code in agent_cache:
        return agent_cache[agent_code]

    with open(filename, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)

    questions = [q["question"] for q in qa_pairs]
    vectors = model.encode(questions)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors))

    agent_cache[agent_code] = (qa_pairs, index)
    return qa_pairs, index

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    print("Received payload:", data)

    agent = data.get("agent", "").strip().upper()
    question = data.get("question", "").strip()

    if not agent or not question:
        print("Missing agent or question")
        return jsonify({"error": "Missing 'agent' or 'question'"}), 400

    qa_pairs, index = load_agent_data(agent)
    if qa_pairs is None:
        return jsonify({"error": f"No data found for agent '{agent}'"}), 404

    input_vec = model.encode([question])
    D, I = index.search(np.array(input_vec), 1)

    distance = D[0][0]  # squared L2 distance (lower is better)
    threshold = 0.5     # tune this threshold as needed

    fallback_messages = {
        "MEDICALL": "Sorry, I couldn't find a good answer. If you like, I can give you our customer service number: 0764570398.",
        "QIC": "Sorry, I couldn't find a good answer. Please ask about personal accident insurance or contact support.",
        "CC": "Sorry, I couldn't find an answer. You can reach our support team for Alienware laptops at 123-456-7890."
    }

    if distance > threshold:
        fallback_answer = fallback_messages.get(agent, "Sorry, I didn't understand that.")
        return jsonify({"answer": fallback_answer})
    else:
        best_answer = qa_pairs[I[0][0]]["answer"]
        return jsonify({"answer": best_answer})

if __name__ == '__main__':
    app.run(port=3002)
