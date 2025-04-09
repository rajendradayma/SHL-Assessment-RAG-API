from flask import Flask, request, jsonify
import pickle
import faiss
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load the data
with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)

df = data["df"]
embeddings = data["embeddings"]
index = faiss.read_index("faiss.index")
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")  # Only used for 1 query


@app.route("/recommend", methods=["POST"])
def recommend():
    content = request.get_json()
    query = content.get("query", "")
    if not query:
        return jsonify({"error": "Query is required"}), 400

    query_vec = model.encode([query])
    _, results = index.search(query_vec, 5)
    matches = df.iloc[results[0]].to_dict(orient="records")
    return jsonify(matches)
