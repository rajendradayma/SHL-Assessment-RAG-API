from flask import Flask, request, jsonify
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

app = Flask(__name__)

class SHLRecommender:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.df.fillna("", inplace=True)
        self.texts = (
            self.df['Individual Test Solutions'] + ". " +
            self.df['Description'] + ". " +
            self.df['Job Levels'] + ". " +
            self.df['Test Type']
        ).tolist()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def query(self, user_query, top_k=5):
        embeddings = self.model.encode(self.texts, show_progress_bar=False)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        query_embedding = self.model.encode([user_query])
        _, indices = index.search(query_embedding, top_k)
        results = self.df.iloc[indices[0]].to_dict(orient="records")
        return results

recommender = SHLRecommender("shl_catalog_detailed.csv")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query'"}), 400

    results = recommender.query(data["query"])
    return jsonify(results)

