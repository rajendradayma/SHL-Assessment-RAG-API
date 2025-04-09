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
        self.embeddings = self.model.encode(self.texts, show_progress_bar=True)

        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def query(self, user_query, top_k=10):
        query_embedding = self.model.encode([user_query])
        _, indices = self.index.search(query_embedding, top_k)
        results = self.df.iloc[indices[0]]
        formatted = []

        for _, row in results.iterrows():
            formatted.append({
                "url": row["URL"],
                "adaptive_support": row["Adaptive/IRT (y/n)"].strip().capitalize().replace("Y", "Yes").replace("N", "No"),
                "description": row["Description"],
                "duration": int(row["Assessment Length"]) if pd.notnull(row["Assessment Length"]) else 0,
                "remote_support": row["Remote Testing (y/n)"].strip().capitalize().replace("Y", "Yes").replace("N", "No"),
                "test_type": row["Test Type"].split(",") if pd.notnull(row["Test Type"]) else []
            })

        return formatted

# Load model
recommender = SHLRecommender("shl_catalog_detailed.csv")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' field"}), 400

    results = recommender.query(data["query"])
    return jsonify({"recommended_assessments": results})
