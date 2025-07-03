# vector_indexer.py

import os
import pickle
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

# Load cleaned dataset
df = pd.read_csv(r"C:\Users\Amenzz\Desktop\week-5\notebooks\data\filtered_complaints.csv")

# Ensure required columns exist
required_cols = {"complaint_id", "text", "category"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")

# 1. Text Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
)

chunks = []
metas = []

print("🔹 Splitting texts into chunks...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    complaint_id = row["complaint_id"]
    text = str(row["text"])
    category = row.get("category", "unknown")
    
    for chunk in splitter.split_text(text):
        chunks.append(chunk)
        metas.append({
            "complaint_id": complaint_id,
            "category": category,
            "original_text": text
        })

# 2. Embedding Model
print("🔹 Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("🔹 Generating embeddings...")
vectors = model.encode(chunks, show_progress_bar=True)

# 3. FAISS Indexing
print("🔹 Creating FAISS index...")
dimension = vectors[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(vectors)

# 4. Save index and metadata
os.makedirs("vector_store", exist_ok=True)
faiss.write_index(index, "vector_store/index.faiss")

with open("vector_store/metadata.pkl", "wb") as f:
    pickle.dump(metas, f)

print("✅ Vector store and metadata saved to 'vector_store/'")
