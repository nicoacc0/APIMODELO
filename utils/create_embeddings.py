from sentence_transformers import SentenceTransformer
import torch

embed_model = SentenceTransformer("BAAI/bge-m3")

# Función para generar el embedding
def get_embedding(data):
    embedding = embed_model.encode(data)  # Generar el embedding usando el modelo
    return embedding.tolist()  # Convertir el embedding a lista para almacenarlo en MongoDB
