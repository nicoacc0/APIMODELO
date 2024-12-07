from pymongo import MongoClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.create_embeddings import *
from utils.database import *
import re



# Función para limpiar el texto
def limpiar_texto(text):
    text = re.sub(r'\s+', ' ', text)  # Eliminar espacios adicionales
    text = re.sub('\uf0a7', '-', text)  # Reemplazar caracteres especiales
    text = re.sub(r'(\w)-\s*(\w)', r'\1\2', text)  # Unir palabras separadas por guiones y espacio (ej. "ext inguirá" -> "extinguirá")
    return text.strip()


async def create_chatbot_with_chunks(user_id: str, chat_id: str, pages, source_name: str):
    # Dividir el texto en chunks, evitando cortes en medio de palabras
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,  
        chunk_overlap=100)

        

    # Procesar cada página y sus chunks
    docs_to_insert = []
    chunks = []

    for page in pages:
        page_number = page.metadata.get("page", "unknown")  # Recuperar el número de página
        chunked_docs = text_splitter.split_text(page.page_content)
        for i, chunk in enumerate(chunked_docs):
            metadata = {
                "source": source_name,
                "page": page_number + 1,
                "chunk_index": i + 1
            }
            docs_to_insert.append({
                "user_id": user_id,
                "chat_id": chat_id,
                "text": limpiar_texto(chunk),
                "embeddings": get_embedding(chunk),
                "metadata": metadata
            })
            chunks.append({"page_content": chunk, "metadata": metadata})

    # Insertar en MongoDB
    collection.insert_many(docs_to_insert)

    return chunks