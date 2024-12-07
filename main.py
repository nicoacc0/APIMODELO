import os
from fastapi import *
from services.create_chunk import *
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from utils.create_embeddings import *
import torch
from bson import ObjectId
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_community.document_loaders import PyPDFLoader
from utils.database import *

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)



# Llama Tokenizer y Model

model_name = "Nico240/EduBot-Llama-3.2-3B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)


model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Función para crear el chatbot y guardar los datos en MongoDB


@app.post("/create_chatbot")
async def create_chatbot(user_id: str = Form(...), chat_id: str = Form(...), pdf_file: UploadFile = File(...)):
    if not pdf_file:
        raise HTTPException(status_code=400, detail="Se requiere un archivo PDF")

    # Guardar el archivo PDF temporalmente
    temp_pdf_path = f"./{pdf_file.filename}"
    try:
        with open(temp_pdf_path, "wb") as f:
            f.write(await pdf_file.read())

        # Leer y dividir el PDF en páginas
        loader = PyPDFLoader(temp_pdf_path)
        pages = loader.load()

        # Limpiar el contenido de cada página
        """for page in pages:
            page.page_content = limpiar_texto(page.page_content)"""

        # Crear chunks para este PDF
        chunks = await create_chatbot_with_chunks(
            user_id=user_id,
            chat_id=chat_id,
            pages=pages,
            source_name=pdf_file.filename
        )

        # Construir el response con los metadatos del PDF
        response = [
            {
                "id": None,
                "metadata": chunk["metadata"],
                "page_content": chunk["page_content"],
                "type": "Document"
            }
            for chunk in chunks
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el PDF: {str(e)}")

    finally:
        # Eliminar el archivo temporal
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

    return pdf_file



@app.post("/vote/")
async def vote(fragment_id: str, chat_id: str, vote: str, pregunta: str = None, respuesta: str = None):
    """
    Registrar un voto ('1' para upvotes o '2' para downvotes) para un fragmento específico.
    """
    if vote not in ["1", "2"]:
        raise HTTPException(status_code=400, detail="Voto inválido. Usa '1' para upvotes o '2' para downvotes.")

    # Determinar el tipo de voto
    upvotes_increment = 1 if vote == "1" else 0
    downvotes_increment = 1 if vote == "2" else 0

    # Buscar el fragmento en la base de datos
    record = faq_collection.find_one({"_id": ObjectId(fragment_id)})

    if record:
        # Si el fragmento ya existe, actualizar votos
        faq_collection.update_one(
            {"_id": ObjectId(fragment_id)},
            {
                "$inc": {
                    "upvotes": upvotes_increment,
                    "downvotes": downvotes_increment
                }
            }
        )
        return {"detail": "Voto actualizado correctamente."}
    
    # Si el fragmento no existe, crear un nuevo registro con el voto inicial
    faq_collection.insert_one({
        "_id": ObjectId(fragment_id),
        "chat_id": chat_id,
        "question": pregunta,
        "embedding": get_embedding(pregunta),
        "text": respuesta,
        "upvotes": upvotes_increment,
        "downvotes": downvotes_increment
    })

    return {"detail": "Voto registrado correctamente."}




@app.post("/search/")
async def search(chat_id: str, pregunta: str):
    question_embedding = get_embedding(pregunta)
    query = pregunta

    # Consulta preliminar en la colección de preguntas previas
    similar_faq_query = [
        {
            "$vectorSearch": {
                "index": "index_query",  # Índice vectorial para la colección FAQ
                "path": "embedding",     # Campo donde se almacenan los embeddings
                "queryVector": question_embedding,
                "numCandidates": 10,      # Buscamos la respuesta más similar
                "limit": 5,                    
            }
        },
        {
            "$match": {"chat_id": chat_id}
        },
        {
            "$addFields": {
                # Calcular un factor de relevancia basado en upvotes y downvotes
                "vote_score": {
                    "$subtract": [
                        {"$ifNull": ["$upvotes", 0]},  # Usar 0 si no hay upvotes
                        {"$ifNull": ["$downvotes", 0]}  # Usar 0 si no hay downvotes
                    ]
                }
            }
        },
        {
            "$addFields": {
                "combined_score": {
                    "$let": {
                        "vars": {
                            "total_votes": { "$add": ["$upvotes", "$downvotes"] },
                            "vote_delta": { "$subtract": ["$upvotes", "$downvotes"] }
                        },
                        "in": {
                            "$min": [
                                1.0,  
                                {
                                    "$multiply": [
                                        { "$meta": "vectorSearchScore" },
                                        {
                                            "$add": [
                                                1,
                                                {
                                                    "$multiply": [
                                                        0.05,
                                                        "$$vote_delta"
                                                    ]
                                                }
                                            ]
                                        }
                                    ]
                                }
                            ]
                        }
                    }
                }
            }
        },
        {
            "$project":{
                "question": 1,
                "text": 1,
                "upvotes": 1,
                "downvotes": 1,
                "vote_score": 1,
                "original_score": { "$meta": "vectorSearchScore" },
                "combined_score": 1,
                "_id" : 1
            }
        },
        {
            "$sort": { "combined_score": -1 }  # Ordenar por el nuevo score combinado
        }
    ]

    # Ejecutar búsqueda en la colección FAQ
    try:
        faq_result = list(faq_collection.aggregate(similar_faq_query))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching FAQ: {str(e)}")

    results = []
    # Si encontramos una coincidencia suficientemente relevante
    for doc in faq_result:
        if doc and doc.get("combined_score", 0) > 0.90:
            results.append({
                "id" : str(doc["_id"]),
                "text": doc["text"],
                "score": doc["combined_score"],
                "upvotes": doc.get("upvotes", 0),
                "downvotes": doc.get("downvotes", 0)
            })

    if results:
        return {"results": results, "fag": True}



    embedding = question_embedding
    vector_priority = 1
    text_priority = 0.7
    limit = 5
    overrequest_factor = 10
    numCandidates = limit * overrequest_factor


    # Filtrar por chat_id
    filter_by_chat_id = {
        "$match": {
            "chat_id": chat_id  # Filtrar los documentos por chat_id
        }
    }  
    # Fase de búsqueda vectorial
    vector_search = {
        "$vectorSearch": {
            "index": "rrf-vector-search",  # Tu índice vectorial
            "path": "embeddings",      # El campo de los embeddings
            "queryVector": embedding,      # El vector de consulta
            "numCandidates": numCandidates,          # Cuántos resultados considerar
            "limit": limit                 # Límite de resultados
        }
    }

    # Agrupar los resultados de la búsqueda vectorial
    make_array = {
        "$group": { "_id": None, "docs": {"$push": "$$ROOT"} }
    }

    # Desempaquetar los resultados y añadir un índice de rank
    add_rank = {
        "$unwind": { "path": "$docs", "includeArrayIndex": "rank" }
    }

    # Calcular el puntaje para la búsqueda vectorial
    def make_compute_score_doc(priority, score_field_name):
        return {
            "$addFields": {
                score_field_name: {
                    "$divide": [
                        1.0,
                        { "$add": ["$rank", priority, 1] }
                    ]
                }
            }
        }

    # Proyección de los resultados con puntajes
    def make_projection_doc(score_field_name):
        return {
            "$project": {
                score_field_name: 1,
                "_id": "$docs._id",
                "chat_id": "$docs.chat_id",
                "metadata": "$docs.metadata",
                "text": "$docs.text",
            }
        }

    # Búsqueda de texto completo
    text_search = {
        "$search": {
            "index": "rrf-full-text-search",  # Tu índice de texto completo
            "text": { "query": query, "path": "text" },
        }
    }

    # Limitar los resultados
    limit_results = {
        "$limit": limit
    }

    # Combinar los resultados de ambas búsquedas (vectorial y texto)
    combine_search_results = {
        "$group": {
            "_id": "$_id",
            "vs_score": {"$max": "$vs_score"},
            "fts_score": {"$max": "$fts_score"},
            "chat_id": {"$first": "$chat_id"},
            "text": {"$first": "$text"},
            "metadata": {"$first": "$metadata"}
        }
    }

    # Proyección de los resultados combinados con el puntaje total
    project_combined_results = {
        "$project": {
            "_id": 1,
            "chat_id": 1,
            "text": 1,
            "metadata": 1,
            "score": {
                "$let": {
                    "vars": {
                        "vs_score":  { "$ifNull": ["$vs_score", 0] },
                        "fts_score":  { "$ifNull": ["$fts_score", 0] }
                    },
                    "in": { "$add": ["$$vs_score", "$$fts_score"] }
                }
            }
        }
    }

    # Ordenar los resultados por el puntaje
    sort_results = {
        "$sort": { "score": -1 }
    }

    # Definir el pipeline de agregación
    pipeline = [
        vector_search,
        make_array,
        add_rank,
        make_compute_score_doc(vector_priority, "vs_score"),
        make_projection_doc("vs_score"),
        {
            "$unionWith": {
                "coll": "users",  # Tu colección
                "pipeline": [
                    text_search,
                    limit_results,
                    make_array,
                    add_rank,
                    make_compute_score_doc(text_priority, "fts_score"),
                    make_projection_doc("fts_score")
                ]
            }
        },
        filter_by_chat_id,
        combine_search_results,
        project_combined_results,
        sort_results,
        limit_results
    ]
    
    try:
        results = list(collection.aggregate(pipeline))  # Convierte el cursor a lista
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing search: {str(e)}")

    if not results:
        return {"results": [], "fag": False}


    # Generar pares pregunta-pasaje para el reranker
    pairs = [(pregunta, doc["text"]) for doc in results]


    ranking = []

    for doc in (results):
        ranking.append({
            "text": doc["text"],
            "metadata": doc["metadata"]["source"],
            "score": doc["score"]
        })


    # Devolver un diccionario que incluya la lista y otras propiedades
    return {"results": ranking, "fag": False}



@app.post("/generate_response/")
async def generate_response(chat_id: str, pregunta: str):

    resultados = await search(chat_id, pregunta)
    print(resultados)
    if not resultados["results"]:
        return {"respuesta": "Lo siento, no encontré información relevante en tus documentos."}

    if resultados.get("fag", True):
        response = {
            "id": resultados['results'][0]['id'],
            "question": pregunta,
            "answer": resultados['results'][0]['text']
        }
        return response

    # Crear prompt y generar respuesta
    fragmentos_json = "\n".join([
        f'{{"text": "{doc["text"]}","metadata":{doc["metadata"]}"}}' for doc in resultados["results"]
    ])

     # Preparar el contexto para el modelo
    contexto = f"Pregunta: {pregunta}\n\nFragmentos de documentos:\n{fragmentos_json}"
    print(contexto)
    system_prompt = f"""
    Eres un asistente de búsqueda y respuesta especializado. Sigue estas instrucciones ESTRICTAMENTE:

    PRINCIPIOS FUNDAMENTALES:
    1. Tu objetivo principal es proporcionar la respuesta MÁS PRECISA y RELEVANTE a la pregunta formulada.
    2. Usa EXCLUSIVAMENTE la información proporcionada en los fragmentos de texto.
    3. NO inventes, adicionar o supongas información que no esté explícitamente en los documentos.
    4. Si tienes múltiples fragmentos, ELIGE el o los más relevante pero NO mezcles información de diferentes documentos.

    INSTRUCCIONES DE RESPUESTA:
    - Responde DIRECTAMENTE a la pregunta planteada.
    - Si la información es insuficiente, indica claramente: "No puedo responder completamente con la información disponible".
    - Mantén la respuesta CONCISA y al GRANO.
    - Prioriza la CLARIDAD sobre la extensión.

    FORMATO DE RESPUESTA:
    - Usa un lenguaje profesional y directo.
    - Estructura la respuesta de manera lógica y comprensible.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": contexto}
    ]
    tokenized_chat = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )

    #tokenized_chat = tokenized_chat.to(device)

    outputs = model.generate(tokenized_chat, max_new_tokens=1024, temperature=0.6,do_sample=True,use_cache=True)
    respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant")[-1].strip()

        # Guardar la respuesta generada en la variable respuestas_guardadas
    objectid = str(ObjectId())  # Generamos un ObjectId para esta respuesta
    response = {
        "id": objectid,
        "question": pregunta,
        "answer": respuesta
    }

    
    return response

