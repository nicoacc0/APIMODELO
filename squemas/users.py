from pydantic import BaseModel, Field
from typing import List, Dict
from datetime import datetime
from fastapi import UploadFile
from bson import ObjectId




class Documento(BaseModel):
    id: str = Field(default_factory=lambda: str(ObjectId()))
    user_id: str 
    chat_id: str
    text: str
    embeddings: List[float]


class Texto(BaseModel):
    text: str

class ResponseRequest(BaseModel):
    question: str
    documents: List[Texto]


class QuestionRequest(BaseModel):
    chat_id: str
    question: str


class UserFront(BaseModel):
    id_user: str
    id_chatbot: str
    pdfs: Dict[str, UploadFile]