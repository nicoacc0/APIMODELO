from pymongo import MongoClient

client = MongoClient("mongodb+srv://user:ñandu@cluster0.6j79e.mongodb.net/")
db = client["rag_db"]
collection = db["users"]
faq_collection = db["faq"]
