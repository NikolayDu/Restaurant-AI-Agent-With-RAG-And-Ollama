from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("data/realistic_restaurant_reviews.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location) # Если нету базы - создадим её, а если есть - пропустим

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        document = Document(
            # page_content - основной текст документа.
            # - поиск по векторной базе
            # - семантическое сравнение
            # - embedding’и создаются именно из этого текста
            page_content=row["Title"] + " " + row["Review"],
            # metadata — вспомогательная информация
            # - не участвуют напрямую в поиске
            # - но хранятся вместе с документом
            # - могут использоваться для фильтрации
            # - могут возвращаться модели как дополнительный контекст
            metadata={"rating": row["Rating"], "date": row["Date"]},
            # id — уникальный идентификатор документа
            # - обновления записи
            # - удаления
            # - предотвращения дублей
            # - обращения к документу в векторной базе
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)
    
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5} # Ищется по 5 отзывам 
)
