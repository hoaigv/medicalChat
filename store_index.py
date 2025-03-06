
from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from pinecone import Pinecone 
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')


# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


#Initializing the Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)


index = pc.Index("medicalchatbot")

#Creating Embeddings for Each of The Text Chunks & storing
batch_size = 100

# Chuyển text thành vector embeddings đúng cách
vectors = [
    {
        "id": str(i),
        "values": embeddings.embed_query(text_chunks[i].page_content),  # Lấy nội dung văn bản
        "metadata": {"text": text_chunks[i].page_content}  # Lưu metadata
    }
    for i in range(len(text_chunks))
]



# Upsert theo batch
for i in range(0, len(vectors), batch_size):
    batch = vectors[i:i + batch_size]  # Chia batch nhỏ
    index.upsert(vectors=batch)  # Gửi lên Pinecone