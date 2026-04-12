import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import uvicorn

today = datetime.now().strftime("%B %d, %Y")

print("Code Started ... m ... v")

load_dotenv()
gen_api_key = os.getenv("gen_api_key")
links = os.getenv("links")


# ---------- Load PDF ----------
def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# ---------- Chunking ----------
def chunk_text(text, chunk_size=400):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

text = load_pdf('Madhu_Vasanth_Resume.pdf')
chunks = chunk_text(text)

# ---------- Embeddings ----------
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks).astype('float32')

# ---------- FAISS ----------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# ---------- Retrieval ----------
def retrieve(query, k=3):
    query_vector = model.encode([query]).astype('float32')
    distances, indices = index.search(query_vector, k)
    results = [chunks[i] for i in indices[0]]
    return results

# ---------- Gemini ----------
def generate_answer(question, context):
    prompt = f"""
You are Madhu’s AI Resume Assistant.

Your role is to help recruiters quickly evaluate the candidate using the resume.

Guidelines:

1. Interpret questions like a recruiter.
2. Infer answers from skills, projects, experience, and achievements.
3. Respond in a SHORT, clean, professional format.
4. Keep answers concise and easy to scan.
5. Use simple bullet points (•) and use number format when needed, not long paragraphs.
6. Limit to 3–6 key points unless more are necessary.
7. Avoid excessive technical listing unless specifically asked.
8. Do NOT use markdown symbols like ** or long decorative formatting.
9. Speak in third person.
10. Do not invent information not present in the resume.
11. Prioritize strengths, impact, and relevance.

If asked about strengths:
- Derive them from skills, projects, experience, and accomplishments
- Present as a concise list titled "Key Strengths"


If asked about experience , you should calculate the experience starting from the year mentioned in right edge of the resume to current date is {today}.
You can access the important links from {links}.

Stricltly, In Skills of Madhu Vasanth, Include this below skills:
GenAI,RAG, DSA

Strictly, In Projects of Madhu Vasanth , Include below :
RAG-Based AI Resume Evaluation Assistant (GenAI Project)

• Designed and implemented a Retrieval-Augmented Generation (RAG) system for intelligent resume analysis
• Built scalable REST APIs using FastAPI for document ingestion, embedding, and query handling
• Integrated LLM to perform contextual understanding, skill extraction, and experience assessment
• Developed interactive web interface using HTML, CSS, and JavaScript



Your goal is to produce recruiter-friendly summaries, not detailed reports.
                Context:
                {context}

                Question:
                {question}

                Answer: 
            """

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=gen_api_key,
        temperature=0.5,
    )

    response = llm.invoke(prompt)
    return response.content

# ---------- RAG Pipeline ----------
def rag_pipeline(query):
    retrieved_docs = retrieve(query)
    context = "\n\n".join(retrieved_docs)
    answer = generate_answer(query, context)
    return answer


# ---------- User Input --------------
# while True:
#     query = input("Ask Your Query : ")
#     if query=="exit":
#         break
#     print(rag_pipeline(query))


# ----------- Backend API ------------
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"]
)

class Query(BaseModel):
    message:str


@app.post("/chat")
def chat(data:Query):
    reply = rag_pipeline(data.message)
    return {"reply":reply}



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)

