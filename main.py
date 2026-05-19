from fastapi import FastAPI
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI()

class Summary(BaseModel):
    # HACK: store original text in db, take the id here
    original_text: str
    student_summary: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/check-summary")
def check_summary(summary: Summary):
    sentences = [
        summary.original_text,
        summary.student_summary
    ]

    # Load model
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    # Convert sentences to embeddings
    embeddings = model.encode(sentences)

    # Calculate cosine similarity
    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )

    score = float(similarity[0][0])
    return {"score": score}