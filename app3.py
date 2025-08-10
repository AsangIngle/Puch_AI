from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import tempfile
import fitz  # PyMuPDF
import docx
from keybert import KeyBERT
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="Career Chat AI",
    description="AI-powered resume analysis and company recommendation system",
    version="1.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock company dataset (Indian startups, AI-focused)
MOCK_COMPANIES = [
    {
        "name": "Sarvam AI",
        "industry": "AI",
        "keywords": ["Deep Learning", "Computer Vision", "NLP", "TensorFlow"],
        "linkedin_url": "https://linkedin.com/company/sarvam-ai",
        "about": "Indian AI startup building LLMs and computer vision solutions for enterprise applications."
    },
    {
        "name": "Razorpay",
        "industry": "FinTech",
        "keywords": ["Python", "AI", "Machine Learning", "TensorFlow"],
        "linkedin_url": "https://linkedin.com/company/razorpay",
        "about": "Leading payment gateway using AI for fraud detection and transaction analytics."
    },
    {
        "name": "Fashinza",
        "industry": "Fashion/E-Commerce",
        "keywords": ["AI", "Computer Vision", "Supply Chain", "Data Science"],
        "linkedin_url": "https://linkedin.com/company/fashinza",
        "about": "B2B manufacturing marketplace leveraging AI for supply chain optimization."
    }
]

# Initialize AI models
kw_model = KeyBERT()
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = OllamaLLM(model="gemma3")
embedding_cache = {}  # Cache for embeddings

# Extract text from resume
def extract_text(file_path):
    logger.info(f"Extracting text from {file_path}")
    start_time = time.time()
    if not file_path.endswith((".pdf", ".docx", ".txt")):
        raise ValueError("Unsupported file format. Use PDF, DOCX, or TXT.")
    text = ""
    if file_path.endswith(".pdf"):
        pdf = fitz.open(file_path)
        for page in pdf:
            text += page.get_text()
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    logger.info(f"Text extraction completed in {time.time() - start_time:.2f} seconds")
    return text.lower()

# Extract keywords using KeyBERT
def extract_keywords(text):
    try:
        logger.info("Extracting keywords with KeyBERT")
        start_time = time.time()
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            top_n=10  # Reduced for speed
        )
        logger.info(f"Keyword extraction completed in {time.time() - start_time:.2f} seconds")
        return [kw[0] for kw in keywords]
    except Exception as e:
        logger.error(f"Keyword extraction failed: {str(e)}")
        return ["python", "data analysis"]

# Cache embeddings
def get_cached_embedding(text):
    if text in embedding_cache:
        return embedding_cache[text]
    start_time = time.time()
    embedding = embeddings.embed_query(text)
    embedding_cache[text] = embedding
    logger.info(f"Embedding generated in {time.time() - start_time:.2f} seconds")
    return embedding

# Calculate semantic similarity score
def calculate_score(resume_keywords, company_keywords):
    if not company_keywords or not resume_keywords:
        return 0
    try:
        logger.info("Calculating semantic similarity")
        start_time = time.time()
        resume_text = " ".join(resume_keywords)
        company_text = " ".join(company_keywords)
        resume_emb = get_cached_embedding(resume_text)
        company_emb = get_cached_embedding(company_text)
        score = cosine_similarity([resume_emb], [company_emb])[0][0] * 100
        logger.info(f"Similarity calculation completed in {time.time() - start_time:.2f} seconds")
        return max(50, min(100, round(score)))
    except Exception as e:
        logger.error(f"Similarity calculation failed: {str(e)}")
        return 50

# Generate LLM explanation
def generate_explanation(resume_keywords, company):
    try:
        logger.info(f"Generating explanation for {company['name']}")
        start_time = time.time()
        prompt = ChatPromptTemplate.from_template(
            "Based on resume skills: {skills}\nCompany: {name}\nDescription: {desc}\nExplain why this company is a good match in one sentence."
        )
        formatted_prompt = prompt.format(skills=", ".join(resume_keywords), name=company["name"], desc=company["about"])
        response = llm.invoke(formatted_prompt)
        result = response.content if hasattr(response, "content") else str(response)
        logger.info(f"Explanation generation completed in {time.time() - start_time:.2f} seconds")
        return result
    except Exception as e:
        logger.error(f"Explanation generation failed: {str(e)}")
        return f"{company['name']} aligns with your skills."

# Suggest missing skills
def suggest_skills(resume_keywords, company_keywords):
    missing = list(set(company_keywords) - set(resume_keywords))
    return [{"skill": s, "resource": f"https://www.coursera.org/search?query={s}"} for s in missing[:3]]

# Search companies (mock data only)
def search_companies(query):
    logger.info(f"Processing query: {query}")
    start_time = time.time()
    companies = [c for c in MOCK_COMPANIES if query.lower() in c["about"].lower() or query.lower() in c["industry"].lower()]
    logger.info(f"Company search completed in {time.time() - start_time:.2f} seconds")
    return companies or MOCK_COMPANIES

# Upload resume endpoint
@app.post("/upload_resume")
async def upload_resume(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            tmp.write(await file.read())
            file_path = tmp.name
        text = extract_text(file_path)
        keywords = extract_keywords(text)
        os.unlink(file_path)
        return {"message": "Resume uploaded", "keywords": keywords}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Resume upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process resume.")

# Search companies endpoint
@app.get("/search_companies")
async def get_companies(query: str = Query(""), keywords: str = Query("")):
    start_time = time.time()
    try:
        keywords_list = keywords.split(",") if keywords else []
        search_query = query if query.strip() else "top companies hiring"
        companies = search_companies(search_query)[:3]  # Limit to 3 for speed
        
        if not companies:
            raise HTTPException(status_code=404, detail="No companies found.")
        
        for company in companies:
            company["score"] = calculate_score(keywords_list, company.get("keywords", []))
            company["explanation"] = generate_explanation(keywords_list, company) if keywords_list else "No resume provided."
            company["missing_skills"] = suggest_skills(keywords_list, company.get("keywords", [])) if keywords_list else []
        
        companies.sort(key=lambda x: x["score"], reverse=True)
        logger.info(f"Search completed in {time.time() - start_time:.2f} seconds")
        return companies
    except Exception as e:
        logger.error(f"Search companies failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# Career path suggestion endpoint
@app.get("/career_path")
async def get_career_path(keywords: str = Query("")):
    try:
        start_time = time.time()
        keywords_list = keywords.split(",") if keywords else []
        if not keywords_list:
            raise HTTPException(status_code=400, detail="No keywords provided.")
        prompt = ChatPromptTemplate.from_template(
            "Based on skills: {skills}, suggest a 3-step career path in one sentence."
        )
        formatted_prompt = prompt.format(skills=", ".join(keywords_list))
        response = llm.invoke(formatted_prompt)
        path = response.content if hasattr(response, "content") else str(response)
        logger.info(f"Career path generation completed in {time.time() - start_time:.2f} seconds")
        return {"path": path}
    except Exception as e:
        logger.error(f"Career path generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate career path: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)